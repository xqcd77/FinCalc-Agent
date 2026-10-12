"""Train a manifest-sized verified dataset with explicit config; no silent truncation."""
import argparse
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from datasets import Dataset
from transformers import (AutoTokenizer, BitsAndBytesConfig,
    Qwen3_5ForConditionalGeneration, Trainer, TrainingArguments, TrainerCallback)
from transformers import AutoConfig
from accelerate import init_empty_weights
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb
from qlora_dataset_gate import validate_dataset_counts

def readl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]

def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def emit(path, event, **fields):
    record = dict(time=datetime.now(timezone.utc).isoformat(), event=event, **fields)
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(json.dumps(record, ensure_ascii=False), flush=True)

class Progress(TrainerCallback):
    def __init__(self, event_file): self.event_file = event_file
    def on_log(self, args, state, control, logs=None, **kwargs):
        emit(self.event_file, 'training_log', step=state.global_step,
             max_steps=state.max_steps, epoch=state.epoch, metrics=logs or {})
    def on_epoch_begin(self, args, state, control, **kwargs):
        emit(self.event_file, 'epoch_begin', step=state.global_step,
             max_steps=state.max_steps, epoch=state.epoch)
    def on_epoch_end(self, args, state, control, **kwargs):
        emit(self.event_file, 'epoch_end', step=state.global_step,
             max_steps=state.max_steps, epoch=state.epoch)
    def on_evaluate(self,args,state,control,metrics=None,**kwargs):
        values=dict(metrics or {})
        loss=values.get('eval_loss')
        if loss is not None and math.isfinite(loss):
            values['completion_perplexity']=math.exp(min(loss,700))
            values['completion_perplexity_clipped']=loss>700
        emit(self.event_file,'validation_metrics',step=state.global_step,epoch=state.epoch,
             metrics=values,scope='completion-only teacher-forced validation; not RAG answer accuracy')
    def on_save(self, args, state, control, **kwargs):
        emit(self.event_file, 'checkpoint_saved', step=state.global_step,
             max_steps=state.max_steps, checkpoint=str(Path(args.output_dir) / f'checkpoint-{state.global_step}'))

class PadCollator:
    def __init__(self, pad_id): self.pad_id = int(pad_id)
    def __call__(self, features):
        width = max(len(x['input_ids']) for x in features)
        ids = torch.full((len(features), width), self.pad_id, dtype=torch.long)
        labels = torch.full((len(features), width), -100, dtype=torch.long)
        mask = torch.zeros((len(features), width), dtype=torch.long)
        for i, x in enumerate(features):
            n = len(x['input_ids'])
            ids[i, :n] = torch.tensor(x['input_ids'], dtype=torch.long)
            labels[i, :n] = torch.tensor(x['labels'], dtype=torch.long)
            mask[i, :n] = 1
        return {'input_ids': ids, 'labels': labels, 'attention_mask': mask}

def tokenize_rows(rows, metadata, tokenizer, max_length, split, event_file):
    if len(rows) != len(metadata): raise ValueError(f'{split}: data/metadata length mismatch')
    encoded = []
    for index, (row, meta) in enumerate(zip(rows, metadata), start=1):
        prompt_text = tokenizer.apply_chat_template(row['prompt'], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        full_text = tokenizer.apply_chat_template(row['prompt'] + row['completion'], tokenize=False,
            add_generation_prompt=False, enable_thinking=False)
        if not full_text.startswith(prompt_text):
            raise ValueError(f'{split} {meta.get("sample_id", index)}: full chat is not prompt-prefixed')
        pids = tokenizer.encode(prompt_text, add_special_tokens=False)
        ids = tokenizer.encode(full_text, add_special_tokens=False)
        if ids[:len(pids)] != pids:
            raise ValueError(f'{split} {meta.get("sample_id", index)}: tokenized prompt/completion boundary differs')
        if len(ids) != meta['sequence_tokens']:
            raise ValueError(f'{split} {meta.get("sample_id", index)}: local/remote token count {meta["sequence_tokens"]}/{len(ids)}')
        if len(ids) > max_length:
            raise ValueError(f'{split} {meta.get("sample_id", index)}: {len(ids)} tokens exceeds {max_length}')
        labels = [-100] * len(pids) + ids[len(pids):]
        if not any(x != -100 for x in labels): raise ValueError(f'{split}: empty completion labels')
        encoded.append(dict(input_ids=ids, attention_mask=[1] * len(ids), labels=labels))
        if index % 100 == 0 or index == len(rows):
            emit(event_file, 'data_tokenized', split=split, completed=index, total=len(rows))
    return Dataset.from_list(encoded)

def make_targets(model_path):
    config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    with init_empty_weights():
        probe = Qwen3_5ForConditionalGeneration(config)
    names = [name for name, mod in probe.named_modules()
             if name.startswith('model.language_model.') and isinstance(mod, nn.Linear)]
    del probe
    if not names or any('.visual.' in name or name.startswith('visual.') for name in names):
        raise RuntimeError('Expected language-only Qwen3.5 linear targets; refusing to adapt the vision tower')
    return names

def load_model(model_path, target_modules, event_file, cfg):
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is unavailable')
    if not torch.cuda.is_bf16_supported(): raise RuntimeError('RTX 3090 BF16 compute not available')
    compute = torch.bfloat16
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute)
    emit(event_file, 'model_load_begin', model=model_path, quantization='NF4 + double quant', compute_dtype='bfloat16',
         gpu=torch.cuda.get_device_name(0), target_linear_modules=len(target_modules))
    model = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, local_files_only=True,
        trust_remote_code=True, quantization_config=quant, dtype=compute, device_map={'': 0})
    model.config.use_cache = False
    if getattr(model.config, 'text_config', None) is not None:
        model.config.text_config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False})
    peft_config = LoraConfig(r=cfg['lora_r'], lora_alpha=cfg['lora_alpha'], lora_dropout=cfg['lora_dropout'], bias='none',
        target_modules=target_modules, task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, peft_config)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or any('visual' in name for name, _ in trainable):
        raise RuntimeError('LoRA trainable parameters are missing or include the vision tower')
    if not all('lora_' in name and 'language_model' in name for name, _ in trainable):
        raise RuntimeError('Unexpected trainable base parameters or adapters outside the language model')
    model.print_trainable_parameters()
    torch.cuda.synchronize()
    emit(event_file, 'model_loaded', total_parameters=sum(p.numel() for p in model.parameters()),
         trainable_parameters=sum(p.numel() for _, p in trainable),
         gpu_allocated_gib=round(torch.cuda.memory_allocated() / 1024**3, 2),
         gpu_reserved_gib=round(torch.cuda.memory_reserved() / 1024**3, 2))
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-path', required=True)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--config', required=True, help='dataset training_config.json')
    ap.add_argument('--epochs', type=int, help='explicit run-level epoch override; dataset config remains immutable')
    ap.add_argument('--smoke', action='store_true', help='one forward/backward/optimizer step only')
    args = ap.parse_args()
    data_dir, output = Path(args.data_dir), Path(args.output_dir)
    cfg=json.loads(Path(args.config).read_text(encoding='utf-8'))
    expected={'packing':False,'completion_only_loss':True,'gradient_checkpointing':True,'enable_thinking':False,'quantization':'nf4','double_quantization':True,'compute_dtype':'bfloat16'}
    for key,value in expected.items():
        if cfg.get(key)!=value:raise ValueError('Unsupported training contract: '+key)
    verification_path=data_dir/'verification.json'
    if not verification_path.exists() or json.loads(verification_path.read_text(encoding='utf-8')).get('status')!='passed':raise ValueError('Dataset verification must pass first')
    import hashlib
    manifest=json.loads((data_dir/'manifest.json').read_text(encoding='utf-8'))
    verified=json.loads(verification_path.read_text(encoding='utf-8'))
    if verified['manifest_sha256']!=hashlib.sha256((data_dir/'manifest.json').read_bytes()).hexdigest():raise ValueError('Verification is stale; rerun dataset verifier')
    if cfg!=json.loads((data_dir/'training_config.json').read_text(encoding='utf-8')):raise ValueError('CLI config differs from verified dataset config; verify a new dataset/config version first')
    if args.epochs is not None:
        if args.epochs<=0:raise ValueError('--epochs must be positive')
        cfg['epochs']=args.epochs
    validate_dataset_counts(manifest,verified,len(readl(data_dir/'train.jsonl')),
        len(readl(data_dir/'validation.jsonl')),require_individual=bool(cfg.get('evidence_representation_version')))
    for name,digest in manifest['outputs'].items():
        if hashlib.sha256((data_dir/name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed dataset file: '+name)
    if cfg.get('evidence_representation_version'):
        project_root=Path(__file__).resolve().parents[1]
        if manifest.get('evidence_representation_version')!=cfg['evidence_representation_version']:
            raise ValueError('Training/runtime evidence contract mismatch')
        for relative,digest in manifest.get('runtime_hashes',{}).items():
            # Dataset manifests may be created on Windows but training runs on Linux.
            path=project_root/relative.replace('\\','/')
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                raise ValueError('Runtime code differs from audited training dataset: '+relative)
    output.mkdir(parents=True, exist_ok=True)
    event_file = output / 'events.jsonl'
    write_json(output / 'run_config.json',dict(effective_training_config=cfg,
        epoch_override=args.epochs,dataset_manifest_sha256=verified['manifest_sha256'],
        dataset=data_dir.name,smoke=args.smoke))
    emit(event_file, 'run_begin', mode='smoke' if args.smoke else 'full',
         train_file=str(data_dir / 'train.jsonl'), validation_file=str(data_dir / 'validation.jsonl'))
    torch.set_num_threads(4)
    if not torch.cuda.is_bf16_supported(): raise RuntimeError('BF16 unsupported on this GPU')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    train_rows = readl(data_dir / 'train.jsonl')
    train_meta = readl(data_dir / 'train.metadata.jsonl')
    val_rows = readl(data_dir / 'validation.jsonl')
    val_meta = readl(data_dir / 'validation.metadata.jsonl')
    train_data = tokenize_rows(train_rows, train_meta, tokenizer, cfg['max_seq_len'], 'train', event_file)
    val_data = tokenize_rows(val_rows, val_meta, tokenizer, cfg['max_seq_len'], 'validation', event_file)
    targets = make_targets(args.model_path)
    write_json(output / 'target_modules.json', targets)
    model = load_model(args.model_path, targets, event_file, cfg)
    collator = PadCollator(tokenizer.pad_token_id)
    if args.smoke:
        sample = collator([max(train_data, key=lambda row:len(row['input_ids']))])
        sample = {k: v.to('cuda') for k, v in sample.items()}
        model.train()
        optimizer = bnb.optim.PagedAdamW8bit([p for p in model.parameters() if p.requires_grad], lr=cfg['learning_rate'])
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        out = model(**sample)
        loss = out.loss
        if not torch.isfinite(loss): raise RuntimeError(f'nonfinite smoke loss: {loss.item()}')
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        emit(event_file, 'smoke_step_passed', loss=float(loss.detach().cpu()),
             seconds=round(time.monotonic() - start, 2),
             peak_gpu_gib=round(torch.cuda.max_memory_allocated() / 1024**3, 2))
        optimizer.zero_grad(set_to_none=True)
        return
    training_args = TrainingArguments(
        output_dir=str(output / 'checkpoints'), num_train_epochs=cfg['epochs'],
        per_device_train_batch_size=cfg['micro_batch_size'], per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg['gradient_accumulation_steps'], learning_rate=cfg['learning_rate'],
        lr_scheduler_type=cfg['lr_scheduler_type'], warmup_ratio=cfg['warmup_ratio'], weight_decay=cfg['weight_decay'],
        max_grad_norm=cfg['max_grad_norm'], bf16=True, fp16=False, tf32=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        optim=cfg['optimizer'], eval_strategy='epoch', save_strategy='epoch',
        save_total_limit=3, load_best_model_at_end=True, metric_for_best_model='eval_loss',
        greater_is_better=False, logging_strategy='steps', logging_steps=1,
        logging_first_step=True, report_to='none', seed=cfg['seed'], data_seed=cfg['seed'],
        dataloader_num_workers=2, dataloader_pin_memory=True,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_data,
        eval_dataset=val_data, data_collator=collator, processing_class=tokenizer,
        callbacks=[Progress(event_file)])
    expected_steps = math.ceil(len(train_data) / (cfg['micro_batch_size'] * cfg['gradient_accumulation_steps'])) * cfg['epochs']
    emit(event_file, 'training_begin', train_examples=len(train_data), validation_examples=len(val_data),
         epochs=cfg['epochs'], optimizer_steps=expected_steps,
         micro_batch=cfg['micro_batch_size'], gradient_accumulation=cfg['gradient_accumulation_steps'], max_length=cfg['max_seq_len'], learning_rate=cfg['learning_rate'],
         warmup_ratio=cfg['warmup_ratio'], lora_r=cfg['lora_r'], lora_alpha=cfg['lora_alpha'], lora_dropout=cfg['lora_dropout'],
         target_linear_modules=len(targets), quant='nf4_double_quant', completion_only_loss=True)
    result = trainer.train()
    trainer.save_model(str(output / 'adapter'))
    tokenizer.save_pretrained(output / 'adapter')
    train_metrics = dict(result.metrics)
    trainer.log_metrics('train', train_metrics)
    trainer.save_metrics('train', train_metrics)
    eval_metrics = trainer.evaluate()
    if math.isfinite(eval_metrics.get('eval_loss',float('nan'))):
        eval_metrics['completion_perplexity']=math.exp(min(eval_metrics['eval_loss'],700))
        eval_metrics['completion_perplexity_clipped']=eval_metrics['eval_loss']>700
    trainer.log_metrics('eval', eval_metrics)
    trainer.save_metrics('eval', eval_metrics)
    write_json(output/'evaluation_scope.json',dict(
        training_metrics=['train_loss','eval_loss','completion_perplexity','learning_rate','epoch','peak_gpu_gib'],
        loss_scope='completion-only teacher-forced token prediction; not financial answer accuracy',
        downstream_evaluator='src/eval/evaluate_v48.py',
        downstream_requires='Serve each saved adapter checkpoint, run src/rag/agent_v4.py with identical model precision and contexts. Generation automatically invokes the offline evaluator.',
        regression_set='163 development questions; not an untouched held-out benchmark'))
    torch.cuda.synchronize()
    emit(event_file, 'training_complete', global_step=trainer.state.global_step,
         train_metrics=train_metrics, eval_metrics=eval_metrics,
         peak_gpu_gib=round(torch.cuda.max_memory_allocated() / 1024**3, 2),
         adapter_dir=str(output / 'adapter'))

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        p = Path(os.environ.get('QLORA_EVENT_FILE', '/tmp/qlora_v1_events.jsonl'))
        try: emit(p, 'fatal_error', error=repr(exc), traceback=traceback.format_exc())
        except Exception: pass
        raise

