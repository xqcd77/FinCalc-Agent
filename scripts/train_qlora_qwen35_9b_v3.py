"""Memory-optimized single-GPU QLoRA, with immutable data and bounded overrides.

Use --smoke for verification only. A future expert-only dataset must have its
own verified manifest/config before it can be used for full training.
"""
import argparse
import hashlib
import json
import math
import time
import traceback
from pathlib import Path

import torch
import bitsandbytes as bnb
from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration, Trainer, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from train_qlora_qwen35_9b_v2 import readl, write_json, emit, Progress, PadCollator, tokenize_rows, make_targets
from qlora_dataset_gate import validate_dataset_counts
from qlora_run_profile import merge_profile
from qlora_memory_optimizations import (parameter_memory, restore_frozen_large_matrices_bf16,
    optimized_model_loss, require_fast_linear_kernels)

ROOT = Path(__file__).resolve().parents[1]
RUN_EVENTS = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified_config(data, profile_path):
    base = json.loads((data/'training_config.json').read_text(encoding='utf-8'))
    profile = json.loads(profile_path.read_text(encoding='utf-8'))
    declared_base = (ROOT/profile['base_config']).resolve()
    if declared_base != (data/'training_config.json').resolve():
        raise ValueError('Profile belongs to a different dataset config')
    cfg = merge_profile(base, profile)
    expected = {'packing':False, 'completion_only_loss':True, 'gradient_checkpointing':True,
                'enable_thinking':False, 'quantization':'nf4', 'double_quantization':True, 'compute_dtype':'bfloat16'}
    if any(cfg.get(k)!=v for k,v in expected.items()):
        raise ValueError('Unsupported training contract')
    manifest = json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    verification = json.loads((data/'verification.json').read_text(encoding='utf-8'))
    if verification.get('status')!='passed' or verification['manifest_sha256']!=sha(data/'manifest.json'):
        raise ValueError('Dataset verification missing or stale')
    for name,digest in manifest['outputs'].items():
        if sha(data/name)!=digest: raise ValueError('Changed audited dataset file: '+name)
    if manifest.get('evidence_representation_version')!=cfg.get('evidence_representation_version'):
        raise ValueError('Evidence contract mismatch')
    review_path = ROOT/profile['runtime_review']
    review = json.loads(review_path.read_text(encoding='utf-8'))
    if review.get('dataset_manifest_sha256')!=verification['manifest_sha256']:
        raise ValueError('Runtime review belongs to another dataset')
    reviewed = review.get('runtime_files', {})
    if set(reviewed)!=set(manifest.get('runtime_hashes',{})):
        raise ValueError('Runtime review must cover every original audited file')
    for relative,original in manifest['runtime_hashes'].items():
        entry=reviewed[relative]
        if entry['original_sha256']!=original or sha(ROOT/relative.replace('\\','/'))!=entry['approved_sha256']:
            raise ValueError('Unreviewed runtime change: '+relative)
        if entry['approved_sha256']!=original and not entry.get('reason'):
            raise ValueError('Changed runtime needs an explicit review reason')
    lineage=dict(dataset_manifest_sha256=verification['manifest_sha256'],
                 profile_sha256=sha(profile_path), runtime_review_sha256=sha(review_path),
                 original_training_config=base, effective_training_config=cfg,
                 optimizations=profile['memory_optimizations'],
                 entrypoint_sha256=sha(__file__), helper_sha256={
                     p:sha(ROOT/'scripts'/p) for p in ['qlora_memory_optimizations.py','qlora_run_profile.py']})
    return cfg, profile['memory_optimizations'], manifest, verification, lineage


def load_optimized_model(model_path, targets, cfg, events, output):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA/BF16 GPU required; CPU installation is not GPU verification')
    audit={}
    quant=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model=Qwen3_5ForConditionalGeneration.from_pretrained(model_path, local_files_only=True,
        trust_remote_code=True, quantization_config=quant, dtype=torch.bfloat16, device_map={'':0})
    model.config.use_cache=False
    model.config.text_config.use_cache=False
    audit['after_quantized_load']=parameter_memory(model)
    model=prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False})
    audit['after_peft_prepare']=parameter_memory(model)
    audit['bf16_restore']=restore_frozen_large_matrices_bf16(model)
    audit['after_bf16_restore']=parameter_memory(model)
    # Do not cast normalization parameters, A_log, quantized weights or adapters.
    model=get_peft_model(model, LoraConfig(r=cfg['lora_r'], lora_alpha=cfg['lora_alpha'],
        lora_dropout=cfg['lora_dropout'], bias='none', target_modules=targets, task_type=TaskType.CAUSAL_LM))
    trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if not trainable or any('lora_' not in n or 'language_model' not in n or 'visual' in n for n,p in trainable):
        raise RuntimeError('Unexpected trainable parameters')
    audit['trainable_parameters']=sum(p.numel() for n,p in trainable)
    audit['fast_linear_layers']=require_fast_linear_kernels(model)
    audit['after_adapters']=parameter_memory(model)
    write_json(output/'model_memory_audit.json', audit)
    torch.cuda.synchronize()
    emit(events,'model_loaded',trainable_parameters=audit['trainable_parameters'],
         target_linear_modules=len(targets), fast_linear_layers=len(audit['fast_linear_layers']),
         frozen_storage_saved_gib=(audit['after_peft_prepare']['storage_bytes']-audit['after_bf16_restore']['storage_bytes'])/1024**3,
         gpu_allocated_gib=torch.cuda.memory_allocated()/1024**3,
         gpu_reserved_gib=torch.cuda.memory_reserved()/1024**3)
    return model


class CompletionProjectionTrainer(Trainer):
    def __init__(self,*args,projection_chunk_tokens=64,**kwargs):
        super().__init__(*args,**kwargs)
        self.projection_chunk_tokens=projection_chunk_tokens
        # Match previous per-microbatch mean CE / accumulation semantics.
        self.model_accepts_loss_kwargs=False

    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        # Accelerator normally autocasts model.forward. We bypass that wrapper
        # to avoid full logits, so explicitly preserve the BF16 compute context.
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=torch.cuda.is_available()):
            loss,stats=optimized_model_loss(model,inputs,self.projection_chunk_tokens)
        return (loss,{'loss':loss}) if return_outputs else loss


def smoke(model,train_data,val_data,collator,cfg,optim,events):
    optimizer=bnb.optim.PagedAdamW8bit([p for p in model.parameters() if p.requires_grad], lr=cfg['learning_rate'],
        weight_decay=cfg['weight_decay'])
    batch=collator([max(train_data,key=lambda x:len(x['input_ids']))])
    batch={k:v.to('cuda') for k,v in batch.items()}
    model.train(); optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(); start=time.monotonic()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        loss,stats=optimized_model_loss(model,batch,optim['projection_chunk_tokens'])
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite smoke loss')
    loss.backward()
    grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not grads or not all(torch.isfinite(g).all().item() for g in grads):raise RuntimeError('Missing/nonfinite adapter gradients')
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],cfg['max_grad_norm'])
    optimizer.step(); torch.cuda.synchronize()
    emit(events,'smoke_step_passed',loss=loss.item(),seconds=time.monotonic()-start,stats=stats,
         sequence_tokens=batch['input_ids'].shape[1],peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3,
         reserved_gpu_gib=torch.cuda.max_memory_reserved()/1024**3)
    optimizer.zero_grad(set_to_none=True); del loss,batch,grads,optimizer
    model.eval()
    batch={k:v.to('cuda') for k,v in collator([max(val_data,key=lambda x:len(x['input_ids']))]).items()}
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        val_loss,stats=optimized_model_loss(model,batch,optim['projection_chunk_tokens'])
    if not torch.isfinite(val_loss):raise RuntimeError('Nonfinite validation smoke loss')
    emit(events,'smoke_validation_passed',loss=val_loss.item(),stats=stats,
         peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3)


def main():
    global RUN_EVENTS
    ap=argparse.ArgumentParser()
    for flag in ['model-path','data-dir','output-dir','profile']:ap.add_argument('--'+flag,required=True)
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args(); data=Path(args.data_dir); output=Path(args.output_dir)
    cfg,optim,manifest,verified,lineage=verified_config(data,Path(args.profile))
    train_rows=readl(data/'train.jsonl'); val_rows=readl(data/'validation.jsonl')
    validate_dataset_counts(manifest,verified,len(train_rows),len(val_rows),require_individual=True)
    if output.exists() and (output/'events.jsonl').exists():raise ValueError('Use a fresh output directory; do not overwrite an earlier run')
    output.mkdir(parents=True,exist_ok=True); events=output/'events.jsonl'
    RUN_EVENTS=events
    write_json(output/'run_config.json',dict(**lineage,smoke=args.smoke,dataset=data.name))
    emit(events,'run_begin',mode='smoke' if args.smoke else 'full',config=cfg,optimizations=optim)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(4); torch.manual_seed(cfg['seed']); torch.cuda.manual_seed_all(cfg['seed'])
    tokenizer=AutoTokenizer.from_pretrained(args.model_path,local_files_only=True,trust_remote_code=True)
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    train_data=tokenize_rows(train_rows,readl(data/'train.metadata.jsonl'),tokenizer,cfg['max_seq_len'],'train',events)
    val_data=tokenize_rows(val_rows,readl(data/'validation.metadata.jsonl'),tokenizer,cfg['max_seq_len'],'validation',events)
    targets=make_targets(args.model_path); write_json(output/'target_modules.json',targets)
    model=load_optimized_model(args.model_path,targets,cfg,events,output)
    collator=PadCollator(tokenizer.pad_token_id)
    if args.smoke:
        smoke(model,train_data,val_data,collator,cfg,optim,events)
        emit(events,'smoke_complete',full_training_started=False); return
    ta=TrainingArguments(output_dir=str(output/'checkpoints'),num_train_epochs=cfg['epochs'],
        per_device_train_batch_size=cfg['micro_batch_size'],per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg['gradient_accumulation_steps'],learning_rate=cfg['learning_rate'],
        lr_scheduler_type=cfg['lr_scheduler_type'],warmup_ratio=cfg['warmup_ratio'],weight_decay=cfg['weight_decay'],
        max_grad_norm=cfg['max_grad_norm'],bf16=True,fp16=False,tf32=True,
        gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        optim=cfg['optimizer'],eval_strategy='epoch',save_strategy='epoch',save_total_limit=3,
        load_best_model_at_end=True,metric_for_best_model='eval_loss',greater_is_better=False,
        logging_strategy='steps',logging_steps=1,logging_first_step=True,report_to='none',
        seed=cfg['seed'],data_seed=cfg['seed'],dataloader_num_workers=2,dataloader_pin_memory=True,
        remove_unused_columns=False,prediction_loss_only=True)
    trainer=CompletionProjectionTrainer(model=model,args=ta,train_dataset=train_data,eval_dataset=val_data,
        data_collator=collator,processing_class=tokenizer,callbacks=[Progress(events)],
        projection_chunk_tokens=optim['projection_chunk_tokens'])
    emit(events,'training_begin',train_examples=len(train_data),validation_examples=len(val_data),
         optimizer_steps=math.ceil(len(train_data)/(cfg['micro_batch_size']*cfg['gradient_accumulation_steps']))*cfg['epochs'])
    result=trainer.train(); trainer.save_model(str(output/'adapter')); tokenizer.save_pretrained(output/'adapter')
    trainer.save_metrics('train',result.metrics); metrics=trainer.evaluate()
    if math.isfinite(metrics.get('eval_loss',float('nan'))):metrics['completion_perplexity']=math.exp(min(metrics['eval_loss'],700))
    trainer.save_metrics('eval',metrics)
    emit(events,'training_complete',global_step=trainer.state.global_step,train_metrics=result.metrics,
         eval_metrics=metrics,peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3,
         adapter_dir=str(output/'adapter'))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if RUN_EVENTS is not None:
            emit(RUN_EVENTS,'fatal_error',error=repr(exc),traceback=traceback.format_exc())
        raise
