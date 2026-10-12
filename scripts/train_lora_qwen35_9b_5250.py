"""Train Qwen3.5-9B with full-BF16-base LoRA on an audited stage-aligned SFT set."""
import argparse
import hashlib
import json
import math
import traceback
from collections import Counter
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    Qwen3_5ForConditionalGeneration,
    TrainingArguments,
)

from train_qlora_qwen35_9b_v2 import (
    PadCollator,
    Progress,
    emit,
    make_targets,
    readl,
    tokenize_rows,
    write_json,
)
from train_qlora_qwen35_9b_v3 import CompletionProjectionTrainer


ROOT = Path(__file__).resolve().parents[1]
EVENTS = None
PROFILE_REL = "configs/lora.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_and_validate_dataset(data_dir, profile):
    manifest_path = data_dir / "manifest.json"
    verified_path = data_dir / "verified.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verified = json.loads(verified_path.read_text(encoding="utf-8"))
    if manifest.get("version") != profile.get("dataset_manifest_version", "three_batch_cleaned_v1"):
        raise ValueError("Unexpected active dataset manifest version")
    if verified.get("status") != "passed" or verified.get("failed") != 0:
        raise ValueError("The cleaned dataset verification did not pass")
    expected_total = profile["expected_train_examples"] + profile["expected_validation_examples"]
    if verified.get("total_samples") != expected_total or verified.get("passed") != expected_total:
        raise ValueError("Verification counts do not match this run profile")
    if manifest.get("counts", {}).get("total") != expected_total:
        raise ValueError("Manifest total does not match this run profile")
    if manifest.get("counts", {}).get("splits") != {
        "train": profile["expected_train_examples"],
        "validation": profile["expected_validation_examples"],
    }:
        raise ValueError("Manifest train/validation counts do not match this run profile")

    for filename, expected_digest in manifest.get("outputs_sha256", {}).items():
        path = data_dir / filename
        if not path.is_file() or sha256(path) != expected_digest:
            raise ValueError(f"Active dataset checksum mismatch: {filename}")

    train_rows = readl(data_dir / "train.jsonl")
    train_meta = readl(data_dir / "train.metadata.jsonl")
    validation_rows = readl(data_dir / "validation.jsonl")
    validation_meta = readl(data_dir / "validation.metadata.jsonl")
    expected_train = profile["expected_train_examples"]
    expected_validation = profile["expected_validation_examples"]
    if (len(train_rows), len(train_meta)) != (expected_train, expected_train):
        raise ValueError("Train row/metadata counts do not match the run profile")
    if (len(validation_rows), len(validation_meta)) != (expected_validation, expected_validation):
        raise ValueError("Validation row/metadata counts do not match the run profile")

    seen_ids = set()
    seen_questions = set()
    train_groups = set()
    validation_groups = set()
    source_train = Counter()
    source_validation = Counter()
    for split, rows, metadata in (
        ("train", train_rows, train_meta),
        ("validation", validation_rows, validation_meta),
    ):
        for index, (row, meta) in enumerate(zip(rows, metadata), start=1):
            qid = meta.get("question_id")
            if not qid or qid in seen_ids:
                raise ValueError(f"{split} row {index}: missing or duplicate question_id")
            seen_ids.add(qid)
            if meta.get("split") != split:
                raise ValueError(f"{qid}: metadata split mismatch")
            if meta.get("training_eligible") is not True:
                raise ValueError(f"{qid}: sample is not marked training_eligible")
            if (meta.get("original_eval_content_used") or meta.get("evaluation_overlap_intentional")) and not profile.get("allow_evaluation_overlap", False):
                raise ValueError(f"{qid}: evaluation-derived supervision requires explicit run-profile authorization")
            # The three audited batches use two names for the same no-eval-leakage flag.
            eval_sent = meta.get("eval_content_sent", meta.get("original_eval_content_sent"))
            if eval_sent is not False:
                raise ValueError(f"{qid}: missing/unsafe evaluation-content leakage flag")
            if meta.get("one_question_one_sample") is not True:
                raise ValueError(f"{qid}: one-question-one-sample gate failed")
            if not isinstance(row.get("prompt"), list) or not row["prompt"]:
                raise ValueError(f"{qid}: missing prompt messages")
            if not isinstance(row.get("completion"), list) or not row["completion"]:
                raise ValueError(f"{qid}: missing completion messages")

            user_text = "\n".join(
                str(message.get("content", ""))
                for message in row["prompt"]
                if message.get("role") in ("user", "human")
            )
            if not user_text.strip():
                raise ValueError(f"{qid}: prompt has no user content")
            question_hash = hashlib.sha256(
                " ".join(user_text.split()).casefold().encode("utf-8")
            ).hexdigest()
            if question_hash in seen_questions:
                raise ValueError(f"{qid}: duplicate normalized user question")
            seen_questions.add(question_hash)

            group = meta.get("company_group")
            if group:
                (train_groups if split == "train" else validation_groups).add(group)
            source = meta.get("source_batch", "unknown")
            (source_train if split == "train" else source_validation)[source] += 1

    if train_groups & validation_groups:
        raise ValueError("Train/validation company-group overlap detected")
    if len(seen_ids) != expected_total or len(seen_questions) != expected_total:
        raise ValueError("Unique ID/question count differs from the expected total")
    expected_val_sources = profile.get("source_validation_counts", {})
    if dict(source_validation) != expected_val_sources:
        raise ValueError("Validation source composition does not match the reviewed split")

    return {
        "manifest": manifest,
        "verified": verified,
        "train_rows": train_rows,
        "train_meta": train_meta,
        "validation_rows": validation_rows,
        "validation_meta": validation_meta,
        "source_train": dict(source_train),
        "source_validation": dict(source_validation),
        "dataset_manifest_sha256": sha256(manifest_path),
        "dataset_verified_sha256": sha256(verified_path),
    }


def add_exact_sequence_lengths(rows, metadata, tokenizer, max_length, split, events):
    if len(rows) != len(metadata):
        raise ValueError(f"{split}: data/metadata row counts differ")
    maximum = 0
    for index, (row, meta) in enumerate(zip(rows, metadata), start=1):
        prompt_text = tokenizer.apply_chat_template(
            row["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        full_text = tokenizer.apply_chat_template(
            row["prompt"] + row["completion"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        if not full_text.startswith(prompt_text):
            raise ValueError(f"{split} {meta.get('question_id', index)}: prompt boundary mismatch")
        token_count = len(tokenizer.encode(full_text, add_special_tokens=False))
        if token_count > max_length:
            raise ValueError(
                f"{split} {meta.get('question_id', index)}: {token_count} tokens exceeds {max_length}"
            )
        meta["sequence_tokens"] = token_count
        maximum = max(maximum, token_count)
        if index % 100 == 0 or index == len(rows):
            emit(events, "exact_sequence_lengths_computed", split=split, completed=index, total=len(rows))
    return maximum


def main():
    global EVENTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=str(ROOT / PROFILE_REL))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
    args_cli = parser.parse_args()

    profile_path = Path(args_cli.profile)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    data_dir = ROOT / profile["dataset"]
    model_path = Path(profile["model_path"])
    config = dict(profile["overrides"])
    dataset = load_and_validate_dataset(data_dir, profile)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    log_dir = ROOT / profile.get("log_dir", "logs/lora_5250_r32_e5_20261010")
    log_dir.mkdir(parents=True, exist_ok=True)
    preflight_events = log_dir / "preflight_events.jsonl"
    max_train_tokens = add_exact_sequence_lengths(
        dataset["train_rows"], dataset["train_meta"], tokenizer,
        config["max_seq_len"], "train", preflight_events,
    )
    max_validation_tokens = add_exact_sequence_lengths(
        dataset["validation_rows"], dataset["validation_meta"], tokenizer,
        config["max_seq_len"], "validation", preflight_events,
    )
    train_data = tokenize_rows(
        dataset["train_rows"], dataset["train_meta"], tokenizer,
        config["max_seq_len"], "train", preflight_events,
    )
    validation_data = tokenize_rows(
        dataset["validation_rows"], dataset["validation_meta"], tokenizer,
        config["max_seq_len"], "validation", preflight_events,
    )
    targets = make_targets(str(model_path))
    if len(targets) != profile["target_linear_modules_expected"]:
        raise ValueError(
            f"Expected {profile['target_linear_modules_expected']} LoRA targets, got {len(targets)}"
        )

    preflight_summary = {
        "status": "passed",
        "train_rows": len(train_data),
        "validation_rows": len(validation_data),
        "max_train_tokens": max_train_tokens,
        "max_validation_tokens": max_validation_tokens,
        "target_linear_modules": len(targets),
        "source_train_counts": dataset["source_train"],
        "source_validation_counts": dataset["source_validation"],
        "dataset_manifest_sha256": dataset["dataset_manifest_sha256"],
        "max_seq_len": config["max_seq_len"],
    }
    write_json(log_dir / "preflight_summary.json", preflight_summary)
    if args_cli.preflight_only:
        print(json.dumps(preflight_summary, ensure_ascii=False))
        return

    runtime_sources = profile.get("runtime_source_sha256", {})
    for relative_path, expected_digest in runtime_sources.items():
        if sha256(ROOT / relative_path) != expected_digest:
            raise ValueError(f"Runtime source checksum mismatch: {relative_path}")

    output = Path(args_cli.output_dir) if args_cli.output_dir else (
        ROOT / profile.get("experiment_dir", "experiments/lora_5250_r32_fullbf16_e5_earlystop_20261010")
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty training output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    events = output / "events.jsonl"
    EVENTS = events

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with BF16 support is required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one visible GPU, found {torch.cuda.device_count()}")
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    emit(
        events,
        "run_begin",
        mode="full_bf16_lora",
        train_examples=len(train_data),
        validation_examples=len(validation_data),
        maximum_epochs=config["epochs"],
        early_stopping=profile["early_stopping"],
        config=config,
        source_validation_counts=dataset["source_validation"],
        gpu=torch.cuda.get_device_name(0),
    )
    run_config = {
        "method": "full_bf16_base_lora",
        "base_quantization": "none",
        "model_path": str(model_path),
        "dataset_path": str(data_dir),
        "dataset_manifest_sha256": dataset["dataset_manifest_sha256"],
        "dataset_verified_sha256": dataset["dataset_verified_sha256"],
        "profile_sha256": sha256(profile_path),
        "dataset_counts": profile["expected_train_examples"] + profile["expected_validation_examples"],
        "source_train_counts": dataset["source_train"],
        "source_validation_counts": dataset["source_validation"],
        "effective_config": config,
        "max_train_tokens": max_train_tokens,
        "max_validation_tokens": max_validation_tokens,
        "target_linear_modules_expected": len(targets),
        "early_stopping": profile["early_stopping"],
        "save_strategy": profile["save_strategy"],
        "save_total_limit": profile["save_total_limit"],
        "load_best_model_at_end": profile["load_best_model_at_end"],
        "runtime_workflow": profile.get("runtime_workflow", "single"),
        "runtime_source_sha256": runtime_sources,
        "role_supervision_note": profile.get("role_supervision_note"),
    }
    write_json(output / "run_config.json", run_config)
    write_json(output / "target_modules.json", targets)

    emit(
        events,
        "model_load_begin",
        dtype="bfloat16",
        base_quantization="none",
        rank=config["lora_r"],
        alpha=config["lora_alpha"],
        target_linear_modules=len(targets),
        maximum_train_tokens=max_train_tokens,
        maximum_validation_tokens=max_validation_tokens,
    )
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=True,
        dtype=torch.bfloat16, device_map={"": 0},
    )
    model.config.use_cache = False
    if getattr(model.config, "text_config", None) is not None:
        model.config.text_config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            bias="none",
            target_modules=targets,
            task_type=TaskType.CAUSAL_LM,
        ),
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any("visual" in name for name, _ in trainable):
        raise RuntimeError("LoRA parameters are missing or include the vision tower")
    if not all("lora_" in name and "language_model" in name for name, _ in trainable):
        raise RuntimeError("Unexpected trainable parameters outside language-model LoRA")
    model_audit = {
        "method": "full_bf16_base_lora",
        "base_quantization": "none",
        "dtype": "bfloat16",
        "rank": config["lora_r"],
        "alpha": config["lora_alpha"],
        "trainable_parameters": sum(parameter.numel() for _, parameter in trainable),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "target_linear_modules": len(targets),
        "target_scope": profile["target_scope"],
        "max_train_tokens": max_train_tokens,
        "max_validation_tokens": max_validation_tokens,
        "gpu_name": torch.cuda.get_device_name(0),
    }
    write_json(output / "model_audit.json", model_audit)
    torch.cuda.synchronize()
    emit(
        events,
        "model_loaded",
        trainable_parameters=model_audit["trainable_parameters"],
        total_parameters=model_audit["total_parameters"],
        target_linear_modules=len(targets),
        gpu_allocated_gib=torch.cuda.memory_allocated() / 1024**3,
        gpu_reserved_gib=torch.cuda.memory_reserved() / 1024**3,
    )

    args = TrainingArguments(
        output_dir=str(output / "checkpoints"),
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["micro_batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        max_grad_norm=config["max_grad_norm"],
        bf16=True,
        fp16=False,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=config["optimizer"],
        eval_strategy="epoch",
        save_strategy=profile["save_strategy"],
        save_total_limit=profile["save_total_limit"],
        load_best_model_at_end=profile["load_best_model_at_end"],
        metric_for_best_model=profile["early_stopping"]["metric_for_best_model"],
        greater_is_better=profile["early_stopping"]["greater_is_better"],
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        report_to="none",
        seed=config["seed"],
        data_seed=config["seed"],
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        prediction_loss_only=True,
    )
    callbacks = [Progress(events)]
    if profile["early_stopping"]["enabled"]:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=profile["early_stopping"]["patience"],
            early_stopping_threshold=profile["early_stopping"]["threshold"],
        ))
    trainer = CompletionProjectionTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=validation_data,
        data_collator=PadCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
        callbacks=callbacks,
        projection_chunk_tokens=profile["projection_chunk_tokens"],
    )
    planned_steps = math.ceil(
        len(train_data) / (config["micro_batch_size"] * config["gradient_accumulation_steps"])
    ) * config["epochs"]
    emit(
        events,
        "training_begin",
        optimizer_steps_per_epoch=math.ceil(len(train_data) / config["gradient_accumulation_steps"]),
        maximum_optimizer_steps=planned_steps,
        max_epochs=config["epochs"],
        early_stopping=profile["early_stopping"],
    )
    resume_checkpoint = args_cli.resume_from_checkpoint or profile.get("resume_from_checkpoint")
    if resume_checkpoint is not None:
        resume_checkpoint = Path(resume_checkpoint)
        if not resume_checkpoint.is_dir() or not (resume_checkpoint / "trainer_state.json").is_file():
            raise FileNotFoundError(f"Invalid resume checkpoint: {resume_checkpoint}")
        resume_state = json.loads((resume_checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
        if int(resume_state.get("global_step", -1)) != int(profile.get("resume_expected_global_step", -2)):
            raise ValueError("Resume checkpoint global_step does not match the reviewed profile")
        emit(events, "resume_from_checkpoint", checkpoint=str(resume_checkpoint),
             global_step=resume_state["global_step"], epoch=resume_state.get("epoch"),
             best_metric=resume_state.get("best_metric"),
             best_model_checkpoint=resume_state.get("best_model_checkpoint"))
    result = trainer.train(resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None)
    trainer.save_model(str(output / "best_adapter"))
    tokenizer.save_pretrained(output / "best_adapter")
    trainer.save_metrics("train", result.metrics)
    eval_metrics = trainer.evaluate()
    if math.isfinite(eval_metrics.get("eval_loss", float("nan"))):
        eval_metrics["completion_perplexity"] = math.exp(min(eval_metrics["eval_loss"], 700))
    trainer.save_metrics("eval", eval_metrics)
    actual_steps = trainer.state.global_step
    emit(
        events,
        "training_complete",
        global_step=actual_steps,
        maximum_optimizer_steps=planned_steps,
        epoch=trainer.state.epoch,
        early_stopped=actual_steps < planned_steps,
        best_metric=trainer.state.best_metric,
        best_model_checkpoint=trainer.state.best_model_checkpoint,
        train_metrics=result.metrics,
        eval_metrics=eval_metrics,
        peak_gpu_gib=torch.cuda.max_memory_allocated() / 1024**3,
        adapter_dir=str(output / "best_adapter"),
        checkpoints=str(output / "checkpoints"),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if EVENTS is not None:
            is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
            emit(EVENTS, "fatal_error", error=repr(exc), is_oom=is_oom, traceback=traceback.format_exc())
        raise
