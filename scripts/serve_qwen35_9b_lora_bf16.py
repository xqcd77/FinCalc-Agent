"""OpenAI-compatible BF16 server for a Qwen3.5-9B LoRA adapter."""
import os
import json
import threading
import time
import uuid
import sys
from pathlib import Path
from exact_inference_cache import ExactInferenceCache
from model_run_identity import model_artifacts, hash_file

import torch
from fastapi import FastAPI, HTTPException
from peft import PeftModel
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

MODEL_ID = os.environ.get("QWEN_MODEL_ID", "Qwen3.5-9B-LoRA-BF16")
MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", str(Path(__file__).resolve().parents[1] / "models/Qwen3.5-9B"))
ADAPTER_PATH = os.environ.get("PEFT_ADAPTER_PATH", "")
CONTEXT_LIMIT = int(os.environ.get("QWEN_CONTEXT_LIMIT", "3200"))
PORT = int(os.environ.get("QWEN_PORT", "8000"))
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU must be enabled before starting the BF16 LoRA server")
if not torch.cuda.is_bf16_supported():
    raise RuntimeError("This evaluation requires BF16 support")
if not ADAPTER_PATH or not os.path.isdir(ADAPTER_PATH):
    raise FileNotFoundError(f"LoRA adapter directory not found: {ADAPTER_PATH}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)
base_model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map={"": 0},
    low_cpu_mem_usage=True,
    attn_implementation="sdpa",
    local_files_only=True,
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, is_trainable=False)
model.eval()
RUN_IDENTITY = dict(model=MODEL_ID, base=model_artifacts(MODEL_PATH),
                    adapter=model_artifacts(ADAPTER_PATH), precision='bfloat16',
                    attention='sdpa', context_limit=CONTEXT_LIMIT,
                    torch_version=torch.__version__,
                    transformers_version=__import__('transformers').__version__,
                    peft_version=__import__('peft').__version__)
RUN_IDENTITY['server_code']={p.name:hash_file(p) for p in (
    Path(__file__),Path(__file__).with_name('model_run_identity.py'),
    Path(__file__).with_name('exact_inference_cache.py'))}
model.config.use_cache = True
if getattr(model.config, "text_config", None) is not None:
    model.config.text_config.use_cache = True

print(
    f"Loaded {MODEL_ID}: base={MODEL_PATH}, adapter={ADAPTER_PATH}, "
    f"base_dtype=bfloat16, GPU={torch.cuda.get_device_name(0)}, "
    f"context_limit={CONTEXT_LIMIT}, allocated_gib={torch.cuda.memory_allocated()/1024**3:.2f}",
    flush=True,
)

lock = threading.Lock()
app = FastAPI()

response_cache = ExactInferenceCache(64)

@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}


@app.get('/v1/run-identity')
def run_identity():
    return RUN_IDENTITY


@app.post("/v1/chat/completions")
def chat(payload: dict):
    if payload.get("model") != MODEL_ID:
        raise HTTPException(400, "Requested model differs from the loaded adapter")
    messages = payload.get("messages", [])
    if not messages or any(not isinstance(m.get("content"), str) for m in messages):
        raise HTTPException(400, "Text chat messages required")
    max_new = int(payload.get("max_tokens", 512))
    if not 1 <= max_new <= 1024:
        raise HTTPException(400, "max_tokens must be between 1 and 1024")

    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    prompt_tokens = inputs.input_ids.shape[-1]
    if prompt_tokens + max_new > CONTEXT_LIMIT:
        raise HTTPException(400, "Prompt plus output exceeds the evaluation context budget")

    started = time.monotonic()
    with lock:
        cached = response_cache.get(payload)
        if cached is not None:
            cached['id'] = 'chatcmpl-' + uuid.uuid4().hex
            print(f"exact_cache_hit=1 seconds={time.monotonic()-started:.3f}", flush=True)
            return cached
        acquired = time.monotonic()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model.generate(
                **inputs.to("cuda"), max_new_tokens=max_new,
                do_sample=False, use_cache=True,
            )
        generated = outputs[0, prompt_tokens:]
        content = tokenizer.decode(generated, skip_special_tokens=True)
        completion_tokens = len(generated)
        result = {
            'id': 'chatcmpl-' + uuid.uuid4().hex, 'object': 'chat.completion',
            'model': MODEL_ID,
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': content},
                        'finish_reason': 'length' if completion_tokens == max_new else 'stop'}],
            'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens,
                      'total_tokens': prompt_tokens + completion_tokens},
        }
        response_cache.put(payload, result)
    print(f"prompt={prompt_tokens} output={completion_tokens} "
          f"seconds={time.monotonic()-started:.2f} exact_cache_hit=0 "
          f"generate_seconds={time.monotonic()-acquired:.3f} "
          f"allocated_gib={torch.cuda.memory_allocated()/1024**3:.2f}", flush=True)
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=PORT, workers=1)
