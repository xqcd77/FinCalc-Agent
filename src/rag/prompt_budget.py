"""Exact Qwen3 chat-prompt token counting for evidence budgeting.

The project intentionally loads the tokenizer file from the exact Qwen model
used for inference. It must not silently substitute a retrieval/reranker
tokenizer or a character-to-token estimate.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_TOKENIZER = PROJECT / "models/qwen3-4b-instruct-2507/tokenizer.json"
QWEN3_CHAT_TEMPLATE = """{messages}<|im_start|>assistant\n"""


def resolve_tokenizer_path(path: Path | str | None = None) -> Path:
    value = path or os.environ.get("QWEN_TOKENIZER_PATH") or DEFAULT_TOKENIZER
    result = Path(value).expanduser().resolve()
    if not result.is_file():
        raise FileNotFoundError(
            "Qwen3 tokenizer.json not found. Set QWEN_TOKENIZER_PATH or pass "
            f"--tokenizer-path with the tokenizer from Qwen3-4B-Instruct-2507. Tried: {result}"
        )
    return result


def load_tokenizer(path: Path | str | None = None) -> Any:
    resolved = resolve_tokenizer_path(path)
    # Qwen3.5 must use its own native template, including the no-thinking prefix.
    config = resolved.parent / "config.json"
    if config.is_file():
        import json
        if json.loads(config.read_text(encoding="utf-8")).get("model_type") == "qwen3_5":
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(str(resolved.parent), local_files_only=True)
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:  # pragma: no cover - dependency is installed in project env
        raise RuntimeError("Install the project's tokenizers dependency to count Qwen tokens") from exc
    tokenizer = Tokenizer.from_file(str(resolved))
    if tokenizer.get_vocab_size(with_added_tokens=True) < 1000:
        raise ValueError(f"Tokenizer vocabulary looks incomplete: {resolved}")
    return tokenizer


def render_qwen3_chat(messages: list[dict[str, str]]) -> str:
    """Render the verified Qwen3 two-message template with a generation prefix."""
    if not messages:
        raise ValueError("At least one chat message is required")
    rendered: list[str] = []
    for message in messages:
        role, content = message.get("role"), message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise ValueError("Qwen3 chat messages need a supported role and string content")
        rendered.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    return "".join(rendered) + "<|im_start|>assistant\n"


def count_chat_tokens(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    """Count the complete serialized chat prompt, including Qwen control tokens."""
    if hasattr(tokenizer, "apply_chat_template"):
        # Transformers 5 can return a BatchEncoding whose length counts keys.
        # Encode the same rendered string as the inference server instead.
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        encoded = tokenizer.encode(rendered, add_special_tokens=False)
        return len(getattr(encoded, "ids", encoded))
    encoded = tokenizer.encode(render_qwen3_chat(messages), add_special_tokens=False)
    ids = getattr(encoded, "ids", encoded)
    return len(ids)


def count_text_tokens(tokenizer: Any, text: str) -> int:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    ids = getattr(encoded, "ids", encoded)
    return len(ids)


def ensure_prompt_fits(tokenizer: Any, messages: list[dict[str, str]],
                       max_input_tokens: int, *, stage: str) -> int:
    used = count_chat_tokens(tokenizer, messages)
    if used > max_input_tokens:
        raise ValueError(f"{stage} prompt uses {used} tokens; budget is {max_input_tokens}")
    return used
