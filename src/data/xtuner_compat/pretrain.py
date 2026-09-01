"""Pure-text behavior matching XTuner's ``PretrainTokenizeFunction``."""

from __future__ import annotations

from typing import Any


def extract_pretrain_text(record: dict[str, Any], text_key: str | None = None) -> str:
    if text_key is not None:
        if text_key not in record:
            raise KeyError(f"text key {text_key!r} is missing from record")
        value = record[text_key]
    elif "messages" in record:
        messages = record["messages"]
        if not messages or messages[0].get("role") != "pretrain":
            raise ValueError("messages[0].role must be 'pretrain' for CPT records")
        value = messages[0]["content"]
    elif "content" in record:
        value = record["content"]
    else:
        raise KeyError("record must contain 'content', 'messages', or the configured text_key")
    return "" if value is None else str(value)


def prepare_pretrain_text(
    text: str,
    tokenizer: Any,
    *,
    add_eos_token: bool = True,
    add_bos_token: bool = False,
) -> str:
    """Apply BOS/EOS as literal tokenizer special-token strings, like XTuner."""

    if add_bos_token:
        if tokenizer.bos_token is None:
            raise ValueError("add_bos_token=true but tokenizer has no bos_token")
        text = tokenizer.bos_token + text
    if add_eos_token:
        if tokenizer.eos_token is None:
            raise ValueError("add_eos_token=true but tokenizer has no eos_token")
        text = text + tokenizer.eos_token
    return text


def proxy_attention_flops(
    num_tokens: int,
    *,
    flash_attn_block_size: int = 128,
    llm_pack_weight: float = 1.0,
) -> float:
    """Match the pure-text proxy used by XTuner's cache-mode tokenizer."""

    llm_num_patch = round(num_tokens / flash_attn_block_size) ** 2
    return float(llm_pack_weight * llm_num_patch)
