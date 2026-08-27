"""Translate a local Hugging Face config into MCore-Bridge settings."""

from __future__ import annotations

import os
from typing import Any

import torch

from .config import MegatronRunConfig


def build_mcore_config_dict(run: MegatronRunConfig) -> dict[str, Any]:
    """Convert model metadata without constructing or loading model weights."""
    # Native MCore Gated DeltaNet is the intended Qwen3.5/Intern-S2 path.
    os.environ.setdefault("USE_MCORE_GDN", "1")

    from mcore_bridge import hf_to_mcore_config
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(
        run.model.path,
        trust_remote_code=run.model.trust_remote_code,
        local_files_only=True,
    )
    result = hf_to_mcore_config(hf_config)
    result.update(
        tensor_model_parallel_size=run.parallel.tensor_parallel,
        pipeline_model_parallel_size=run.parallel.pipeline_parallel,
        context_parallel_size=run.parallel.context_parallel,
        expert_model_parallel_size=run.parallel.expert_parallel,
        expert_tensor_parallel_size=(
            run.parallel.expert_tensor_parallel or run.parallel.tensor_parallel
        ),
        sequence_parallel=run.parallel.sequence_parallel,
        # PP=2 uses MCore's non-interleaved pipeline schedule. MCore 0.16
        # rejects overlap_p2p_comm for that schedule.
        overlap_p2p_comm=False,
        bf16=run.training.bf16,
        fp16=False,
        params_dtype=torch.bfloat16 if run.training.bf16 else torch.float32,
        pipeline_dtype=torch.bfloat16 if run.training.bf16 else torch.float32,
    )
    if run.parallel.num_layers_in_first_pipeline_stage is not None:
        result["num_layers_in_first_pipeline_stage"] = (
            run.parallel.num_layers_in_first_pipeline_stage
        )
    if run.parallel.num_layers_in_last_pipeline_stage is not None:
        result["num_layers_in_last_pipeline_stage"] = (
            run.parallel.num_layers_in_last_pipeline_stage
        )
    _validate_pipeline_layer_partition(result)
    # The standardized TP4/PP2/EP4 path intentionally does not use activation
    # recomputation / gradient checkpointing.
    for key in ("recompute_granularity", "recompute_method", "recompute_num_layers"):
        result.pop(key, None)
    return result


def _validate_pipeline_layer_partition(config: dict[str, Any]) -> None:
    """Validate MCore's first/middle/last uneven PP layer assignment."""
    pp_size = int(config["pipeline_model_parallel_size"])
    first = config.get("num_layers_in_first_pipeline_stage")
    last = config.get("num_layers_in_last_pipeline_stage")
    if first is None and last is None:
        return

    num_layers = int(config["num_layers"])
    specified_layers = int(first or 0) + int(last or 0)
    specified_stages = int(first is not None) + int(last is not None)
    remaining_layers = num_layers - specified_layers
    remaining_stages = pp_size - specified_stages
    if remaining_layers < 0 or remaining_stages < 0:
        raise ValueError("uneven pipeline assignment exceeds model layers or PP stages")
    if remaining_stages == 0:
        if remaining_layers:
            raise ValueError("uneven pipeline assignment does not cover every model layer")
        return
    if remaining_layers <= 0 or remaining_layers % remaining_stages:
        raise ValueError(
            "remaining model layers must divide evenly across middle pipeline stages"
        )


def summarize_mcore_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return stable, JSON-safe fields useful in a CPU preflight report."""
    keys = (
        "hf_model_type",
        "llm_model_type",
        "mcore_model_type",
        "num_layers",
        "hidden_size",
        "num_attention_heads",
        "num_query_groups",
        "num_moe_experts",
        "moe_router_topk",
        "max_position_embeddings",
        "padded_vocab_size",
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "num_layers_in_first_pipeline_stage",
        "num_layers_in_last_pipeline_stage",
        "context_parallel_size",
        "expert_model_parallel_size",
        "expert_tensor_parallel_size",
        "sequence_parallel",
    )
    return {key: config.get(key) for key in keys}


def summarize_hf_config(run: MegatronRunConfig) -> dict[str, Any]:
    """Inspect local HF metadata without importing TE or MCore-Bridge."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        run.model.path,
        trust_remote_code=run.model.trust_remote_code,
        local_files_only=True,
    )
    payload = config.to_dict()
    text = payload.get("text_config") or payload
    if hasattr(text, "to_dict"):
        text = text.to_dict()
    keys = (
        "model_type",
        "num_hidden_layers",
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "num_experts",
        "num_experts_per_tok",
        "max_position_embeddings",
        "vocab_size",
    )
    summary = {key: text.get(key) for key in keys}
    summary["outer_model_type"] = payload.get("model_type")
    return summary
