"""Model-aware trainable-parameter categories shared by both backends.

The public modes deliberately describe semantic parameter groups instead of
relying on a generic substring such as ``attention`` or ``expert``.  In
particular, block-level normalization parameters stay frozen even when a
backend fuses them into an attention or MLP projection module.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

ATTENTION = "attention"
EXPERT = "expert"
ROUTER = "router"
FULL = "full"

SUPPORTED_TRAINABLE_PARAM_MODES = frozenset({ATTENTION, EXPERT, ROUTER, FULL})
_MODE_CATEGORIES = {
    ATTENTION: frozenset({ATTENTION}),
    EXPERT: frozenset({EXPERT}),
    ROUTER: frozenset({ROUTER}),
    FULL: frozenset({ATTENTION, EXPERT, ROUTER}),
}

QWEN35_FAMILY = "qwen3_5_moe"
GEMMA4_FAMILY = "gemma4"

_HF_LANGUAGE_LAYER = re.compile(
    r"(?:^|\.)model\.language_model\.layers\.(?P<layer>\d+)\.(?P<relative>.+)$"
)


@dataclass(frozen=True)
class TrainableParameterSummary:
    mode: str
    family: str
    total_numel: int
    trainable_numel: int
    names: tuple[tuple[str, str], ...]
    category_tensors: dict[str, int]
    category_numel: dict[str, int]


def normalize_trainable_param_mode(mode: str) -> str:
    """Validate and normalize one public trainable-parameter mode."""

    normalized = str(mode).strip().lower()
    if normalized not in SUPPORTED_TRAINABLE_PARAM_MODES:
        raise ValueError(
            f"unsupported trainable_param_mode={mode!r}; "
            f"choose one of {sorted(SUPPORTED_TRAINABLE_PARAM_MODES)}"
        )
    return normalized


def trainable_categories(mode: str) -> frozenset[str]:
    """Return the disjoint semantic categories enabled by ``mode``."""

    return _MODE_CATEGORIES[normalize_trainable_param_mode(mode)]


def model_family(model_type: str, model_class_name: str = "") -> str:
    """Map the supported HF/MCore identifiers to a parameter-layout family."""

    identity = f"{model_type} {model_class_name}".lower().replace("-", "_")
    if "gemma4" in identity or "gemma_4" in identity:
        return GEMMA4_FAMILY
    if "qwen3_5" in identity or "qwen35" in identity or "intern_s2" in identity:
        # Intern-S2 uses the Qwen3.5 MoE text architecture.
        return QWEN35_FAMILY
    raise ValueError(
        "trainable parameter selection supports only Qwen3.5-35B, "
        f"Intern-S2-35B, and Gemma4-26B-A4B; got model_type={model_type!r}, "
        f"model_class={model_class_name!r}"
    )


def classify_hf_parameter(name: str, *, family: str) -> str | None:
    """Classify an HF checkpoint/runtime parameter, or return ``None``.

    Only parameters in the main text decoder layers are eligible.  This
    automatically excludes embeddings, LM heads, vision/audio towers,
    multimodal projectors, final norms, and MTP layers.
    """

    match = _HF_LANGUAGE_LAYER.search(name)
    if match is None:
        return None
    relative = match.group("relative")

    if family == QWEN35_FAMILY:
        if relative.startswith(("self_attn.", "linear_attn.")):
            # q_norm/k_norm and the Gated DeltaNet's internal norm are children
            # of these modules and therefore intentionally included.
            return ATTENTION
        if relative.startswith(("mlp.shared_expert_gate.", "mlp.gate.")):
            return ROUTER
        if relative.startswith(("mlp.experts.", "mlp.shared_expert.")):
            return EXPERT
        return None

    if family == GEMMA4_FAMILY:
        if relative.startswith("self_attn."):
            # Covers both sliding and full attention, including q/k/v norms.
            return ATTENTION
        if relative.startswith("router."):
            # Includes proj, scale, per_expert_scale and any future trainable
            # parameter owned by the router module.
            return ROUTER
        if relative.startswith(("experts.", "mlp.")):
            # Gemma4's always-on MLP is treated as its shared expert.
            return EXPERT
        return None

    raise ValueError(f"unsupported model family: {family!r}")


def configure_hf_trainable_parameters(model, mode: str) -> TrainableParameterSummary:
    """Apply one semantic mode to an HF model without importing Transformers."""

    normalized_mode = normalize_trainable_param_mode(mode)
    model_type = str(getattr(getattr(model, "config", None), "model_type", ""))
    family = model_family(model_type, type(model).__name__)
    enabled_categories = trainable_categories(normalized_mode)
    matched: list[tuple[str, str]] = []
    category_tensors: Counter[str] = Counter()
    category_numel: Counter[str] = Counter()
    total_numel = 0
    trainable_numel = 0

    for name, parameter in model.named_parameters():
        category = classify_hf_parameter(name, family=family)
        is_trainable = category in enabled_categories
        parameter.requires_grad_(is_trainable)
        parameter_numel = parameter.numel()
        total_numel += parameter_numel
        if not is_trainable:
            continue
        matched.append((category, name))
        trainable_numel += parameter_numel
        category_tensors[category] += 1
        category_numel[category] += parameter_numel

    if not matched:
        raise ValueError(
            f"No parameters matched trainable_param_mode={normalized_mode}; "
            f"model_type={model_type}, model_family={family}"
        )
    return TrainableParameterSummary(
        mode=normalized_mode,
        family=family,
        total_numel=total_numel,
        trainable_numel=trainable_numel,
        names=tuple(matched),
        category_tensors=dict(category_tensors),
        category_numel=dict(category_numel),
    )


def _is_fused_block_norm(relative_name: str, projection: str) -> bool:
    """Identify a block norm fused into a Transformer Engine projection."""

    return relative_name in {
        f"{projection}.layer_norm_weight",
        f"{projection}.layer_norm_bias",
    }


def classify_mcore_layer_parameter(name: str, *, family: str) -> str | None:
    """Classify a parameter name relative to one MCore Transformer layer."""

    if name.startswith("self_attention."):
        attention_name = name.removeprefix("self_attention.")
        if _is_fused_block_norm(attention_name, "linear_qkv") or _is_fused_block_norm(
            attention_name, "in_proj"
        ):
            # These are HF input_layernorm weights moved into a fused module;
            # they are not attention-internal q/k/v norms.
            return None
        return ATTENTION

    if family == QWEN35_FAMILY:
        if name.startswith("mlp.router.") or name in {
            "mlp.shared_experts.gate_weight",
            "mlp.shared_expert.gate_weight",
        }:
            return ROUTER
        if name.startswith(("mlp.experts.", "mlp.shared_experts.", "mlp.shared_expert.")):
            return EXPERT
        return None

    if family == GEMMA4_FAMILY:
        if name.startswith("experts_mlp.router.") or name in {
            "experts_mlp.scale",
            "experts_mlp.per_expert_scale",
        }:
            return ROUTER
        if name.startswith("experts_mlp.experts."):
            return EXPERT
        if name.startswith("mlp."):
            mlp_name = name.removeprefix("mlp.")
            if _is_fused_block_norm(mlp_name, "linear_fc1"):
                # HF pre_feedforward_layernorm is fused into Gemma4's shared
                # MLP projection by MCore-Bridge and must remain frozen.
                return None
            return EXPERT
        return None

    raise ValueError(f"unsupported model family: {family!r}")
