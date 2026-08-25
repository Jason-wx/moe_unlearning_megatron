"""Pure tensor objectives shared by future Megatron trainers.

Inputs use per-token negative log likelihood rather than Hugging Face model
outputs.  Megatron can produce these losses with a vocab-parallel cross entropy
without gathering the full vocabulary logits on every rank.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _tp_info() -> tuple[int, int, object | None]:
    if not (dist.is_available() and dist.is_initialized()):
        return 1, 0, None
    from megatron.core import parallel_state

    return (
        parallel_state.get_tensor_model_parallel_world_size(),
        parallel_state.get_tensor_model_parallel_rank(),
        parallel_state.get_tensor_model_parallel_group(),
    )


def _tp_sum(value: torch.Tensor, tp_size: int) -> torch.Tensor:
    if tp_size == 1:
        return value
    # Forward all-reduce, backward identity: every local shard contributes to
    # the global value and receives its own derivative.
    from megatron.core.tensor_parallel.mappings import (
        reduce_from_tensor_model_parallel_region,
    )

    return reduce_from_tensor_model_parallel_region(value)


def _tp_max_with_local_gradient(
    local_max: torch.Tensor,
    tp_size: int,
    tp_group,
) -> torch.Tensor:
    if tp_size == 1:
        return local_max
    global_max = local_max.detach().clone()
    dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=tp_group)
    owner = local_max.detach().eq(global_max).to(local_max.dtype)
    owner_count = owner.clone()
    dist.all_reduce(owner_count, op=dist.ReduceOp.SUM, group=tp_group)
    # The forward value is the global maximum. Its gradient is routed only to
    # the owning vocab shard (equally split for the extremely rare exact tie).
    return global_max + owner / owner_count * (local_max - local_max.detach())


def _validate_token_tensors(values: torch.Tensor, mask: torch.Tensor) -> None:
    if values.shape != mask.shape:
        raise ValueError(
            f"values and mask must have the same shape, got {values.shape} and {mask.shape}"
        )
    if values.ndim < 1:
        raise ValueError("token tensors must have at least one dimension")


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return the mean over valid positions and reject an empty batch."""

    _validate_token_tensors(values, mask)
    mask = mask.to(device=values.device, dtype=values.dtype)
    count = mask.sum()
    if not torch.any(count > 0):
        raise ValueError("mask contains no valid tokens")
    return (values * mask).sum() / count


def masked_sequence_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return valid-token means for nonempty sequences only.

    Cached document chunks can occasionally contain no supervised next-token
    positions after shifting/masking. Such rows carry no learning signal and
    are excluded instead of introducing a divide-by-zero value.
    """

    _validate_token_tensors(values, mask)
    mask = mask.to(device=values.device, dtype=values.dtype)
    count = mask.sum(dim=-1)
    nonempty = count > 0
    if not torch.any(nonempty):
        raise ValueError("batch contains no valid sequences")
    token_sum = (values * mask).sum(dim=-1)
    return token_sum[nonempty] / count[nonempty]


def grad_ascent_objective(
    forget_token_nll: torch.Tensor,
    forget_mask: torch.Tensor,
) -> torch.Tensor:
    """Gradient-ascent objective minimized by the optimizer."""

    return -masked_mean(forget_token_nll, forget_mask)


def grad_diff_objective(
    forget_token_nll: torch.Tensor,
    forget_mask: torch.Tensor,
    retain_token_nll: torch.Tensor,
    retain_mask: torch.Tensor,
    *,
    gamma: float = 1.0,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combine forget gradient ascent with retain language-model training."""

    forget_loss = grad_ascent_objective(forget_token_nll, forget_mask)
    retain_loss = masked_mean(retain_token_nll, retain_mask)
    loss = gamma * forget_loss + alpha * retain_loss
    return loss, {"forget_loss": forget_loss, "retain_loss": retain_loss}


def simnpo_objective(
    forget_token_nll: torch.Tensor,
    forget_mask: torch.Tensor,
    *,
    beta: float = 0.1,
    delta: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Token-level SimNPO objective used by the long-document adaptation."""

    if beta <= 0:
        raise ValueError("beta must be positive")
    _validate_token_tensors(forget_token_nll, forget_mask)
    valid_nll = forget_token_nll[forget_mask.bool()]
    if valid_nll.numel() == 0:
        raise ValueError("forget batch contains no valid tokens")
    sigmoid_argument = beta * (valid_nll - delta)
    token_loss = -2.0 / beta * F.logsigmoid(sigmoid_argument)
    gradient_weight = 2.0 * torch.sigmoid(-sigmoid_argument)
    return token_loss.mean(), {
        "forget_mean_nll": valid_nll.mean(),
        "sigmoid_argument": sigmoid_argument.mean(),
        "forget_gradient_weight": gradient_weight.mean(),
    }


def wga_objective(
    forget_token_nll: torch.Tensor,
    forget_mask: torch.Tensor,
    *,
    beta: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Weighted gradient ascent with detached token-probability weights."""
    _validate_token_tensors(forget_token_nll, forget_mask)
    probability = torch.exp(-forget_token_nll).detach()
    weight = probability.pow(beta)
    loss = -masked_mean(weight * forget_token_nll, forget_mask)
    return loss, {"forget_weight": masked_mean(weight, forget_mask)}


def satimp_objective(
    forget_token_nll: torch.Tensor,
    forget_mask: torch.Tensor,
    *,
    beta1: float = 5.0,
    beta2: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Saturation/importance-weighted gradient ascent."""
    _validate_token_tensors(forget_token_nll, forget_mask)
    probability = torch.exp(-forget_token_nll).detach()
    saturation = probability.pow(beta1)
    importance = (1.0 - probability).clamp_min(0.0).pow(beta2)
    weight = saturation * importance
    loss = -masked_mean(weight * forget_token_nll, forget_mask)
    return loss, {
        "forget_weight": masked_mean(weight, forget_mask),
        "saturation_weight": masked_mean(saturation, forget_mask),
        "importance_weight": masked_mean(importance, forget_mask),
    }


def ceu_objective(
    forget_logits: torch.Tensor,
    forget_labels: torch.Tensor,
    *,
    ignore_first_n_tokens: int = 1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Cross-entropy unlearning over already next-token-aligned MCore labels."""
    if forget_logits.shape[:-1] != forget_labels.shape:
        raise ValueError("logits and labels have incompatible shapes")
    valid = forget_labels.ne(-100)
    if ignore_first_n_tokens:
        ignored = valid & (valid.cumsum(dim=-1) <= ignore_first_n_tokens)
        valid = valid & ~ignored
    if not torch.any(valid):
        raise ValueError("CEU batch contains no valid tokens")

    logits = forget_logits[valid].float()
    labels = forget_labels[valid]
    tp_size, tp_rank, tp_group = _tp_info()
    local_vocab_size = logits.shape[-1]
    vocab_start = tp_rank * local_vocab_size
    vocab_end = vocab_start + local_vocab_size

    global_max = logits.detach().max(dim=-1).values
    if tp_size > 1:
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=tp_group)
    shifted = logits - global_max.unsqueeze(-1)
    local_exp = shifted.exp()
    global_exp_sum = _tp_sum(local_exp.sum(dim=-1), tp_size)
    probabilities = local_exp / global_exp_sum.unsqueeze(-1)
    log_probabilities = shifted - global_exp_sum.log().unsqueeze(-1)

    owns_label = labels.ge(vocab_start) & labels.lt(vocab_end)
    local_label = (labels - vocab_start).clamp(0, local_vocab_size - 1)
    local_correct_probability = probabilities.gather(1, local_label.unsqueeze(1)).squeeze(1)
    local_correct_probability = local_correct_probability * owns_label
    correct_probability = _tp_sum(local_correct_probability, tp_size).detach()

    target = probabilities.detach().clone()
    if torch.any(owns_label):
        rows = torch.arange(target.shape[0], device=target.device)[owns_label]
        target[rows, local_label[owns_label]] = 0.0
    target = target / (1.0 - correct_probability).clamp_min(1e-12).unsqueeze(-1)
    local_loss = -(target * log_probabilities).sum(dim=-1)
    loss = _tp_sum(local_loss, tp_size).mean()
    return loss, {"ceu_valid_tokens": valid.sum().detach()}


def pdu_forget_objective(
    forget_logits: torch.Tensor,
    forget_mask: torch.Tensor,
) -> torch.Tensor:
    """PDU logit dispersion over a full or tensor-parallel vocabulary."""
    if forget_logits.shape[:-1] != forget_mask.shape:
        raise ValueError("logits and mask have incompatible shapes")
    logits = forget_logits.float()
    tp_size, _, tp_group = _tp_info()
    local_max = logits.max(dim=-1).values
    global_max = _tp_max_with_local_gradient(local_max, tp_size, tp_group)
    global_sum = _tp_sum(logits.sum(dim=-1), tp_size)
    global_vocab_size = logits.shape[-1] * tp_size
    dispersion = (global_max - global_sum / global_vocab_size).square()
    return masked_mean(dispersion, forget_mask)


def vocab_parallel_token_nll(
    local_logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Unreduced NLL for full or tensor-parallel vocabulary logits."""
    tp_size, _, _ = _tp_info()
    if tp_size == 1:
        return F.cross_entropy(
            local_logits.transpose(1, 2).float(),
            labels,
            ignore_index=-100,
            reduction="none",
        )
    from megatron.core.tensor_parallel.cross_entropy import (
        vocab_parallel_cross_entropy,
    )

    safe_labels = labels.masked_fill(labels.eq(-100), 0)
    losses = vocab_parallel_cross_entropy(
        local_logits.transpose(0, 1).contiguous(),
        safe_labels.transpose(0, 1).contiguous(),
    ).transpose(0, 1)
    return losses.masked_fill(labels.eq(-100), 0.0)


def pure_npo_objective(
    current_token_nll: torch.Tensor,
    reference_token_nll: torch.Tensor,
    token_mask: torch.Tensor,
    *,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Length-normalized pure NPO objective for forget sequences.

    Reference NLL may later come from an offline cache, so this function does
    not require a live reference model.
    """

    if beta <= 0:
        raise ValueError("beta must be positive")
    sequence_token_count = token_mask.sum(dim=-1)
    valid_sequences = sequence_token_count.gt(0)
    current = masked_sequence_mean(current_token_nll, token_mask)
    reference = masked_sequence_mean(reference_token_nll, token_mask)
    logit = beta * (current - reference)
    loss = (-2.0 / beta * F.logsigmoid(logit)).mean()
    return loss, {
        "current_sequence_nll": current.mean(),
        "reference_sequence_nll": reference.mean(),
        "sequence_nll_difference": (current - reference).mean(),
        "npo_logit": logit.mean(),
        "sequence_gradient_weight": (2.0 * torch.sigmoid(-logit)).mean(),
        "valid_sequences": valid_sequences.sum().detach(),
        "skipped_empty_sequences": (~valid_sequences).sum().detach(),
    }


def token_pure_npo_objective(
    current_token_nll: torch.Tensor,
    reference_token_nll: torch.Tensor,
    token_mask: torch.Tensor,
    *,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Pure NPO with an independent bounded objective for each valid token."""

    if beta <= 0:
        raise ValueError("beta must be positive")
    if current_token_nll.shape != reference_token_nll.shape:
        raise ValueError("current/reference token NLL shapes must match")
    if current_token_nll.shape != token_mask.shape:
        raise ValueError("token NLL and mask shapes must match")

    logit = beta * (current_token_nll - reference_token_nll)
    token_loss = -2.0 / beta * F.logsigmoid(logit)
    loss = masked_mean(token_loss, token_mask)
    return loss, {
        "current_token_nll": masked_mean(current_token_nll, token_mask),
        "reference_token_nll": masked_mean(reference_token_nll, token_mask),
        "token_nll_difference": masked_mean(current_token_nll - reference_token_nll, token_mask),
        "token_npo_logit": masked_mean(logit, token_mask),
        # Magnitude of dL/d(current NLL), before gamma and token averaging.
        "token_gradient_weight": masked_mean(2.0 * torch.sigmoid(-logit), token_mask),
    }


def undial_objective(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    token_mask: torch.Tensor,
    *,
    beta: float = 10.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """UNDIAL soft-target CE over a full or tensor-parallel vocabulary.

    MCore labels are already next-token aligned. The teacher's correct-token
    logit is reduced by ``beta`` before constructing the frozen soft target.
    """
    if beta < 0:
        raise ValueError("beta must be nonnegative")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have identical local shapes")
    if student_logits.shape[:-1] != labels.shape or labels.shape != token_mask.shape:
        raise ValueError("logits, labels, and token_mask have incompatible shapes")

    tp_size, tp_rank, tp_group = _tp_info()
    local_vocab_size = student_logits.shape[-1]
    vocab_start = tp_rank * local_vocab_size
    vocab_end = vocab_start + local_vocab_size
    valid = token_mask.bool() & labels.ne(-100)
    if not torch.any(valid):
        raise ValueError("UNDIAL batch contains no valid tokens")

    teacher = teacher_logits.detach().float().clone()
    owns_label = valid & labels.ge(vocab_start) & labels.lt(vocab_end)
    local_label = (labels - vocab_start).clamp(0, local_vocab_size - 1)
    if torch.any(owns_label):
        rows, columns = owns_label.nonzero(as_tuple=True)
        teacher[rows, columns, local_label[rows, columns]] -= beta

    teacher_global_max = teacher.max(dim=-1).values
    if tp_size > 1:
        dist.all_reduce(teacher_global_max, op=dist.ReduceOp.MAX, group=tp_group)
    teacher_shifted = teacher - teacher_global_max.unsqueeze(-1)
    teacher_denominator = _tp_sum(teacher_shifted.exp().sum(dim=-1), tp_size)
    soft_targets = (teacher_shifted - teacher_denominator.log().unsqueeze(-1)).exp().detach()

    student = student_logits.float()
    student_global_max = student.detach().max(dim=-1).values
    if tp_size > 1:
        dist.all_reduce(student_global_max, op=dist.ReduceOp.MAX, group=tp_group)
    student_shifted = student - student_global_max.unsqueeze(-1)
    student_denominator = _tp_sum(student_shifted.exp().sum(dim=-1), tp_size)
    student_log_probabilities = student_shifted - student_denominator.log().unsqueeze(-1)

    local_cross_entropy = -(soft_targets * student_log_probabilities).sum(dim=-1)
    cross_entropy = _tp_sum(local_cross_entropy, tp_size)
    loss = masked_mean(cross_entropy, valid)
    return loss, {"undial_valid_tokens": valid.sum().detach()}


def rmu_mse_statistics(
    activation: torch.Tensor,
    target: torch.Tensor,
    token_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return additive numerator/count for an RMU activation MSE.

    The two values can be all-reduced over tensor/sequence/context-parallel
    groups before division, which avoids gathering a full activation tensor.
    """

    if activation.shape != target.shape:
        raise ValueError(
            f"activation and target must have the same shape, got {activation.shape} and {target.shape}"
        )
    if activation.ndim < 2 or token_mask.shape != activation.shape[:-1]:
        raise ValueError(
            "token_mask must match every activation dimension except the hidden dimension"
        )
    mask = token_mask.to(device=activation.device, dtype=activation.dtype)
    numerator = ((activation - target).square() * mask.unsqueeze(-1)).sum()
    count = mask.sum() * activation.shape[-1]
    if not torch.any(count > 0):
        raise ValueError("RMU mask contains no valid tokens")
    return numerator, count


def parallel_rmu_mse(
    activation: torch.Tensor,
    target: torch.Tensor,
    token_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Globally normalized RMU MSE for sequence-parallel activations.

    Each TP rank owns a different sequence shard but the complete hidden
    dimension. The all-reduced numerator has an identity backward mapping, so
    every rank receives gradients only for its local activation shard.
    """
    if activation.shape != target.shape:
        raise ValueError("RMU activation and target shapes differ")
    if activation.ndim != 3 or token_mask.shape != activation.shape[:-1]:
        raise ValueError("RMU token_mask must match [local_sequence, batch]")
    mask = token_mask.to(device=activation.device, dtype=activation.dtype)
    local_numerator = ((activation - target).square() * mask.unsqueeze(-1)).sum()
    local_count = mask.sum() * activation.shape[-1]
    tp_size, _, tp_group = _tp_info()
    numerator = _tp_sum(local_numerator, tp_size)
    count = local_count.detach().clone()
    if tp_size > 1:
        dist.all_reduce(count, op=dist.ReduceOp.SUM, group=tp_group)
    if not torch.any(count > 0):
        raise ValueError("RMU mask contains no valid tokens across the TP group")
    return numerator / count, {
        "rmu_squared_error_sum": numerator.detach(),
        "rmu_element_count": count.detach(),
    }
