"""Pure tensor objectives shared by future Megatron trainers.

Inputs use per-token negative log likelihood rather than Hugging Face model
outputs.  Megatron can produce these losses with a vocab-parallel cross entropy
without gathering the full vocabulary logits on every rank.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

# Debug switch for the gemma4 x PDU rmsnorm_bwd illegal-memory-access:
# when OU_PDU_CONTIG_LOSS=1 the full-vocab PDU losses are computed on fully
# contiguous fp32 tensors (plain 2D cross entropy, contiguous dispersion
# input) instead of the strided layouts produced by transpose().float().
# Off by default — behavior is unchanged for every other run.
_CONTIG_LOSS = os.environ.get("OU_PDU_CONTIG_LOSS", "") == "1"


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
    """Original sequence-level SimNPO with valid-token normalization."""

    if beta <= 0:
        raise ValueError("beta must be positive")
    sequence_nll = masked_sequence_mean(forget_token_nll, forget_mask)
    sigmoid_argument = beta * (sequence_nll - delta)
    sequence_loss = -2.0 / beta * F.logsigmoid(sigmoid_argument)
    gradient_weight = 2.0 * torch.sigmoid(-sigmoid_argument)
    return sequence_loss.mean(), {
        "forget_mean_nll": sequence_nll.mean(),
        "sigmoid_argument": sigmoid_argument.mean(),
        "forget_gradient_weight": gradient_weight.mean(),
    }


def token_simnpo_objective(
    forget_token_nll: torch.Tensor,
    forget_mask: torch.Tensor,
    *,
    beta: float = 0.1,
    delta: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """SimNPO with the bounded objective applied independently per token."""

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
    valid = forget_mask.bool()
    valid_probability = probability[valid].float()
    valid_weight = weight[valid].float()
    if valid_probability.numel() == 0:
        raise ValueError("forget batch contains no valid tokens")
    probability_quantiles = torch.quantile(
        valid_probability, valid_probability.new_tensor([0.1, 0.5, 0.9])
    )
    weight_quantiles = torch.quantile(valid_weight, valid_weight.new_tensor([0.1, 0.5, 0.9]))
    weight_square_sum = valid_weight.square().sum()
    effective_token_fraction = valid_weight.sum().square() / (
        valid_weight.numel() * weight_square_sum.clamp_min(torch.finfo(valid_weight.dtype).tiny)
    )
    return loss, {
        "token_probability_mean": valid_probability.mean(),
        "token_probability_p10": probability_quantiles[0],
        "token_probability_p50": probability_quantiles[1],
        "token_probability_p90": probability_quantiles[2],
        "forget_weight": valid_weight.mean(),
        "forget_weight_p10": weight_quantiles[0],
        "forget_weight_p50": weight_quantiles[1],
        "forget_weight_p90": weight_quantiles[2],
        "forget_weight_max": valid_weight.max(),
        "forget_weight_below_1e-3_fraction": (valid_weight < 1.0e-3).float().mean(),
        "forget_weight_below_1e-4_fraction": (valid_weight < 1.0e-4).float().mean(),
        "forget_weight_effective_token_fraction": effective_token_fraction,
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

    owns_label = labels.ge(vocab_start) & labels.lt(vocab_end)
    local_label = (labels - vocab_start).clamp(0, local_vocab_size - 1)

    # Leave-gold-out target, built in log space: softmax over the logits with
    # the gold entry masked out is exactly p_v / (1 - p_gold), and stays exact
    # no matter how confident the gold logit is. The previous direct
    # renormalization divided by (1 - p_gold).clamp_min(1e-12): once p_gold
    # rounded to 1.0 in fp32 (gold leading by >~17 nats) the clamp inflated
    # the target up to ~6e3x, and once the non-gold softmax underflowed the
    # target vanished and the gradient was zeroed. Shifting by the masked
    # (non-gold) global max instead of the overall max keeps the largest
    # non-gold entry at exp(0)=1, so the target never underflows either.
    masked_shifted = shifted.detach().clone()
    if torch.any(owns_label):
        rows = torch.arange(masked_shifted.shape[0], device=masked_shifted.device)[owns_label]
        masked_shifted[rows, local_label[owns_label]] = torch.finfo(masked_shifted.dtype).min
    masked_max = masked_shifted.max(dim=-1).values
    if tp_size > 1:
        dist.all_reduce(masked_max, op=dist.ReduceOp.MAX, group=tp_group)
    masked_shifted = (masked_shifted - masked_max.unsqueeze(-1)).exp()
    masked_sum = _tp_sum(masked_shifted.sum(dim=-1), tp_size)
    target = masked_shifted / masked_sum.unsqueeze(-1)
    # -sum_v target * (shifted - log Z), split so that log Z is weighted by the
    # GLOBAL target mass. Summing local -(target * log_probabilities) instead
    # would backpropagate log Z on each TP rank with only that shard's target
    # mass (the _tp_sum backward is identity), giving a wrong gradient for TP>1.
    target_mass = _tp_sum(target.sum(dim=-1), tp_size)
    cross = _tp_sum((target * shifted).sum(dim=-1), tp_size)
    loss = (global_exp_sum.log() * target_mass - cross).mean()
    return loss, {"ceu_valid_tokens": valid.sum().detach()}


def pdu_forget_objective(
    forget_logits: torch.Tensor,
    forget_mask: torch.Tensor,
) -> torch.Tensor:
    """PDU logit dispersion over a full or tensor-parallel vocabulary."""
    if forget_logits.shape[:-1] != forget_mask.shape:
        raise ValueError("logits and mask have incompatible shapes")
    logits = forget_logits.float()
    if _CONTIG_LOSS:
        logits = logits.contiguous()
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
        if _CONTIG_LOSS:
            # Debug variant: plain 2D CE over a fully contiguous fp32 copy
            # (the [B, V, S] strided layout from transpose().float() is the
            # prime suspect for the gemma4 x PDU rmsnorm_bwd illegal access).
            vocab = local_logits.shape[-1]
            flat = local_logits.float().reshape(-1, vocab)
            losses = F.cross_entropy(
                flat,
                labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
            )
            return losses.reshape(labels.shape)
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


def npo_objective(
    current_token_nll: torch.Tensor,
    reference_token_nll: torch.Tensor,
    token_mask: torch.Tensor,
    *,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Original NPO: the negative/loser-only DPO term over sequence sums."""

    if beta <= 0:
        raise ValueError("beta must be positive")
    if current_token_nll.shape != reference_token_nll.shape:
        raise ValueError("current/reference token NLL shapes must match")
    _validate_token_tensors(current_token_nll, token_mask)
    mask = token_mask.to(device=current_token_nll.device, dtype=current_token_nll.dtype)
    valid_sequences = mask.sum(dim=-1).gt(0)
    if not torch.any(valid_sequences):
        raise ValueError("batch contains no valid sequences")
    current = (current_token_nll * mask).sum(dim=-1)[valid_sequences]
    reference = (reference_token_nll * mask).sum(dim=-1)[valid_sequences]
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


def token_npo_objective(
    current_token_nll: torch.Tensor,
    reference_token_nll: torch.Tensor,
    token_mask: torch.Tensor,
    *,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """NPO with an independent bounded objective for each valid token."""

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


# Backward-compatible Python symbol. Algorithm configuration uses ``token_npo``.
token_pure_npo_objective = token_npo_objective


def undial_objective(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    token_mask: torch.Tensor,
    *,
    beta: float = 10.0,
    chunk_rows: int | None = None,
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

    owns_label = valid & labels.ge(vocab_start) & labels.lt(vocab_end)
    local_label = (labels - vocab_start).clamp(0, local_vocab_size - 1)
    if chunk_rows is None:
        chunk_rows = max(1, _UNDIAL_CHUNK_ELEMENTS // local_vocab_size)
    cross_entropy = _UndialSoftTargetCrossEntropy.apply(
        student_logits.reshape(-1, local_vocab_size),
        teacher_logits.detach().reshape(-1, local_vocab_size),
        local_label.reshape(-1),
        owns_label.reshape(-1),
        float(beta),
        tp_size,
        tp_group,
        int(chunk_rows),
    ).reshape(labels.shape)
    loss = masked_mean(cross_entropy, valid)
    return loss, {"undial_valid_tokens": valid.sum().detach()}


# fp32 elements per UNDIAL row chunk (2**27 -> 512 MiB per temporary).
_UNDIAL_CHUNK_ELEMENTS = 1 << 27


def _tp_all_reduce_(value: torch.Tensor, op, tp_size: int, tp_group) -> torch.Tensor:
    if tp_size > 1:
        dist.all_reduce(value, op=op, group=tp_group)
    return value


def _undial_teacher_chunk(teacher, local_label, owns_label, beta, start, end):
    chunk = teacher[start:end].to(torch.float32, copy=True)
    owns = owns_label[start:end]
    if torch.any(owns):
        rows = owns.nonzero(as_tuple=True)[0]
        chunk[rows, local_label[start:end][rows]] -= beta
    return chunk


class _UndialSoftTargetCrossEntropy(torch.autograd.Function):
    """Row-wise ``-sum_v p_v log softmax(s)_v`` over a full or TP vocab shard.

    ``p`` is the (label-penalized) teacher softmax. The global target mass is
    exactly 1, so the loss is ``logsumexp(s) - sum_v p_v s_v`` and the local
    gradient is ``softmax(s) - p`` on every TP shard. Rows are processed in
    chunks and only per-row global log-normalizers are saved, so no full-vocab
    fp32 tensor is ever materialized (the naive form keeps ~6 of them alive).
    """

    @staticmethod
    def forward(
        ctx, student, teacher, local_label, owns_label, beta, tp_size, tp_group, chunk_rows
    ):
        rows = student.shape[0]
        student_lse = torch.empty(rows, device=student.device, dtype=torch.float32)
        teacher_lse = torch.empty_like(student_lse)
        loss = torch.empty_like(student_lse)
        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            chunk = _undial_teacher_chunk(teacher, local_label, owns_label, beta, start, end)
            teacher_max = _tp_all_reduce_(
                chunk.max(dim=-1).values, dist.ReduceOp.MAX, tp_size, tp_group
            )
            chunk.sub_(teacher_max.unsqueeze(-1))
            teacher_sum = _tp_all_reduce_(
                chunk.exp().sum(dim=-1), dist.ReduceOp.SUM, tp_size, tp_group
            )
            teacher_lse[start:end] = teacher_max + teacher_sum.log()
            soft_targets = chunk.sub_(teacher_sum.log().unsqueeze(-1)).exp_()

            chunk = student[start:end].to(torch.float32, copy=True)
            student_max = _tp_all_reduce_(
                chunk.max(dim=-1).values, dist.ReduceOp.MAX, tp_size, tp_group
            )
            chunk.sub_(student_max.unsqueeze(-1))
            student_sum = _tp_all_reduce_(
                chunk.exp().sum(dim=-1), dist.ReduceOp.SUM, tp_size, tp_group
            )
            student_lse[start:end] = student_max + student_sum.log()
            cross = _tp_all_reduce_(
                (soft_targets * chunk).sum(dim=-1), dist.ReduceOp.SUM, tp_size, tp_group
            )
            loss[start:end] = student_sum.log() - cross
            del chunk, soft_targets
        ctx.save_for_backward(student, teacher, local_label, owns_label, student_lse, teacher_lse)
        ctx.beta = beta
        ctx.chunk_rows = chunk_rows
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        student, teacher, local_label, owns_label, student_lse, teacher_lse = ctx.saved_tensors
        grad_student = torch.empty_like(student)
        grad_loss = grad_loss.float()
        rows = student.shape[0]
        for start in range(0, rows, ctx.chunk_rows):
            end = min(start + ctx.chunk_rows, rows)
            chunk = _undial_teacher_chunk(teacher, local_label, owns_label, ctx.beta, start, end)
            soft_targets = chunk.sub_(teacher_lse[start:end].unsqueeze(-1)).exp_()
            chunk = student[start:end].to(torch.float32, copy=True)
            chunk.sub_(student_lse[start:end].unsqueeze(-1)).exp_().sub_(soft_targets)
            grad_student[start:end] = chunk.mul_(grad_loss[start:end].unsqueeze(-1))
            del chunk, soft_targets
        return grad_student, None, None, None, None, None, None, None


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
