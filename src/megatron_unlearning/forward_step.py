"""Megatron pipeline forward/loss adapter for the first migrated objectives."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import torch
import torch.distributed as dist

from .batching import FORGET_ROLE, RETAIN_ROLE
from .config import FULL_LOGIT_ALGORITHMS, AlgorithmConfig
from .objectives import (
    ceu_objective,
    npo_objective,
    pdu_forget_objective,
    pure_npo_objective,
    satimp_objective,
    simnpo_objective,
    token_npo_objective,
    token_simnpo_objective,
    undial_objective,
    vocab_parallel_token_nll,
    wga_objective,
)
from .rmu import RMUHookState, RMUTargetLayerReached, pipeline_input_tensor


@dataclass
class PDUState:
    retain_preference: float

    def update(self, violation: torch.Tensor, step_size: float) -> None:
        value = violation.detach().float().clone()
        if dist.is_available() and dist.is_initialized():
            from megatron.core import parallel_state

            group = parallel_state.get_data_parallel_group(with_context_parallel=True)
            dist.all_reduce(value, group=group)
            value /= dist.get_world_size(group)
        self.retain_preference = max(0.0, self.retain_preference + step_size * value.item())


def reduce_role_losses(
    token_losses: torch.Tensor,
    loss_mask: torch.Tensor,
    roles: torch.Tensor,
    *,
    algorithm: str,
    gamma: float = 1.0,
    alpha: float = 1.0,
    beta: float = 0.1,
    delta: float = 0.0,
    beta1: float = 5.0,
    beta2: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reduce MCore per-token LM losses with DeepSpeed-equivalent signs."""
    losses = token_losses.float()
    mask = loss_mask.to(device=losses.device, dtype=torch.bool)
    roles = roles.to(device=losses.device)
    if losses.shape != mask.shape:
        raise ValueError(f"loss/mask shapes differ: {losses.shape} != {mask.shape}")
    if roles.ndim != 1 or roles.shape[0] != losses.shape[0]:
        raise ValueError("roles must contain one value per sequence")

    def role_mean(role: int) -> torch.Tensor:
        role_mask = mask & roles.eq(role).unsqueeze(1)
        count = role_mask.sum()
        if not bool(count):
            raise ValueError(f"batch contains no valid tokens for role={role}")
        return (losses * role_mask).sum() / count

    forget_role_mask = mask & roles.eq(FORGET_ROLE).unsqueeze(1)
    forget_nll = role_mean(FORGET_ROLE)
    forget_objective = -forget_nll
    metrics = {
        "forget_nll": forget_nll.detach(),
        "forget_loss": forget_objective.detach(),
        "weighted_forget_loss": (gamma * forget_objective).detach(),
        "forget_valid_tokens": forget_role_mask.sum().detach(),
    }
    if algorithm == "grad_ascent":
        return gamma * forget_objective, metrics
    if algorithm in {"simnpo", "token_simnpo"}:
        objective = simnpo_objective if algorithm == "simnpo" else token_simnpo_objective
        forget_objective, extra = objective(losses, forget_role_mask, beta=beta, delta=delta)
        metrics.update({key: value.detach() for key, value in extra.items()})
        metrics["forget_loss"] = forget_objective.detach()
    elif algorithm == "wga":
        forget_objective, extra = wga_objective(losses, forget_role_mask, beta=beta)
        metrics.update({key: value.detach() for key, value in extra.items()})
        metrics["forget_loss"] = forget_objective.detach()
        if alpha == 0.0:
            metrics["weighted_forget_loss"] = (gamma * forget_objective).detach()
            return gamma * forget_objective, metrics
    elif algorithm == "satimp":
        forget_objective, extra = satimp_objective(
            losses, forget_role_mask, beta1=beta1, beta2=beta2
        )
        metrics.update({key: value.detach() for key, value in extra.items()})
        metrics["forget_loss"] = forget_objective.detach()
        if alpha == 0.0:
            metrics["weighted_forget_loss"] = (gamma * forget_objective).detach()
            return gamma * forget_objective, metrics
    elif algorithm != "grad_diff":
        raise ValueError(f"unsupported token-loss algorithm: {algorithm}")

    retain_nll = role_mean(RETAIN_ROLE)
    retain_role_mask = mask & roles.eq(RETAIN_ROLE).unsqueeze(1)
    metrics["retain_loss"] = retain_nll.detach()
    metrics["retain_valid_tokens"] = retain_role_mask.sum().detach()
    metrics["weighted_forget_loss"] = (gamma * forget_objective).detach()
    metrics["weighted_retain_loss"] = (alpha * retain_nll).detach()
    return gamma * forget_objective + alpha * retain_nll, metrics


def reduce_role_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    roles: torch.Tensor,
    *,
    config: AlgorithmConfig,
    pdu_state: PDUState | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if logits.shape[:-1] != labels.shape or labels.shape != loss_mask.shape:
        raise ValueError("logits, labels, and loss_mask have incompatible shapes")
    roles = roles.to(logits.device)
    forget_rows = roles.eq(FORGET_ROLE)
    if not torch.any(forget_rows):
        raise ValueError("batch contains no forget rows")

    if config.name == "ceu":
        loss, metrics = ceu_objective(
            logits[forget_rows],
            labels[forget_rows],
            ignore_first_n_tokens=config.ignore_first_n_tokens,
        )
        metrics["forget_loss"] = loss.detach()
        metrics["weighted_forget_loss"] = (config.gamma * loss).detach()
        return config.gamma * loss, metrics
    if config.name != "pdu":
        raise ValueError(f"unsupported full-logit algorithm: {config.name}")
    if pdu_state is None:
        raise ValueError("PDU requires mutable dual state")

    retain_rows = roles.eq(RETAIN_ROLE)
    if not torch.any(retain_rows):
        raise ValueError("PDU batch contains no retain rows")
    forget_loss = pdu_forget_objective(logits[forget_rows], loss_mask[forget_rows])
    token_nll = vocab_parallel_token_nll(logits[retain_rows], labels[retain_rows])
    retain_loss = (token_nll * loss_mask[retain_rows]).sum() / loss_mask[retain_rows].sum()
    violation = retain_loss - config.retain_loss_eps
    preference = pdu_state.retain_preference
    loss = config.gamma * forget_loss + preference * violation
    if config.primal_dual:
        pdu_state.update(violation, config.dual_step_size)
    return loss, {
        "forget_loss": forget_loss.detach(),
        "weighted_forget_loss": (config.gamma * forget_loss).detach(),
        "retain_loss": retain_loss.detach(),
        "retain_violation": violation.detach(),
        "retain_preference": loss.detach().new_tensor(preference),
        "weighted_retain_penalty": (preference * violation).detach(),
    }


def reduce_reference_outputs(
    output: torch.Tensor,
    reference_output: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    config: AlgorithmConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    roles = batch["roles"].to(output.device)
    forget_rows = roles.eq(FORGET_ROLE)
    if not torch.any(forget_rows):
        raise ValueError("reference algorithm batch contains no forget rows")

    if config.name in {"npo", "pure_npo", "token_npo"}:
        objective = {
            "npo": npo_objective,
            "pure_npo": pure_npo_objective,
            "token_npo": token_npo_objective,
        }[config.name]
        loss, metrics = objective(
            output[forget_rows].float(),
            reference_output.float(),
            batch["loss_mask"][forget_rows],
            beta=config.beta,
        )
        metrics = {key: value.detach() for key, value in metrics.items()}
        metrics["forget_valid_tokens"] = batch["loss_mask"][forget_rows].sum().detach()
        metrics["forget_loss"] = loss.detach()
        metrics["weighted_forget_loss"] = (config.gamma * loss).detach()
        return config.gamma * loss, metrics

    if config.name != "undial":
        raise ValueError(f"unsupported reference algorithm: {config.name}")
    forget_loss, metrics = undial_objective(
        output[forget_rows],
        reference_output,
        batch["labels"][forget_rows],
        batch["loss_mask"][forget_rows],
        beta=config.beta,
    )
    loss = config.gamma * forget_loss
    metrics = {key: value.detach() for key, value in metrics.items()}
    metrics["forget_loss"] = forget_loss.detach()
    metrics["forget_valid_tokens"] = batch["loss_mask"][forget_rows].sum().detach()
    metrics["weighted_forget_loss"] = (config.gamma * forget_loss).detach()
    retain_rows = roles.eq(RETAIN_ROLE)
    if config.alpha != 0.0:
        if not torch.any(retain_rows):
            raise ValueError("UNDIAL with nonzero alpha requires retain rows")
        retain_nll = vocab_parallel_token_nll(output[retain_rows], batch["labels"][retain_rows])
        retain_mask = batch["loss_mask"][retain_rows]
        retain_loss = (retain_nll * retain_mask).sum() / retain_mask.sum()
        loss = loss + config.alpha * retain_loss
        metrics["retain_loss"] = retain_loss.detach()
        metrics["retain_valid_tokens"] = retain_mask.sum().detach()
        metrics["weighted_retain_loss"] = (config.alpha * retain_loss).detach()
    return loss, metrics


def _loss_func(
    batch: dict[str, torch.Tensor],
    config: AlgorithmConfig,
    pdu_state: PDUState | None,
    reference_outputs,
    loss_scale: float,
    output: torch.Tensor,
):
    if config.name in {"npo", "pure_npo", "token_npo", "undial"}:
        if reference_outputs is None:
            raise ValueError(f"{config.name} requires reference outputs")
        try:
            reference_output = next(reference_outputs)
        except StopIteration as error:
            raise RuntimeError("reference output count is smaller than student batches") from error
        loss, metrics = reduce_reference_outputs(
            output,
            reference_output,
            batch,
            config=config,
        )
    elif config.name in FULL_LOGIT_ALGORITHMS:
        loss, metrics = reduce_role_logits(
            output,
            batch["labels"],
            batch["loss_mask"],
            batch["roles"],
            config=config,
            pdu_state=pdu_state,
        )
    else:
        loss, metrics = reduce_role_losses(
            output,
            batch["loss_mask"],
            batch["roles"],
            algorithm=config.name,
            gamma=config.gamma,
            alpha=config.alpha,
            beta=config.beta,
            delta=config.delta,
            beta1=config.beta1,
            beta2=config.beta2,
        )
    metrics["total_loss"] = loss.detach()
    # Legacy two-item MCore loss return: the schedule scales per microbatch.
    return loss * loss_scale, metrics


def forward_step(
    data_iterator,
    model,
    *,
    algorithm: AlgorithmConfig,
    pdu_state: PDUState | None = None,
    reference_outputs=None,
    loss_scale: float = 1.0,
):
    batch = next(data_iterator)
    output = model(
        input_ids=batch["input_ids"],
        position_ids=batch["position_ids"],
        attention_mask=batch["attention_mask"],
        labels=None if algorithm.name in FULL_LOGIT_ALGORITHMS else batch["labels"],
        loss_mask=batch["loss_mask"],
    )
    return output, partial(
        _loss_func,
        batch,
        algorithm,
        pdu_state,
        reference_outputs,
        loss_scale,
    )


def reference_forward_step(data_iterator, model, *, algorithm: AlgorithmConfig):
    """Forward-only reference step returning forget NLL or local logits."""
    batch = next(data_iterator)
    forget_rows = batch["roles"].eq(FORGET_ROLE)
    reference_batch = {
        key: value[forget_rows] for key, value in batch.items() if key not in {"roles"}
    }
    output = model(
        input_ids=reference_batch["input_ids"],
        position_ids=reference_batch["position_ids"],
        attention_mask=reference_batch["attention_mask"],
        labels=(
            reference_batch["labels"]
            if algorithm.name in {"npo", "pure_npo", "token_npo"}
            else None
        ),
        loss_mask=reference_batch["loss_mask"],
    )

    def collect_reference(value: torch.Tensor, *, non_loss_data: bool = False):
        if not non_loss_data:
            raise RuntimeError("reference schedule must collect non-loss data")
        return value.detach()

    return output, collect_reference


def _rmu_dummy_loss(output: torch.Tensor):
    # Backpropagate a zero main loss through the complete PP graph. The target
    # stage's RMUAuxLossInjector supplies the actual RMU gradient.
    loss = output.float().sum() * 0.0
    return loss, {"rmu_main_loss": loss.detach()}


def rmu_forward_step(data_iterator, model, *, state: RMUHookState):
    batch = next(data_iterator)
    state.begin_student_batch(batch)
    if state.pipeline_position == "after":
        # Keep pipeline input storage intact when MCore pseudo-deallocates the
        # stage output before backward.
        output = pipeline_input_tensor(model).clone()
    else:
        try:
            output = model(
                input_ids=batch["input_ids"],
                position_ids=batch["position_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                loss_mask=batch["loss_mask"],
            )
        except RMUTargetLayerReached as reached:
            output = reached.activation
    return output, _rmu_dummy_loss


def rmu_reference_forward_step(data_iterator, model, *, state: RMUHookState):
    batch = next(data_iterator)
    if state.pipeline_position == "after":
        output = pipeline_input_tensor(model).clone()
    else:
        try:
            output = model(
                input_ids=batch["input_ids"],
                position_ids=batch["position_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                loss_mask=batch["loss_mask"],
            )
        except RMUTargetLayerReached as reached:
            output = reached.activation

    def discard_reference_output(value: torch.Tensor, *, non_loss_data: bool = False):
        if not non_loss_data:
            raise RuntimeError("RMU reference schedule must collect non-loss data")
        return value.detach().new_empty(0)

    return output, discard_reference_output
