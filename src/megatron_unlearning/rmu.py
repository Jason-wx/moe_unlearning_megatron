"""Intermediate-activation plumbing for Megatron RMU.

The target layer may live before the final pipeline stage. RMU therefore
injects its scalar objective into that layer's autograd graph without changing
the activation value sent to later pipeline stages.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

from .batching import FORGET_ROLE, RETAIN_ROLE
from .config import AlgorithmConfig
from .objectives import _tp_info, parallel_rmu_mse


class RMUAuxLossInjector(torch.autograd.Function):
    """Pass an activation through unchanged and backpropagate an auxiliary loss."""

    @staticmethod
    def forward(ctx, activation, auxiliary_loss, loss_scale):
        ctx.save_for_backward(auxiliary_loss, loss_scale)
        return activation

    @staticmethod
    def backward(ctx, grad_activation):
        auxiliary_loss, loss_scale = ctx.saved_tensors
        auxiliary_grad = torch.ones_like(auxiliary_loss) * loss_scale
        return grad_activation, auxiliary_grad, None


class RMUTargetLayerReached(Exception):
    """Internal early-exit carrying the unchanged target-layer activation."""

    def __init__(self, activation: torch.Tensor):
        super().__init__("RMU target layer reached")
        self.activation = activation


def _activation_from_output(output):
    return output[0] if isinstance(output, tuple) else output


def local_transformer_layers(model) -> list[torch.nn.Module]:
    """Return this model chunk's layers, including layers below model wrappers."""
    candidates = []
    for module in model.modules():
        layers = getattr(module, "layers", None)
        if layers is None:
            continue
        layers = list(layers)
        if layers and all(getattr(layer, "layer_number", None) is not None for layer in layers):
            candidates.append(layers)
    if len(candidates) > 1:
        raise RuntimeError("MCore RMU model contains multiple Transformer layer containers")
    return candidates[0] if candidates else []


def local_sequence_mask(
    batch_mask: torch.Tensor,
    activation: torch.Tensor,
    *,
    sequence_parallel: bool,
) -> torch.Tensor:
    """Map a [batch, sequence] mask onto MCore's [local_sequence, batch]."""
    if batch_mask.ndim != 2 or activation.ndim != 3:
        raise ValueError("RMU expects a 2D batch mask and 3D activation")
    if activation.shape[1] != batch_mask.shape[0]:
        raise ValueError("RMU activation batch dimension does not match the mask")
    sequence_mask = batch_mask.transpose(0, 1).contiguous()
    tp_size, tp_rank, _ = _tp_info()
    if sequence_parallel and tp_size > 1:
        if sequence_mask.shape[0] % tp_size:
            raise ValueError("RMU sequence length is not divisible by TP")
        sequence_mask = sequence_mask.chunk(tp_size, dim=0)[tp_rank]
    if sequence_mask.shape != activation.shape[:-1]:
        raise ValueError(
            "RMU local mask/activation shapes differ: "
            f"{sequence_mask.shape} != {activation.shape[:-1]}"
        )
    return sequence_mask


def find_local_transformer_layer(models, target_layer: int):
    """Find a zero-indexed global layer on this PP rank, or return None."""
    matches = []
    for model in models:
        for layer in local_transformer_layers(model):
            layer_number = getattr(layer, "layer_number", None)
            if layer_number is None:
                raise RuntimeError("MCore Transformer layer has no global layer_number")
            if layer_number - 1 == target_layer:
                matches.append(layer)
    if len(matches) > 1:
        raise RuntimeError(f"RMU target layer {target_layer} appears more than once locally")
    return matches[0] if matches else None


def local_pipeline_position(models, target_layer: int) -> str:
    """Return whether this PP rank is before, owns, or follows the target."""
    layer_indices = []
    for model in models:
        for layer in local_transformer_layers(model):
            layer_number = getattr(layer, "layer_number", None)
            if layer_number is None:
                raise RuntimeError("MCore Transformer layer has no global layer_number")
            layer_indices.append(layer_number - 1)
    if not layer_indices:
        raise RuntimeError("MCore RMU pipeline stage contains no Transformer layers")
    if target_layer in layer_indices:
        return "target"
    if max(layer_indices) < target_layer:
        return "before"
    if min(layer_indices) > target_layer:
        return "after"
    raise RuntimeError("RMU target layer falls inside a non-contiguous PP layer gap")


def pipeline_input_tensor(model) -> torch.Tensor:
    """Read the tensor installed by the Megatron pipeline schedule."""
    decoder = None
    for module in model.modules():
        layers = getattr(module, "layers", None)
        if layers is not None and list(layers) and hasattr(module, "input_tensor"):
            decoder = module
            break
    if decoder is None:
        raise RuntimeError("cannot locate MCore decoder.input_tensor")
    value = decoder.input_tensor
    if not isinstance(value, torch.Tensor):
        raise TypeError("Megatron pipeline input tensor is unavailable")
    return value


@dataclass
class RMUHookState:
    algorithm: AlgorithmConfig
    sequence_parallel: bool
    seed: int
    loss_scale: float = 1.0
    reference_activations: deque[torch.Tensor] = field(default_factory=deque)
    current_batch: dict[str, torch.Tensor] | None = None
    latest_metrics: dict[str, torch.Tensor] = field(default_factory=dict)
    step_metrics: list[dict[str, torch.Tensor]] = field(default_factory=list)
    _control_directions: dict[int, torch.Tensor] = field(default_factory=dict)
    _hook_handles: list = field(default_factory=list)
    pipeline_position: str | None = None
    _hooks_enabled: bool = True

    def begin_student_batch(self, batch: dict[str, torch.Tensor]) -> None:
        self.current_batch = batch

    def begin_optimizer_step(self) -> None:
        self.step_metrics.clear()

    @contextmanager
    def hooks_disabled(self) -> Iterator[None]:
        """Temporarily bypass RMU hooks for ordinary model forwards."""
        previous = self._hooks_enabled
        self._hooks_enabled = False
        try:
            yield
        finally:
            self._hooks_enabled = previous

    def reference_hook(self, _module, _inputs, output):
        if not self._hooks_enabled:
            return output
        activation = _activation_from_output(output)
        self.reference_activations.append(activation.detach())
        raise RMUTargetLayerReached(activation)

    def _control_direction(self, activation: torch.Tensor) -> torch.Tensor:
        hidden_size = activation.shape[-1]
        if hidden_size not in self._control_directions:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed)
            direction = torch.rand((1, 1, hidden_size), generator=generator)
            direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            self._control_directions[hidden_size] = direction
        return self._control_directions[hidden_size].to(
            device=activation.device,
            dtype=activation.dtype,
        )

    def _forget_target(
        self,
        activation: torch.Tensor,
        reference_activation: torch.Tensor,
    ) -> torch.Tensor:
        direction = self._control_direction(activation).expand_as(activation)
        mode = self.algorithm.control_norm_mode
        if mode == "fixed":
            return direction * self.algorithm.steering_coeff
        if mode == "ref_activation_l2":
            target_norm = reference_activation.norm(dim=-1, keepdim=True)
        elif mode == "batch_activation_l2":
            target_norm = activation.detach().norm(dim=-1, keepdim=True)
        else:
            raise ValueError(f"unsupported RMU control_norm_mode={mode!r}")
        return direction * target_norm * self.algorithm.control_scale_coeff

    def student_hook(self, _module, _inputs, output):
        if not self._hooks_enabled:
            return output
        if self.current_batch is None:
            raise RuntimeError("RMU student batch was not registered before forward")
        if not self.reference_activations:
            raise RuntimeError("RMU reference activation queue is empty")
        activation = _activation_from_output(output)
        reference = self.reference_activations.popleft().to(
            device=activation.device,
            dtype=activation.dtype,
        )
        if reference.shape != activation.shape:
            raise RuntimeError("RMU student/reference local activation shapes differ")

        roles = self.current_batch["roles"].to(activation.device)
        valid = self.current_batch["activation_mask"].to(activation.device).bool()
        forget_batch_mask = valid & roles.eq(FORGET_ROLE).unsqueeze(1)
        retain_batch_mask = valid & roles.eq(RETAIN_ROLE).unsqueeze(1)
        forget_mask = local_sequence_mask(
            forget_batch_mask,
            activation,
            sequence_parallel=self.sequence_parallel,
        )
        retain_mask = local_sequence_mask(
            retain_batch_mask,
            activation,
            sequence_parallel=self.sequence_parallel,
        )

        control_target = self._forget_target(activation, reference)
        forget_loss, _ = parallel_rmu_mse(activation, control_target, forget_mask)
        retain_loss, _ = parallel_rmu_mse(activation, reference, retain_mask)
        loss = self.algorithm.gamma * forget_loss + self.algorithm.alpha * retain_loss
        self.latest_metrics = {
            "forget_loss": forget_loss.detach(),
            "weighted_forget_loss": (self.algorithm.gamma * forget_loss).detach(),
            "retain_loss": retain_loss.detach(),
            "weighted_retain_loss": (self.algorithm.alpha * retain_loss).detach(),
            "rmu_loss": loss.detach(),
            "total_loss": loss.detach(),
        }
        self.step_metrics.append(self.latest_metrics)
        scaled = activation.new_tensor(self.loss_scale)
        # A custom autograd Function returning its input produces a view. MCore
        # deallocates pipeline outputs and explicitly requires a viewless tensor.
        # clone() preserves the auxiliary autograd edge while satisfying that
        # pipeline invariant.
        injected = RMUAuxLossInjector.apply(activation, loss, scaled).clone()
        self.current_batch = None
        raise RMUTargetLayerReached(injected)

    def install(self, student_models, reference_models) -> bool:
        student_position = local_pipeline_position(student_models, self.algorithm.target_layer)
        reference_position = local_pipeline_position(reference_models, self.algorithm.target_layer)
        if student_position != reference_position:
            raise RuntimeError("student/reference RMU target layers use different PP stages")
        self.pipeline_position = student_position
        student_layer = find_local_transformer_layer(student_models, self.algorithm.target_layer)
        reference_layer = find_local_transformer_layer(
            reference_models, self.algorithm.target_layer
        )
        if (student_layer is None) != (reference_layer is None):
            raise RuntimeError("student/reference RMU target layers are on different PP ranks")
        if student_layer is None:
            return False
        self._hook_handles.extend(
            [
                reference_layer.register_forward_hook(self.reference_hook),
                student_layer.register_forward_hook(self.student_hook),
            ]
        )
        return True

    def assert_reference_queue_empty(self) -> None:
        if self.reference_activations:
            raise RuntimeError(
                f"RMU has {len(self.reference_activations)} unconsumed reference activations"
            )
