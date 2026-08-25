"""Configuration types for the optional Megatron backend.

This module intentionally has no Megatron imports, so configuration and layout
errors can be caught on a CPU login node before an rjob is submitted.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .parallel import ParallelLayout

SUPPORTED_ALGORITHMS = {
    "grad_ascent",
    "grad_diff",
    "simnpo",
    "wga",
    "satimp",
    "ceu",
    "pdu",
    "npo",
    "token_npo",
    "undial",
    "rmu",
}
RETAIN_ALGORITHMS = {"grad_diff", "simnpo", "wga", "satimp", "pdu", "rmu"}
FULL_LOGIT_ALGORITHMS = {"ceu", "pdu", "undial"}
REFERENCE_ALGORITHMS = {"npo", "token_npo", "undial", "rmu"}


@dataclass(frozen=True)
class ModelSource:
    path: str
    num_experts: int
    trust_remote_code: bool = True


@dataclass(frozen=True)
class DataSource:
    forget_cache: str
    retain_cache: str | None = None
    sequence_length: int = 1024
    pad_token_id: int | None = None


@dataclass(frozen=True)
class ParallelConfig:
    world_size: int = 8
    tensor_parallel: int = 4
    pipeline_parallel: int = 2
    context_parallel: int = 1
    expert_parallel: int = 4
    expert_tensor_parallel: int | None = 1
    sequence_parallel: bool = True
    use_distributed_optimizer: bool = True

    def layout(self, *, num_experts: int) -> ParallelLayout:
        return ParallelLayout(
            world_size=self.world_size,
            tensor_parallel=self.tensor_parallel,
            pipeline_parallel=self.pipeline_parallel,
            context_parallel=self.context_parallel,
            expert_parallel=self.expert_parallel,
            expert_tensor_parallel=self.expert_tensor_parallel,
            sequence_parallel=self.sequence_parallel,
            num_experts=num_experts,
        ).validate()


@dataclass(frozen=True)
class ReferenceConfig:
    """Placement policy for algorithms that need a frozen reference model."""

    allocation: str = "same_as_student"
    frozen: bool = True
    path: str | None = None
    logit_window_microbatches: int = 2

    def validate(self) -> None:
        if self.allocation != "same_as_student":
            raise ValueError(
                "Megatron reference models must use the same TP/PP/CP/EP/ETP layout as the student"
            )
        if not self.frozen:
            raise ValueError("Megatron reference models must be frozen")
        if self.logit_window_microbatches <= 0:
            raise ValueError("reference.logit_window_microbatches must be positive")


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 5e-6
    weight_decay: float = 0.0
    min_learning_rate: float = 0.0
    gradient_clip: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8


@dataclass(frozen=True)
class TrainingConfig:
    micro_batch_size: int = 1
    global_batch_size: int = 8
    train_steps: int = -1
    num_train_epochs: int = 1
    save_every_epoch: bool = False
    save_final_checkpoint: bool = False
    save_steps: tuple[int, ...] = ()
    wandb_mode: str = "offline"
    wandb_project: str = "open-unlearning-megatron"
    wandb_dir: str = "wandb_offline"
    seed: int = 3
    bf16: bool = True
    trainable_param_mode: str = "moe_experts"
    output_dir: str = "outputs/megatron"


@dataclass(frozen=True)
class AlgorithmConfig:
    name: str = "grad_ascent"
    gamma: float = 1.0
    alpha: float = 1.0
    beta: float = 0.1
    delta: float = 0.0
    beta1: float = 5.0
    beta2: float = 1.0
    ignore_first_n_tokens: int = 1
    retain_loss_eps: float = 0.3
    primal_dual: bool = True
    dual_step_size: float = 5.0
    target_layer: int = 9
    train_layers: tuple[int, ...] = (7, 8, 9)
    steering_coeff: float = 300.0
    control_norm_mode: str = "fixed"
    control_scale_coeff: float = 1.0


@dataclass(frozen=True)
class MegatronRunConfig:
    model: ModelSource
    data: DataSource
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)

    def validate(self, *, check_paths: bool = True) -> ParallelLayout:
        if self.algorithm.name not in SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"algorithm={self.algorithm.name!r} is not implemented yet; "
                f"supported={sorted(SUPPORTED_ALGORITHMS)}"
            )
        if algorithm_uses_retain(self.algorithm) and not self.data.retain_cache:
            raise ValueError(f"{self.algorithm.name} requires data.retain_cache")
        if self.data.sequence_length <= 1:
            raise ValueError("data.sequence_length must be greater than 1")
        if self.training.micro_batch_size <= 0 or self.training.global_batch_size <= 0:
            raise ValueError("micro/global batch sizes must be positive")
        if self.training.num_train_epochs <= 0:
            raise ValueError("num_train_epochs must be positive")
        if self.training.train_steps == 0 or self.training.train_steps < -1:
            raise ValueError("train_steps must be -1 or a positive integer")
        if any(step <= 0 for step in self.training.save_steps):
            raise ValueError("save_steps must contain only positive integers")
        if len(set(self.training.save_steps)) != len(self.training.save_steps):
            raise ValueError("save_steps must not contain duplicates")
        if self.training.wandb_mode != "offline":
            raise ValueError("Megatron W&B logging must use offline mode")
        if self.algorithm.name == "rmu":
            if self.algorithm.target_layer < 0:
                raise ValueError("RMU target_layer must be nonnegative")
            if not self.algorithm.train_layers or min(self.algorithm.train_layers) < 0:
                raise ValueError("RMU train_layers must contain nonnegative layers")
            if max(self.algorithm.train_layers) > self.algorithm.target_layer:
                raise ValueError("RMU train_layers cannot be after target_layer")
            if self.algorithm.control_norm_mode not in {
                "fixed",
                "ref_activation_l2",
                "batch_activation_l2",
            }:
                raise ValueError("unsupported RMU control_norm_mode")
            if self.data.sequence_length % self.parallel.tensor_parallel:
                raise ValueError(
                    "RMU sequence_length must be divisible by tensor_parallel "
                    "for sequence-parallel activation masks"
                )
        self.reference.validate()

        layout = self.parallel.layout(num_experts=self.model.num_experts)
        data_parallel = layout.data_parallel_size
        micro_global = self.training.micro_batch_size * data_parallel
        if self.training.global_batch_size % micro_global:
            raise ValueError(
                "global_batch_size must be divisible by "
                f"micro_batch_size * data_parallel_size ({micro_global})"
            )

        if check_paths:
            _require_directory(self.model.path, "model.path")
            if self.reference.path:
                _require_directory(self.reference.path, "reference.path")
            _require_cache(self.data.forget_cache, "data.forget_cache")
            if self.data.retain_cache:
                _require_cache(self.data.retain_cache, "data.retain_cache")
        return layout


def algorithm_uses_retain(config: AlgorithmConfig) -> bool:
    return config.name in RETAIN_ALGORITHMS or (config.name == "undial" and config.alpha != 0.0)


def _require_directory(value: str, field_name: str) -> None:
    if not Path(value).is_dir():
        raise FileNotFoundError(f"{field_name} is not a directory: {value}")


def _require_cache(value: str, field_name: str) -> None:
    root = Path(value)
    _require_directory(value, field_name)
    required = ("meta.json", "chunks.jsonl", "token_ids.uint32.bin")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{field_name} is missing {missing}: {value}")


def _section(mapping: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = mapping.get(name, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return dict(value)


def from_mapping(mapping: Mapping[str, Any]) -> MegatronRunConfig:
    training = _section(mapping, "training")
    training["save_steps"] = tuple(int(step) for step in training.get("save_steps", ()))
    return MegatronRunConfig(
        model=ModelSource(**_section(mapping, "model")),
        data=DataSource(**_section(mapping, "data")),
        parallel=ParallelConfig(**_section(mapping, "parallel")),
        reference=ReferenceConfig(**_section(mapping, "reference")),
        optimizer=OptimizerConfig(**_section(mapping, "optimizer")),
        training=TrainingConfig(**training),
        algorithm=AlgorithmConfig(**_section(mapping, "algorithm")),
    )


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = value
    return result


def _load_mapping(path: Path, seen: frozenset[Path]) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"cyclic Megatron config inheritance at {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Megatron config must contain a mapping: {path}")
    payload = dict(payload)
    parent = payload.pop("extends", None)
    if parent is None:
        return payload
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    base = _load_mapping(parent_path, seen | {path})
    return _deep_merge(base, payload)


def load_config(path: str | Path) -> MegatronRunConfig:
    return from_mapping(_load_mapping(Path(path), frozenset()))
