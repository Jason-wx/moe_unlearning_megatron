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

from open_unlearning.trainable_parameters import normalize_trainable_param_mode

from .parallel import ParallelLayout

SUPPORTED_ALGORITHMS = {
    "grad_ascent",
    "grad_diff",
    "simnpo",
    "token_simnpo",
    "wga",
    "satimp",
    "ceu",
    "pdu",
    "npo",
    "pure_npo",
    "token_npo",
    "undial",
    "rmu",
}
RETAIN_ALGORITHMS = {"grad_diff", "pdu", "rmu"}
FULL_LOGIT_ALGORITHMS = {"ceu", "pdu", "undial"}
REFERENCE_ALGORITHMS = {"npo", "pure_npo", "token_npo", "undial", "rmu"}
SUPPORTED_EVAL_TASKS = {"wmdp_bio", "wmdp_cyber", "mmlu"}


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
    shuffle: bool = True


@dataclass(frozen=True)
class ParallelConfig:
    world_size: int = 8
    tensor_parallel: int = 4
    pipeline_parallel: int = 2
    context_parallel: int = 1
    expert_parallel: int = 4
    expert_tensor_parallel: int | None = 1
    sequence_parallel: bool = True
    activation_recompute: str | None = None
    use_distributed_optimizer: bool = True
    num_layers_in_first_pipeline_stage: int | None = None
    num_layers_in_last_pipeline_stage: int | None = None

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

    def validate_pipeline_partition(self) -> None:
        uneven = (
            self.num_layers_in_first_pipeline_stage,
            self.num_layers_in_last_pipeline_stage,
        )
        if all(value is None for value in uneven):
            return
        if self.pipeline_parallel <= 1:
            raise ValueError("uneven pipeline stages require pipeline_parallel > 1")
        if any(value is not None and value <= 0 for value in uneven):
            raise ValueError("uneven pipeline layer counts must be positive")


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
class EvaluationConfig:
    """Optional in-training online evaluation on lm-eval MCQ benchmarks.

    Disabled by default so the many existing configs keep working unchanged: a
    config without an ``evaluation`` block leaves ``enabled=False`` and no other
    field is consulted. When enabled, the training loop scores the resident GPU
    parameters (no checkpoint export) on the requested tasks at the given steps.
    """

    enabled: bool = False
    steps: tuple[int, ...] = ()
    tasks: tuple[str, ...] = ()
    limit: int | None = None
    # Per-task override of ``limit`` (lm-eval semantics). Value is passed straight
    # to lm-eval's ``limit``, which for a *group* task like mmlu caps each of its
    # 57 leaf subtasks (deterministic head slice, identical across ranks -> EP8
    # stays in lockstep). So mmlu: 35 keeps ~35 questions per subtask (35x57~=2000
    # total, stratified) while a task absent from the map falls back to ``limit``
    # (None => full). Stored as a tuple of (task, limit) pairs for immutability.
    task_limits: tuple[tuple[str, int], ...] = ()
    num_fewshot: int = 0
    # Questions scored per online-eval forward pass. Purely an eval-speed knob:
    # each forward packs ``batch_size`` MCQ questions (x their answer choices)
    # instead of one, cutting the forward count ~batch_size-fold (e.g. mmlu's
    # 14042 questions). Independent of training ``micro_batch_size``. Default 1
    # keeps the historical one-question-per-forward behaviour.
    batch_size: int = 1
    # Optional directory for a live, human-readable JSONL log of the online-eval
    # results (one record per eval step, appended by rank 0 as training runs so
    # the file can be tailed). ``None`` disables the file (results still print /
    # log to wandb).
    output_dir: str | None = None

    def validate(self) -> None:
        if not self.enabled:
            return
        if not self.steps:
            raise ValueError("evaluation.steps must be non-empty when evaluation is enabled")
        if any(step <= 0 for step in self.steps):
            raise ValueError("evaluation.steps must contain only positive integers")
        if len(set(self.steps)) != len(self.steps):
            raise ValueError("evaluation.steps must not contain duplicates")
        if not self.tasks:
            raise ValueError("evaluation.tasks must be non-empty when evaluation is enabled")
        unsupported = [task for task in self.tasks if task not in SUPPORTED_EVAL_TASKS]
        if unsupported:
            raise ValueError(
                f"evaluation.tasks contains unsupported tasks {unsupported}; "
                f"supported={sorted(SUPPORTED_EVAL_TASKS)}"
            )
        if len(set(self.tasks)) != len(self.tasks):
            raise ValueError("evaluation.tasks must not contain duplicates")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("evaluation.limit must be a positive integer or null")
        limit_tasks = [name for name, _ in self.task_limits]
        if len(set(limit_tasks)) != len(limit_tasks):
            raise ValueError("evaluation.task_limits must not contain duplicate tasks")
        for name, value in self.task_limits:
            if name not in SUPPORTED_EVAL_TASKS:
                raise ValueError(
                    f"evaluation.task_limits references unsupported task {name!r}; "
                    f"supported={sorted(SUPPORTED_EVAL_TASKS)}"
                )
            if value <= 0:
                raise ValueError("evaluation.task_limits values must be positive integers")
        if self.num_fewshot < 0:
            raise ValueError("evaluation.num_fewshot must be nonnegative")
        if self.batch_size <= 0:
            raise ValueError("evaluation.batch_size must be a positive integer")

    def limit_for(self, task: str) -> int | None:
        """Resolve the lm-eval ``limit`` for ``task``: a per-task override from
        ``task_limits`` if present, otherwise the shared ``limit`` (default None)."""
        for name, value in self.task_limits:
            if name == task:
                return value
        return self.limit


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 5e-6
    weight_decay: float = 0.0
    min_learning_rate: float = 0.0
    gradient_clip: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    lr_scheduler_type: str = "cosine"
    lr_warmup_ratio: float = 0.03
    lr_warmup_init: float = 0.0
    lr_decay_steps: str = "auto"


@dataclass(frozen=True)
class TrainingConfig:
    micro_batch_size: int = 1
    global_batch_size: int = 8
    train_steps: int = -1
    num_train_epochs: int = 1
    save_every_epoch: bool = False
    save_final_checkpoint: bool = False
    save_steps: tuple[int, ...] = ()
    save_interval_steps: int = 0
    wandb_mode: str = "offline"
    wandb_project: str = "open-unlearning-megatron"
    wandb_dir: str = "wandb_offline"
    seed: int = 3
    bf16: bool = True
    trainable_param_mode: str = "expert"
    output_dir: str = "outputs/megatron"


@dataclass(frozen=True)
class AlgorithmConfig:
    name: str = "grad_ascent"
    gamma: float = 1.0
    alpha: float = 0.0
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
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

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
        if self.training.save_interval_steps < 0:
            raise ValueError("save_interval_steps must be nonnegative")
        if self.training.wandb_mode != "offline":
            raise ValueError("Megatron W&B logging must use offline mode")
        if self.optimizer.lr_scheduler_type != "cosine":
            raise ValueError("only cosine lr_scheduler_type is currently supported")
        if not 0.0 <= self.optimizer.lr_warmup_ratio < 1.0:
            raise ValueError("lr_warmup_ratio must be in [0, 1)")
        if self.optimizer.lr_warmup_init < 0.0:
            raise ValueError("lr_warmup_init must be nonnegative")
        if self.optimizer.lr_warmup_init > self.optimizer.learning_rate:
            raise ValueError("lr_warmup_init cannot exceed learning_rate")
        if not 0.0 <= self.optimizer.min_learning_rate <= self.optimizer.learning_rate:
            raise ValueError("min_learning_rate must be between 0 and learning_rate")
        if self.optimizer.lr_decay_steps != "auto":
            raise ValueError("lr_decay_steps must be 'auto' so it follows the training endpoint")
        normalize_trainable_param_mode(self.training.trainable_param_mode)
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
            if self.parallel.activation_recompute == "full":
                raise ValueError(
                    "RMU's intermediate-layer forward hooks cannot run inside "
                    "full activation recomputation: the hook computes its activation "
                    "loss during the checkpoint's no-grad forward, so the returned "
                    "activation has no autograd edge and the optimizer receives zero "
                    "gradients. Use activation_recompute=null (or selective only if "
                    "verified) for RMU."
                )
        self.reference.validate()
        self.parallel.validate_pipeline_partition()

        self.evaluation.validate()
        if self.evaluation.enabled and (
            self.parallel.pipeline_parallel != 1 or self.parallel.context_parallel != 1
        ):
            raise ValueError(
                "online evaluation currently requires pipeline_parallel==1 and context_parallel==1"
            )

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
    return config.name in RETAIN_ALGORITHMS or (
        config.name in {"wga", "satimp", "undial", "simnpo", "token_simnpo", "pure_npo", "npo"}
        and config.alpha != 0.0
    )


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
    training["save_interval_steps"] = int(training.get("save_interval_steps", 0))
    evaluation = _section(mapping, "evaluation")
    if "steps" in evaluation:
        evaluation["steps"] = tuple(int(step) for step in evaluation["steps"])
    if "tasks" in evaluation:
        evaluation["tasks"] = tuple(str(task) for task in evaluation["tasks"])
    if evaluation.get("limit") is not None:
        evaluation["limit"] = int(evaluation["limit"])
    if "task_limits" in evaluation:
        task_limits = evaluation["task_limits"]
        if not isinstance(task_limits, Mapping):
            raise TypeError("evaluation.task_limits must be a mapping")
        evaluation["task_limits"] = tuple(
            (str(name), int(value)) for name, value in task_limits.items()
        )
    if "batch_size" in evaluation:
        evaluation["batch_size"] = int(evaluation["batch_size"])
    return MegatronRunConfig(
        model=ModelSource(**_section(mapping, "model")),
        data=DataSource(**_section(mapping, "data")),
        parallel=ParallelConfig(**_section(mapping, "parallel")),
        reference=ReferenceConfig(**_section(mapping, "reference")),
        optimizer=OptimizerConfig(**_section(mapping, "optimizer")),
        training=TrainingConfig(**training),
        algorithm=AlgorithmConfig(**_section(mapping, "algorithm")),
        evaluation=EvaluationConfig(**evaluation),
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
