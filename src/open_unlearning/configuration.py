"""Composable experiment configuration shared by both training backends."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .runtime_paths import RuntimePaths
from .trainable_parameters import normalize_trainable_param_mode

_BACKENDS = {"deepspeed", "megatron"}
_SUPPORTED_ALGORITHMS = {
    "deepspeed": {
        "grad_ascent",
        "grad_diff",
        "npo",
        "pure_npo",
        "token_npo",
        "simnpo",
        "token_simnpo",
        "wga",
        "satimp",
        "ceu",
        "pdu",
        "undial",
    },
    "megatron": {
        "grad_ascent",
        "grad_diff",
        "npo",
        "pure_npo",
        "token_npo",
        "simnpo",
        "token_simnpo",
        "wga",
        "satimp",
        "ceu",
        "pdu",
        "undial",
        "rmu",
    },
}
_RETAIN_ALGORITHMS = {"grad_diff", "pdu", "rmu"}
_ALIASES = {
    "token_pure_npo": "token_npo",
}
_ENV_PATTERN = re.compile(r"\$\{([A-Z][A-Z0-9_]*)(?::-([^}]*))?\}")


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = value
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError(f"configuration must be a mapping: {path}")
    return dict(payload)


def _expand_string(value: str, environment: Mapping[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        if name in environment:
            return environment[name]
        if default is not None:
            return default
        return match.group(0)

    return _ENV_PATTERN.sub(replace, value)


def _expand(value: Any, environment: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return _expand_string(value, environment)
    if isinstance(value, list):
        return [_expand(item, environment) for item in value]
    if isinstance(value, Mapping):
        return {key: _expand(item, environment) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    payload: dict[str, Any]

    @property
    def name(self) -> str:
        return str(self.payload["name"])

    @property
    def algorithm_name(self) -> str:
        name = str(self.payload["algorithm"]["name"])
        return _ALIASES.get(name, name)

    def backend(self, name: str) -> dict[str, Any]:
        if name not in _BACKENDS:
            raise ValueError(f"unknown backend {name!r}; choose one of {sorted(_BACKENDS)}")
        value = self.payload.get("backends", {}).get(name)
        if not isinstance(value, Mapping) or not value.get("enabled", True):
            raise ValueError(f"backend {name!r} is not enabled for {self.name}")
        return dict(value)

    @property
    def uses_retain(self) -> bool:
        if self.algorithm_name in _RETAIN_ALGORITHMS:
            return True
        return (
            self.algorithm_name
            in {"wga", "satimp", "undial", "simnpo", "token_simnpo", "pure_npo", "npo"}
            and float(self.payload["algorithm"].get("alpha", 0.0)) != 0.0
        )

    def validate(self, *, backend: str | None = None, check_paths: bool = False) -> None:
        required_sections = ("model", "data", "algorithm", "training", "backends")
        missing = [section for section in required_sections if section not in self.payload]
        if missing:
            raise ValueError(f"experiment is missing sections: {missing}")
        required_fields = {
            "model": (
                "name",
                "path",
                "num_experts",
                "trust_remote_code",
            ),
            "data": (
                "name",
                "forget_cache",
                "retain_cache",
                "sequence_length",
                "pad_token_id",
                "shuffle",
            ),
            "algorithm": ("name",),
            "training": (
                "global_batch_size",
                "train_steps",
                "num_train_epochs",
                "learning_rate",
                "weight_decay",
                "min_learning_rate",
                "adam_beta1",
                "adam_beta2",
                "adam_eps",
                "gradient_clip",
                "seed",
                "bf16",
                "trainable_param_mode",
                "save_steps",
                "save_every_epoch",
                "save_final_checkpoint",
                "wandb_mode",
                "wandb_project",
                "wandb_dir",
            ),
        }
        for section, fields in required_fields.items():
            absent = [field for field in fields if field not in self.payload[section]]
            if absent:
                raise ValueError(f"{section} is missing explicit fields: {absent}")
        configured_backends = set(self.payload["backends"])
        unknown_backends = configured_backends - _BACKENDS
        if unknown_backends:
            raise ValueError(f"unknown configured backends: {sorted(unknown_backends)}")
        if len(configured_backends) != 1:
            raise ValueError("a training config must contain exactly one backend")
        configured_backend = next(iter(configured_backends))
        if configured_backend == "deepspeed":
            if "output_dir" not in self.payload["training"]:
                raise ValueError("training.output_dir must be explicit for DeepSpeed")
            deepspeed = self.payload["backends"]["deepspeed"]
            for section, fields in {
                "model": ("tokenizer_path", "handler"),
                "data": ("handler",),
                "algorithm": ("deepspeed_trainer",),
            }.items():
                absent = [field for field in fields if field not in self.payload[section]]
                if absent:
                    raise ValueError(f"{section} is missing DeepSpeed fields: {absent}")
            for field in (
                "enabled",
                "world_size",
                "micro_batch_size",
                "optimizer",
                "gradient_checkpointing",
                "do_train",
                "do_eval",
                "remove_unused_columns",
                "save_only_model",
                "report_to",
                "ddp_find_unused_parameters",
                "dataloader_drop_last",
                "per_device_eval_batch_size",
                "warmup_steps",
                "lr_scheduler_type",
                "logging_steps",
                "launcher",
                "deepspeed_config",
            ):
                if field not in deepspeed:
                    raise ValueError(f"backends.deepspeed.{field} must be explicit")
            deepspeed_runtime = deepspeed["deepspeed_config"]
            if float(deepspeed_runtime["gradient_clipping"]) != float(
                self.payload["training"]["gradient_clip"]
            ):
                raise ValueError(
                    "training.gradient_clip and "
                    "backends.deepspeed.deepspeed_config.gradient_clipping must match"
                )
            if bool(deepspeed_runtime["bf16"]["enabled"]) != bool(self.payload["training"]["bf16"]):
                raise ValueError(
                    "training.bf16 and backends.deepspeed.deepspeed_config.bf16.enabled must match"
                )
        else:
            output_name = str(self.payload["training"].get("output_name", ""))
            if (
                not output_name
                or output_name in {".", ".."}
                or Path(output_name).name != output_name
            ):
                raise ValueError(
                    "training.output_name must be a safe single directory name for Megatron"
                )
            for field in (
                "lr_scheduler_type",
                "lr_warmup_ratio",
                "lr_warmup_init",
                "lr_decay_steps",
            ):
                if field not in self.payload["training"]:
                    raise ValueError(f"training.{field} must be explicit for Megatron")
            megatron = self.payload["backends"]["megatron"]
            for field in ("enabled", "output_root", "micro_batch_size", "parallel", "reference"):
                if field not in megatron:
                    raise ValueError(f"backends.megatron.{field} must be explicit")
        for field in ("global_batch_size", "num_train_epochs"):
            if int(self.payload["training"].get(field, 0)) <= 0:
                raise ValueError(f"training.{field} must be positive")
        # Periodic saving is an optional extension; legacy configs default to disabled.
        save_interval_steps = int(self.payload["training"].get("save_interval_steps", 0))
        if save_interval_steps < 0:
            raise ValueError("training.save_interval_steps must be nonnegative")
        normalize_trainable_param_mode(self.payload["training"]["trainable_param_mode"])
        algorithm = self.algorithm_name
        retain_cache = self.payload["data"].get("retain_cache")
        if self.uses_retain and not retain_cache:
            raise ValueError(f"algorithm {algorithm} requires data.retain_cache")
        selected_backend = backend or configured_backend
        backend_cfg = self.backend(selected_backend)
        if algorithm not in _SUPPORTED_ALGORITHMS[selected_backend]:
            raise ValueError(f"{selected_backend} does not support algorithm {algorithm}")
        self._validate_batch(selected_backend, backend_cfg)
        if check_paths:
            paths = {
                "model.path": self.payload["model"].get("path"),
                "data.forget_cache": self.payload["data"].get("forget_cache"),
            }
            if self.uses_retain and retain_cache:
                paths["data.retain_cache"] = retain_cache
            for label, value in paths.items():
                if not value or "${" in str(value) or not Path(value).is_dir():
                    raise FileNotFoundError(f"{label} is not a resolved directory: {value}")

    def _validate_batch(self, backend: str, config: Mapping[str, Any]) -> None:
        training = self.payload["training"]
        micro = int(config.get("micro_batch_size", 0))
        if micro <= 0:
            raise ValueError(f"backends.{backend}.micro_batch_size must be positive")
        global_batch = int(training["global_batch_size"])
        if backend == "deepspeed":
            world = int(config.get("world_size", 1))
            divisor = micro * world
        else:
            parallel = config["parallel"]
            dense_parallel = (
                int(parallel["tensor_parallel"])
                * int(parallel["pipeline_parallel"])
                * int(parallel.get("context_parallel", 1))
            )
            world = int(parallel["world_size"])
            if world % dense_parallel:
                raise ValueError("Megatron world size is not divisible by TP*PP*CP")
            divisor = micro * (world // dense_parallel)
        if global_batch % divisor:
            raise ValueError(
                f"global batch {global_batch} is not divisible by {backend} micro-global {divisor}"
            )

    def resolved(self) -> dict[str, Any]:
        return dict(self.payload)


def load_experiment(
    path: str | Path,
    *,
    environment: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> ExperimentConfig:
    path = Path(path).resolve()
    payload = _read_yaml(path)
    if "defaults" in payload:
        raise ValueError("training configs must be self-contained; remove the 'defaults' section")
    merged = payload
    if overrides:
        merged = _deep_merge(merged, overrides)

    env = dict(os.environ if environment is None else environment)
    runtime = RuntimePaths.from_environment(environment=env)
    env.setdefault("OU_REPO_ROOT", str(runtime.repo))
    env.setdefault("OU_RUNTIME_ROOT", str(runtime.runtime))
    env.setdefault("OU_ENV_ROOT", str(runtime.env))
    env.setdefault("OU_CACHE_ROOT", str(runtime.cache))
    env.setdefault("OU_CHECKPOINT_ROOT", str(runtime.checkpoints))
    env.setdefault("OU_RUN_ROOT", str(runtime.runs))
    env.setdefault("OU_EVAL_ROOT", str(runtime.evaluations))
    env.setdefault("OU_WANDB_ROOT", str(runtime.wandb))
    merged = _expand(merged, env)
    config = ExperimentConfig(path=path, payload=merged)
    config.validate(check_paths=False)
    return config
