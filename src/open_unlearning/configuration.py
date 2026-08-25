"""Composable experiment configuration shared by both training backends."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .runtime_paths import RuntimePaths, repository_root

_COMPONENT_DIRS = {
    "model": "models",
    "dataset": "datasets",
    "algorithm": "algorithms",
    "profile": "profiles",
}
_BACKENDS = {"deepspeed", "megatron"}
_RETAIN_ALGORITHMS = {"grad_diff", "simnpo", "wga", "satimp", "pdu", "rmu"}
_ALIASES = {
    "pure_npo": "npo",
    "token_pure_npo": "token_npo",
    "token_simnpo": "simnpo",
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
            self.algorithm_name == "undial"
            and float(self.payload["algorithm"].get("alpha", 0.0)) != 0.0
        )

    def validate(self, *, backend: str | None = None, check_paths: bool = False) -> None:
        required_sections = ("model", "data", "algorithm", "training", "backends")
        missing = [section for section in required_sections if section not in self.payload]
        if missing:
            raise ValueError(f"experiment is missing sections: {missing}")
        for field in ("global_batch_size", "num_train_epochs"):
            if int(self.payload["training"].get(field, 0)) <= 0:
                raise ValueError(f"training.{field} must be positive")
        algorithm = self.algorithm_name
        retain_cache = self.payload["data"].get("retain_cache")
        alpha = float(self.payload["algorithm"].get("alpha", 0.0))
        if algorithm in _RETAIN_ALGORITHMS and not retain_cache:
            raise ValueError(f"algorithm {algorithm} requires data.retain_cache")
        if algorithm == "undial" and alpha != 0.0 and not retain_cache:
            raise ValueError("UNDIAL with nonzero alpha requires data.retain_cache")
        if backend is not None:
            backend_cfg = self.backend(backend)
            supported = set(backend_cfg.get("supported_algorithms", ()))
            if supported and algorithm not in supported:
                raise ValueError(f"{backend} does not support algorithm {algorithm}")
            self._validate_batch(backend, backend_cfg)
        if check_paths:
            paths = {
                "model.path": self.payload["model"].get("path"),
                "data.forget_cache": self.payload["data"].get("forget_cache"),
            }
            if retain_cache:
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
    config_root: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
    components: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> ExperimentConfig:
    path = Path(path).resolve()
    root = Path(config_root or repository_root() / "configs").resolve()
    payload = _read_yaml(path)
    defaults = payload.pop("defaults", {})
    if not isinstance(defaults, Mapping):
        raise TypeError("experiment.defaults must be a mapping")
    component_names = dict(defaults)
    for kind, name in (components or {}).items():
        if kind not in _COMPONENT_DIRS:
            raise ValueError(f"unknown component kind {kind!r}")
        component_names[kind] = name

    merged: dict[str, Any] = {}
    for kind, directory in _COMPONENT_DIRS.items():
        name = component_names.get(kind)
        if name:
            merged = _deep_merge(merged, _read_yaml(root / directory / f"{name}.yaml"))
    backend_defaults = defaults.get("backends", {})
    if not isinstance(backend_defaults, Mapping):
        raise TypeError("experiment.defaults.backends must be a mapping")
    for backend, name in backend_defaults.items():
        if backend not in _BACKENDS:
            raise ValueError(f"unknown backend component {backend!r}")
        component = _read_yaml(root / "backends" / backend / f"{name}.yaml")
        merged = _deep_merge(merged, component)
    merged = _deep_merge(merged, payload)
    if components and not (overrides or {}).get("name"):
        generated_name = "-".join(
            str(merged[section]["name"]).replace("_", "-")
            for section in ("model", "algorithm", "data")
        )
        merged["name"] = generated_name
        merged.setdefault("training", {})["output_dir"] = "${OU_CHECKPOINT_ROOT}/" + generated_name
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
