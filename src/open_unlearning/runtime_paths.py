"""Resolve large runtime assets without hard-coding them into experiment files."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RuntimePaths:
    """Filesystem layout shared by local launches and RJob.

    Existing in-repository directories remain the compatibility defaults. A
    deployment can move them to a sibling runtime tree by setting only
    ``OU_RUNTIME_ROOT`` or the more specific variables.
    """

    repo: Path
    runtime: Path
    env: Path
    cache: Path
    checkpoints: Path
    runs: Path
    evaluations: Path
    wandb: Path

    @classmethod
    def from_environment(
        cls,
        repo: str | Path | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> RuntimePaths:
        env = os.environ if environment is None else environment
        repo_path = Path(repo or env.get("OU_REPO_ROOT") or repository_root()).resolve()
        runtime_value = env.get("OU_RUNTIME_ROOT")
        runtime = Path(runtime_value).resolve() if runtime_value else repo_path

        def select(variable: str, default: Path) -> Path:
            value = env.get(variable)
            return Path(value).resolve() if value else default

        # Compatibility defaults intentionally point at the current tree. New
        # deployments set OU_RUNTIME_ROOT to open-unlearning-runtime.
        env_default = runtime / "envs" / "open-unlearning"
        cache_default = runtime / ("caches" if runtime_value else "data_cache")
        checkpoints_default = runtime / ("checkpoints" if runtime_value else "saves")
        runs_default = runtime / ("runs" if runtime_value else "outputs")
        eval_default = runtime / "evaluations"
        wandb_default = runtime / ("wandb" if runtime_value else "wandb_offline")
        return cls(
            repo=repo_path,
            runtime=runtime,
            env=select("OU_ENV_ROOT", env_default),
            cache=select("OU_CACHE_ROOT", cache_default),
            checkpoints=select("OU_CHECKPOINT_ROOT", checkpoints_default),
            runs=select("OU_RUN_ROOT", runs_default),
            evaluations=select("OU_EVAL_ROOT", eval_default),
            wandb=select("OU_WANDB_ROOT", wandb_default),
        )

    def as_dict(self) -> dict[str, str]:
        return {name: str(value) for name, value in vars(self).items()}
