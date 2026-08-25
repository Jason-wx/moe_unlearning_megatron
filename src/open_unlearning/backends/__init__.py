"""Backend dispatch without eager DeepSpeed or Megatron imports."""

from __future__ import annotations

from open_unlearning.configuration import ExperimentConfig


def launch(config: ExperimentConfig, backend: str, *, dry_run: bool = False) -> int:
    if backend == "deepspeed":
        from .deepspeed import launch as launch_deepspeed

        return launch_deepspeed(config, dry_run=dry_run)
    if backend == "megatron":
        from .megatron import launch as launch_megatron

        return launch_megatron(config, dry_run=dry_run)
    raise ValueError(f"unknown backend: {backend}")


__all__ = ["launch"]
