"""Translate the shared experiment schema into the existing MCore runtime."""

from __future__ import annotations

from pathlib import Path

from open_unlearning.configuration import ExperimentConfig


def to_megatron_mapping(config: ExperimentConfig) -> dict:
    payload = config.payload
    backend = config.backend("megatron")
    training = payload["training"]
    output_root = Path(str(backend["output_root"]))
    output_dir = output_root / str(training["output_name"])
    allowed_algorithm_fields = {
        "gamma",
        "alpha",
        "beta",
        "delta",
        "beta1",
        "beta2",
        "ignore_first_n_tokens",
        "retain_loss_eps",
        "primal_dual",
        "dual_step_size",
        "target_layer",
        "train_layers",
        "steering_coeff",
        "control_norm_mode",
        "control_scale_coeff",
    }
    algorithm = {
        key: value for key, value in payload["algorithm"].items() if key in allowed_algorithm_fields
    }
    algorithm["name"] = config.algorithm_name
    return {
        "model": {
            "path": payload["model"]["path"],
            "num_experts": int(payload["model"]["num_experts"]),
            "trust_remote_code": bool(payload["model"]["trust_remote_code"]),
        },
        "data": {
            "forget_cache": payload["data"]["forget_cache"],
            "retain_cache": payload["data"].get("retain_cache") if config.uses_retain else None,
            "sequence_length": int(payload["data"]["sequence_length"]),
            "pad_token_id": payload["data"]["pad_token_id"],
            "shuffle": bool(payload["data"]["shuffle"]),
        },
        "parallel": dict(backend["parallel"]),
        "reference": dict(backend.get("reference", {})),
        "evaluation": dict(backend.get("evaluation", {})),
        "optimizer": {
            "learning_rate": float(training["learning_rate"]),
            "weight_decay": float(training["weight_decay"]),
            "min_learning_rate": float(training["min_learning_rate"]),
            "gradient_clip": float(training["gradient_clip"]),
            "adam_beta1": float(training["adam_beta1"]),
            "adam_beta2": float(training["adam_beta2"]),
            "adam_eps": float(training["adam_eps"]),
            "lr_scheduler_type": str(training["lr_scheduler_type"]),
            "lr_warmup_ratio": float(training["lr_warmup_ratio"]),
            "lr_warmup_init": float(training["lr_warmup_init"]),
            "lr_decay_steps": str(training["lr_decay_steps"]),
        },
        "training": {
            "micro_batch_size": int(backend["micro_batch_size"]),
            "global_batch_size": int(training["global_batch_size"]),
            "train_steps": int(training["train_steps"]),
            "num_train_epochs": int(training["num_train_epochs"]),
            "save_every_epoch": bool(training["save_every_epoch"]),
            "save_final_checkpoint": bool(training["save_final_checkpoint"]),
            "save_steps": [int(step) for step in training["save_steps"]],
            "save_interval_steps": int(training.get("save_interval_steps", 0)),
            "wandb_mode": str(training["wandb_mode"]),
            "wandb_project": str(training["wandb_project"]),
            "wandb_dir": str(training["wandb_dir"]),
            "seed": int(training["seed"]),
            "bf16": bool(training["bf16"]),
            "trainable_param_mode": str(training["trainable_param_mode"]),
            "output_dir": str(output_dir),
        },
        "algorithm": algorithm,
    }


def launch(config: ExperimentConfig, *, dry_run: bool = False) -> int:
    config.validate(backend="megatron", check_paths=not dry_run)
    mapping = to_megatron_mapping(config)
    if dry_run:
        import yaml

        print(yaml.safe_dump(mapping, sort_keys=False, allow_unicode=True), end="")
        return 0
    from megatron_unlearning.config import from_mapping
    from megatron_unlearning.runtime import train

    run = from_mapping(mapping)
    run.validate(check_paths=True)
    train(run, save_weights=True)
    return 0
