"""Translate the shared experiment schema into the existing MCore runtime."""

from __future__ import annotations

from pathlib import Path

from open_unlearning.configuration import ExperimentConfig


def to_megatron_mapping(config: ExperimentConfig) -> dict:
    payload = config.payload
    backend = config.backend("megatron")
    training = payload["training"]
    output_root = Path(str(backend["output_root"]))
    output_dir = output_root / Path(str(training["output_dir"])).name
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
            "trust_remote_code": bool(payload["model"].get("trust_remote_code", True)),
        },
        "data": {
            "forget_cache": payload["data"]["forget_cache"],
            "retain_cache": payload["data"].get("retain_cache") if config.uses_retain else None,
            "sequence_length": int(payload["data"].get("sequence_length", 1024)),
            "pad_token_id": payload["data"].get("pad_token_id"),
        },
        "parallel": dict(backend["parallel"]),
        "reference": dict(backend.get("reference", {})),
        "optimizer": {
            "learning_rate": float(training["learning_rate"]),
            "weight_decay": float(training.get("weight_decay", 0.0)),
            "min_learning_rate": float(training.get("min_learning_rate", 0.0)),
            "gradient_clip": float(training.get("gradient_clip", 1.0)),
            "adam_beta1": float(training.get("adam_beta1", 0.9)),
            "adam_beta2": float(training.get("adam_beta2", 0.999)),
            "adam_eps": float(training.get("adam_eps", 1e-8)),
        },
        "training": {
            "micro_batch_size": int(backend["micro_batch_size"]),
            "global_batch_size": int(training["global_batch_size"]),
            "train_steps": int(training.get("train_steps", -1)),
            "num_train_epochs": int(training.get("num_train_epochs", 1)),
            "save_every_epoch": bool(training.get("save_every_epoch", False)),
            "save_final_checkpoint": bool(training.get("save_final_checkpoint", False)),
            "save_steps": [int(step) for step in training.get("save_steps", ())],
            "wandb_mode": "offline",
            "wandb_project": str(training.get("wandb_project", "OpenUnlearning")),
            "wandb_dir": str(training["wandb_dir"]),
            "seed": int(training.get("seed", 42)),
            "bf16": bool(training.get("bf16", True)),
            "trainable_param_mode": str(training.get("trainable_param_mode", "moe_experts")),
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
