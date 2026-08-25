"""Build a single canonical Accelerate/DeepSpeed launch command."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from open_unlearning.configuration import ExperimentConfig
from open_unlearning.runtime_paths import RuntimePaths


def _bool(value: object) -> str:
    return "true" if bool(value) else "false"


def build_command(config: ExperimentConfig) -> list[str]:
    payload = config.payload
    backend = config.backend("deepspeed")
    training = payload["training"]
    if training.get("save_steps"):
        raise ValueError(
            "step checkpoints combined with epoch checkpoints are currently supported "
            "only by the Megatron backend"
        )
    model = payload["model"]
    data = payload["data"]
    algorithm = payload["algorithm"]
    paths = RuntimePaths.from_environment()
    accelerate = os.environ.get("OU_ACCELERATE")
    if not accelerate:
        candidate = paths.env / "bin" / "accelerate"
        accelerate = str(
            candidate if candidate.exists() else shutil.which("accelerate") or candidate
        )
    config_file = Path(str(backend["accelerate_config"]))
    if not config_file.is_absolute():
        config_file = paths.repo / config_file

    world_size = int(backend["world_size"])
    micro_batch = int(backend["micro_batch_size"])
    global_batch = int(training["global_batch_size"])
    grad_accum = global_batch // (world_size * micro_batch)
    requested_trainer = str(algorithm.get("deepspeed_trainer", config.algorithm_name))
    trainer_names = {
        "grad_ascent": "GradAscent",
        "grad_diff": "GradDiff",
        "npo": "NPO",
        "simnpo": "SimNPO",
        "undial": "UNDIAL",
        "ceu": "CEU",
        "pdu": "PDU",
        "rmu": "RMU",
        "dpo": "DPO",
        "altpo": "DPO",
        "wga": "WGA",
        "satimp": "SatImp",
    }
    trainer_name = trainer_names.get(requested_trainer.lower(), requested_trainer)

    overrides = [
        "experiment=unlearn/wmdp/default.yaml",
        f"model={model['hydra_name']}",
        f"model.model_args.pretrained_model_name_or_path={model['path']}",
        f"model.tokenizer_args.pretrained_model_name_or_path={model['path']}",
        f"trainer={trainer_name}",
        "~trainer.method_args",
        f"+trainer.method_args.trainable_param_mode={training.get('trainable_param_mode', 'moe_experts')}",
        f"task_name={config.name}",
        "data_split=bio",
        "data.forget.WMDP_forget.handler=CachedDocumentChunkDataset",
        f"+data.forget.WMDP_forget.args.cache_dir={data['forget_cache']}",
        "data.anchor=forget",
        f"trainer.args.output_dir={training['output_dir']}/deepspeed",
        "trainer.args.do_train=true",
        "trainer.args.do_eval=false",
        "trainer.args.remove_unused_columns=false",
        f"trainer.args.per_device_train_batch_size={micro_batch}",
        "trainer.args.per_device_eval_batch_size=1",
        f"trainer.args.gradient_accumulation_steps={grad_accum}",
        f"trainer.args.learning_rate={training['learning_rate']}",
        f"trainer.args.weight_decay={training.get('weight_decay', 0.0)}",
        f"trainer.args.optim={backend.get('optimizer', 'adamw_torch')}",
        f"+trainer.args.adam_beta1={training.get('adam_beta1', 0.9)}",
        f"+trainer.args.adam_beta2={training.get('adam_beta2', 0.999)}",
        f"+trainer.args.adam_epsilon={training.get('adam_eps', 1e-8)}",
        f"trainer.args.max_grad_norm={training.get('gradient_clip', 1.0)}",
        f"trainer.args.max_steps={training.get('train_steps', -1)}",
        f"trainer.args.num_train_epochs={training.get('num_train_epochs', 1)}",
        "+trainer.args.warmup_steps=0",
        "trainer.args.lr_scheduler_type=constant",
        "trainer.args.logging_steps=1",
        f"trainer.args.save_strategy={'epoch' if training.get('save_every_epoch') else 'no'}",
        f"trainer.args.bf16={_bool(training.get('bf16', True))}",
        f"trainer.args.gradient_checkpointing={_bool(backend.get('gradient_checkpointing', True))}",
        "trainer.args.ddp_find_unused_parameters=true",
        "trainer.args.report_to=wandb",
        f"+trainer.args.run_name={config.name}-deepspeed",
        f"trainer.args.seed={training.get('seed', 42)}",
        f"+trainer.args.data_seed={training.get('seed', 42)}",
        "+trainer.args.dataloader_drop_last=true",
    ]
    if config.uses_retain and data.get("retain_cache"):
        overrides.extend(
            [
                "data.retain.WMDP_retain.handler=CachedDocumentChunkDataset",
                f"+data.retain.WMDP_retain.args.cache_dir={data['retain_cache']}",
            ]
        )
    else:
        overrides.append("~data.retain")
    deepspeed_method_keys = {
        # The shared/Megatron spelling is shorter; the HF trainer keeps its
        # historical public argument name.
        "ignore_first_n_tokens": "ignore_first_n_answer_tokens",
    }
    for key, value in algorithm.items():
        if key in {"name", "deepspeed_trainer"}:
            continue
        if isinstance(value, list):
            continue
        method_key = deepspeed_method_keys.get(key, key)
        overrides.append(f"+trainer.method_args.{method_key}={value}")

    return [
        accelerate,
        "launch",
        "--config_file",
        str(config_file),
        str(paths.repo / "src" / "train.py"),
        "--config-name=unlearn.yaml",
        *overrides,
    ]


def launch(config: ExperimentConfig, *, dry_run: bool = False) -> int:
    config.validate(backend="deepspeed", check_paths=not dry_run)
    command = build_command(config)
    if dry_run:
        import shlex

        print(shlex.join(command))
        return 0
    environment = dict(os.environ)
    environment.setdefault("WANDB_MODE", "offline")
    environment.setdefault("WANDB_DIR", str(config.payload["training"]["wandb_dir"]))
    return subprocess.run(command, env=environment, check=False).returncode
