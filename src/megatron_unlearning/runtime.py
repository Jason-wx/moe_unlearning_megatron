"""GPU runtime pieces for the project-owned Megatron-Core training loop.

The module uses lazy Megatron imports so CPU unit tests do not initialize CUDA.
"""

from __future__ import annotations

import os
import random
import shutil
import statistics
import sys
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from data.cached_pretraining import CachedDocumentChunkDataset

from .batching import collate_role_batch
from .config import (
    REFERENCE_ALGORITHMS,
    MegatronRunConfig,
    algorithm_uses_retain,
)
from .forward_step import (
    PDUState,
    forward_step,
    reference_forward_step,
    rmu_forward_step,
    rmu_reference_forward_step,
)
from .model_config import build_mcore_config_dict
from .rmu import RMUHookState, local_transformer_layers

_SUM_STEP_METRICS = {
    "ceu_valid_tokens",
    "undial_valid_tokens",
    "rmu_squared_error_sum",
    "rmu_element_count",
}
_LAST_STEP_METRICS = {"retain_preference"}


def copy_hf_checkpoint_metadata(model_path: str | Path, output_dir: str | Path) -> None:
    """Copy HF model metadata that MCore-Bridge's weight exporter omits."""

    model_path = Path(model_path)
    output_dir = Path(output_dir)
    source_config = model_path / "config.json"
    if not source_config.is_file():
        raise FileNotFoundError(
            f"cannot export an HF checkpoint without source config: {source_config}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_config, output_dir / "config.json")

    # Generation settings are optional in HF checkpoints, but preserve them
    # whenever the source model provides them.
    source_generation_config = model_path / "generation_config.json"
    if source_generation_config.is_file():
        shutil.copy2(
            source_generation_config,
            output_dir / "generation_config.json",
        )


def aggregate_step_metrics(records) -> dict[str, torch.Tensor | float]:
    """Aggregate micro-batch metrics into one optimizer-step record."""

    records = [record for record in records if isinstance(record, dict)]
    keys = {key for record in records for key in record}
    aggregated: dict[str, torch.Tensor | float] = {}
    for key in sorted(keys):
        values = [record[key] for record in records if key in record]
        if not values:
            continue
        if key in _LAST_STEP_METRICS:
            aggregated[key] = values[-1]
            continue
        sum_metric = key in _SUM_STEP_METRICS or key.endswith(("_valid_tokens", "_sequences"))
        if isinstance(values[0], torch.Tensor):
            scalars = [value.detach().float() for value in values]
            aggregated[key] = (
                torch.stack(scalars).sum() if sum_metric else torch.stack(scalars).mean()
            )
        else:
            numeric = [float(value) for value in values]
            aggregated[key] = sum(numeric) if sum_metric else statistics.fmean(numeric)
    return aggregated


class PairedCacheDataset(Dataset):
    """Forget-anchored pairing that is deterministic across model-parallel ranks."""

    def __init__(self, forget: Dataset, retain: Dataset | None, seed: int):
        self.forget = forget
        self.retain = retain
        self.seed = seed

    def __len__(self) -> int:
        return len(self.forget)

    def __getitem__(self, index: int):
        item = {"forget": self.forget[index]}
        if self.retain is not None:
            # Avoid rank-local torch RNG: TP/PP/EP peers must construct the same batch.
            retain_index = random.Random(self.seed + index).randrange(len(self.retain))
            item["retain"] = self.retain[retain_index]
        return item


def initialize_distributed(run: MegatronRunConfig) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Megatron training requires CUDA; use `ou inspect` offline")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    if dist.get_world_size() != run.parallel.world_size:
        raise RuntimeError(
            f"launched world_size={dist.get_world_size()} but config expects "
            f"{run.parallel.world_size}"
        )

    from megatron.core import parallel_state

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=run.parallel.tensor_parallel,
        pipeline_model_parallel_size=run.parallel.pipeline_parallel,
        context_parallel_size=run.parallel.context_parallel,
        expert_model_parallel_size=run.parallel.expert_parallel,
        expert_tensor_parallel_size=(
            run.parallel.expert_tensor_parallel or run.parallel.tensor_parallel
        ),
        order="tp-cp-ep-dp-pp",
    )
    random.seed(run.training.seed)
    torch.manual_seed(run.training.seed)
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(run.training.seed)


def build_model(run: MegatronRunConfig):
    from mcore_bridge import ModelConfig, get_mcore_model

    config = ModelConfig(**build_mcore_config_dict(run))
    models = get_mcore_model(config)
    for model in models:
        model.cuda(torch.cuda.current_device())
    config.bridge.load_weights(models, run.model.path)
    _set_trainable_parameters(
        models,
        run.training.trainable_param_mode,
        model_type=config.hf_model_type,
        train_layers=run.algorithm.train_layers,
    )
    return models, config


def build_reference_model(run: MegatronRunConfig):
    """Load a frozen reference on the already-initialized student MP groups.

    Consequently the reference uses exactly the student's TP/PP/CP/EP/ETP
    allocation; it is never placed on a dedicated rank or a smaller subgroup.
    """
    run.reference.validate()
    from mcore_bridge import ModelConfig, get_mcore_model

    config = ModelConfig(**build_mcore_config_dict(run))
    models = get_mcore_model(config)
    for model in models:
        model.cuda(torch.cuda.current_device())
    config.bridge.load_weights(models, run.reference.path or run.model.path)
    _freeze_reference_parameters(models)
    return models, config


def _freeze_reference_parameters(models) -> None:
    for model in models:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)


def reference_window_ranges(
    algorithm: str,
    num_microbatches: int,
    logit_window_microbatches: int,
) -> list[tuple[int, int]]:
    """Return bounded reference/student windows for one optimizer step."""
    if num_microbatches <= 0 or logit_window_microbatches <= 0:
        raise ValueError("microbatch counts must be positive")
    if algorithm in {"npo", "token_npo"}:
        # NPO variants store only [batch, sequence] token NLL, so preserve a full
        # pipeline schedule and avoid extra pipeline bubbles.
        return [(0, num_microbatches)]
    if algorithm != "undial":
        raise ValueError(f"unsupported reference algorithm: {algorithm}")
    return [
        (start, min(start + logit_window_microbatches, num_microbatches))
        for start in range(0, num_microbatches, logit_window_microbatches)
    ]


def _set_trainable_parameters(
    models,
    mode: str,
    *,
    model_type: str,
    train_layers: tuple[int, ...] | list[int] = (),
) -> None:
    if mode not in {"moe_experts", "rmu_experts"}:
        raise ValueError(f"unsupported trainable_param_mode={mode!r}")
    matched = 0
    for model in models:
        if mode == "rmu_experts":
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            for layer in local_transformer_layers(model):
                layer_number = getattr(layer, "layer_number", None)
                if layer_number is None:
                    raise RuntimeError("MCore Transformer layer has no global layer_number")
                if layer_number - 1 not in train_layers:
                    continue
                for name, parameter in layer.named_parameters():
                    is_routed_expert = ".experts." in f".{name}"
                    is_shared_expert = (
                        ".shared_experts." in f".{name}" or ".shared_expert." in f".{name}"
                    )
                    is_gemma_dense_mlp = "gemma4" in model_type and (
                        ".mlp.linear_fc1." in f".{name}" or ".mlp.linear_fc2." in f".{name}"
                    )
                    is_expert = is_routed_expert or is_shared_expert or is_gemma_dense_mlp
                    parameter.requires_grad_(is_expert)
                    matched += int(is_expert)
            continue
        for name, parameter in model.named_parameters():
            is_routed_expert = ".experts." in name or ".local_experts." in name
            is_shared_expert = ".shared_experts." in name
            # Gemma 4 uses ordinary dense MLP blocks between sparse MoE blocks;
            # the existing HF backend intentionally trains those as well.
            is_gemma_dense_mlp = "gemma4" in model_type and (
                ".mlp.linear_fc1." in name or ".mlp.linear_fc2." in name
            )
            is_expert = is_routed_expert or is_shared_expert or is_gemma_dense_mlp
            parameter.requires_grad_(is_expert)
            matched += int(is_expert)
    if mode == "rmu_experts" and dist.is_available() and dist.is_initialized():
        global_matched = torch.tensor(
            matched,
            device=torch.device("cuda", torch.cuda.current_device()),
            dtype=torch.long,
        )
        dist.all_reduce(global_matched)
        if not global_matched.item():
            raise RuntimeError("RMU found no trainable expert parameters on any PP rank")
    elif not matched:
        raise RuntimeError("MCore model contains no local/shared expert parameters")


def wrap_distributed_and_build_optimizer(run: MegatronRunConfig, models, model_config):
    from megatron.core.distributed import DistributedDataParallel
    from megatron.core.distributed.distributed_data_parallel_config import (
        DistributedDataParallelConfig,
    )
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    from megatron.core.optimizer import get_megatron_optimizer
    from megatron.core.optimizer.optimizer_config import OptimizerConfig

    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        use_distributed_optimizer=run.parallel.use_distributed_optimizer,
    )
    wrapped = [
        DistributedDataParallel(
            config=model_config,
            ddp_config=ddp_config,
            module=model,
            disable_bucketing=index > 0,
        )
        for index, model in enumerate(models)
    ]
    model_config.no_sync_func = wrapped[0].no_sync
    model_config.finalize_model_grads_func = finalize_model_grads

    optim_config = OptimizerConfig(
        optimizer="adam",
        lr=run.optimizer.learning_rate,
        min_lr=run.optimizer.min_learning_rate,
        weight_decay=run.optimizer.weight_decay,
        adam_beta1=run.optimizer.adam_beta1,
        adam_beta2=run.optimizer.adam_beta2,
        adam_eps=run.optimizer.adam_eps,
        clip_grad=run.optimizer.gradient_clip,
        bf16=run.training.bf16,
        fp16=False,
        params_dtype=model_config.params_dtype,
        use_distributed_optimizer=run.parallel.use_distributed_optimizer,
    )
    optimizer = get_megatron_optimizer(optim_config, wrapped)
    return wrapped, optimizer


def build_data_loader(run: MegatronRunConfig):
    from megatron.core import parallel_state
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        run.model.path,
        trust_remote_code=run.model.trust_remote_code,
        local_files_only=True,
    )
    pad_token_id = run.data.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer has no pad_token_id or eos_token_id")

    forget = CachedDocumentChunkDataset(run.data.forget_cache)
    retain = CachedDocumentChunkDataset(run.data.retain_cache) if run.data.retain_cache else None
    dataset = PairedCacheDataset(forget, retain, run.training.seed)
    sampler = DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(with_context_parallel=True),
        rank=parallel_state.get_data_parallel_rank(with_context_parallel=True),
        shuffle=True,
        seed=run.training.seed,
        drop_last=True,
    )

    def collate(items):
        forget_items = [item["forget"] for item in items]
        retain_items = [item["retain"] for item in items] if "retain" in items[0] else None
        batch = collate_role_batch(
            forget_items,
            retain_items,
            pad_token_id=pad_token_id,
            sequence_length=run.data.sequence_length,
        )
        return {key: value.cuda(non_blocking=True) for key, value in batch.items()}

    loader = DataLoader(
        dataset,
        batch_size=run.training.micro_batch_size,
        sampler=sampler,
        collate_fn=collate,
        num_workers=0,
        # collate() already transfers the completed batch to the rank-local
        # CUDA device. Pinning is only valid for CPU tensors and PyTorch 2.7
        # rejects attempting to pin the CUDA tensors returned here.
        pin_memory=False,
        drop_last=True,
    )
    return loader, sampler, tokenizer


def train(run: MegatronRunConfig, *, save_weights: bool = True) -> None:
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

    initialize_distributed(run)
    models, model_config = build_model(run)
    reference_models = (
        build_reference_model(run)[0] if run.algorithm.name in REFERENCE_ALGORITHMS else None
    )
    rmu_state = None
    if run.algorithm.name == "rmu":
        if reference_models is None:
            raise RuntimeError("RMU requires a reference model")
        rmu_state = RMUHookState(
            algorithm=run.algorithm,
            sequence_parallel=run.parallel.sequence_parallel,
            seed=run.training.seed,
        )
        rmu_state.install(models, reference_models)
    models, optimizer = wrap_distributed_and_build_optimizer(run, models, model_config)
    loader, sampler, tokenizer = build_data_loader(run)

    layout = run.parallel.layout(num_experts=run.model.num_experts)
    num_microbatches = run.training.global_batch_size // (
        run.training.micro_batch_size * layout.data_parallel_size
    )
    physical_micro_batch = run.training.micro_batch_size * (
        2 if algorithm_uses_retain(run.algorithm) else 1
    )
    steps_per_epoch = len(loader) // num_microbatches
    max_steps = (
        run.training.train_steps
        if run.training.train_steps > 0
        else steps_per_epoch * run.training.num_train_epochs
    )
    if max_steps <= 0:
        raise RuntimeError("dataset is too small for one global batch")
    unreachable_save_steps = [step for step in run.training.save_steps if step > max_steps]
    if unreachable_save_steps:
        raise ValueError(
            f"save_steps {unreachable_save_steps} exceed the training endpoint {max_steps}"
        )
    if rmu_state is not None:
        rmu_state.loss_scale = 1.0 / num_microbatches

    iterator = iter(loader)
    sampler_epoch = 0
    schedule = get_forward_backward_func()
    pdu_state = (
        PDUState(retain_preference=run.algorithm.alpha) if run.algorithm.name == "pdu" else None
    )
    benchmark_records: list[dict[str, object]] = []

    from megatron.core import parallel_state

    is_wandb_tracker = (
        parallel_state.is_pipeline_last_stage()
        and parallel_state.get_tensor_model_parallel_rank() == 0
        and parallel_state.get_data_parallel_rank() == 0
    )
    wandb_run = None
    progress_bar = None
    if dist.get_rank() == 0:
        print(
            "training_schedule="
            f"steps_per_epoch={steps_per_epoch} max_steps={max_steps} "
            f"micro_batch_size={run.training.micro_batch_size} "
            f"global_batch_size={run.training.global_batch_size} "
            f"micro_batches_per_step={num_microbatches} "
            f"epochs={run.training.num_train_epochs}",
            flush=True,
        )
    if is_wandb_tracker:
        import wandb
        from tqdm.auto import tqdm

        wandb_dir = Path(os.environ.get("WANDB_DIR", run.training.wandb_dir)).resolve()
        wandb_dir.mkdir(parents=True, exist_ok=True)
        wandb_run = wandb.init(
            project=run.training.wandb_project,
            name=Path(run.training.output_dir).name,
            dir=str(wandb_dir),
            mode="offline",
            config=asdict(run),
            tags=["megatron", run.algorithm.name],
        )
        print(f"wandb_offline_run_dir={wandb_run.dir}", flush=True)
        progress_bar = tqdm(
            total=max_steps,
            desc=f"{run.algorithm.name} epoch 1/{run.training.num_train_epochs}",
            unit="step",
            dynamic_ncols=True,
            mininterval=0.5,
            file=sys.stdout,
        )

    def export_model_weights(output_dir: Path) -> None:
        """Export an HF-compatible model/tokenizer without training state."""
        if dist.get_rank() == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(output_dir)
            copy_hf_checkpoint_metadata(run.model.path, output_dir)
        dist.barrier()
        # mcore-bridge exports model weights only. The optimizer, scheduler,
        # gradients, and Megatron checkpoint state are intentionally omitted.
        model_config.bridge.save_weights(models, str(output_dir))
        dist.barrier()
        if dist.get_rank() == 0:
            print(f"exported_model_weights={output_dir}", flush=True)

    def finish_step(step, success, grad_norm, metrics, started_at) -> None:
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started_at
        local = torch.tensor(
            [
                elapsed,
                torch.cuda.max_memory_allocated() / (1024**3),
                torch.cuda.max_memory_reserved() / (1024**3),
            ],
            device=torch.cuda.current_device(),
            dtype=torch.float64,
        )
        gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, local)
        if (
            parallel_state.is_pipeline_last_stage()
            and parallel_state.get_tensor_model_parallel_rank() == 0
        ):
            metric_line = (
                f"loss_metrics_step={step + 1} "
                f"epoch={(step // steps_per_epoch) + 1} "
                f"step_in_epoch={(step % steps_per_epoch) + 1} "
                f"metrics={metrics}"
            )
            if progress_bar is not None:
                progress_bar.write(metric_line)
            else:
                print(metric_line, flush=True)
        per_rank = [value.cpu().tolist() for value in gathered]
        record = {
            "step": step + 1,
            "step_time_seconds": max(value[0] for value in per_rank),
            "allocated_gib_per_rank": [value[1] for value in per_rank],
            "reserved_gib_per_rank": [value[2] for value in per_rank],
        }
        if dist.get_rank() == 0:
            benchmark_records.append(record)
            print(
                f"step={step + 1} success={success} grad_norm={grad_norm} metrics={metrics}",
                flush=True,
            )
            print(f"benchmark_step={record}", flush=True)
        if wandb_run is not None:
            wandb_metrics = {
                "train/optimizer_step": step + 1,
                "train/epoch": (step + 1) / steps_per_epoch,
                "train/step_in_epoch": (step % steps_per_epoch) + 1,
                "train/learning_rate": run.optimizer.learning_rate,
                "train/global_batch_size": run.training.global_batch_size,
                "train/micro_batch_size": run.training.micro_batch_size,
                "train/micro_batches_per_step": num_microbatches,
                "train/steps_per_epoch": steps_per_epoch,
                "time/step_seconds": record["step_time_seconds"],
                "throughput/sequences_per_second": (
                    run.training.global_batch_size / record["step_time_seconds"]
                ),
                "throughput/padded_tokens_per_second": (
                    run.training.global_batch_size
                    * run.data.sequence_length
                    / record["step_time_seconds"]
                ),
                "memory/allocated_peak_gib": max(record["allocated_gib_per_rank"]),
                "memory/reserved_peak_gib": max(record["reserved_gib_per_rank"]),
            }
            if grad_norm is not None:
                if isinstance(grad_norm, torch.Tensor):
                    grad_norm = grad_norm.detach().float().item()
                wandb_metrics["train/grad_norm"] = float(grad_norm)
            for key, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    if value.numel() != 1:
                        continue
                    value = value.detach().float().item()
                try:
                    wandb_metrics[f"train/{key}"] = float(value)
                except (TypeError, ValueError):
                    continue
            for rank, value in enumerate(record["allocated_gib_per_rank"]):
                wandb_metrics[f"memory/rank_{rank}_allocated_peak_gib"] = value
            wandb_run.log(wandb_metrics, step=step + 1)
        if progress_bar is not None:

            def display_scalar(value, default=float("nan")):
                if value is None:
                    return default
                if isinstance(value, torch.Tensor):
                    if value.numel() != 1:
                        return default
                    value = value.detach().float().item()
                return float(value)

            epoch = (step // steps_per_epoch) + 1
            progress_bar.set_description(
                f"{run.algorithm.name} epoch {epoch}/{run.training.num_train_epochs}"
            )
            progress_bar.set_postfix(
                loss=f"{display_scalar(metrics.get('total_loss')):.4f}",
                grad=f"{display_scalar(grad_norm):.3f}",
                sec=f"{record['step_time_seconds']:.2f}",
                mem=f"{max(record['allocated_gib_per_rank']):.1f}GiB",
                refresh=False,
            )
            progress_bar.update(1)
        if not success:
            raise FloatingPointError(f"optimizer rejected step {step + 1}")
        if save_weights and (step + 1) in run.training.save_steps:
            export_model_weights(Path(run.training.output_dir) / f"checkpoint-step-{step + 1}")
        if save_weights and run.training.save_every_epoch and (step + 1) % steps_per_epoch == 0:
            epoch = (step + 1) // steps_per_epoch
            export_model_weights(Path(run.training.output_dir) / f"checkpoint-epoch-{epoch}")

    for step in range(max_steps):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        step_started_at = time.perf_counter()
        step_batches = []
        while len(step_batches) < num_microbatches:
            try:
                step_batches.append(next(iterator))
            except StopIteration:
                sampler_epoch += 1
                sampler.set_epoch(sampler_epoch)
                iterator = iter(loader)
        for model in models:
            model.zero_grad_buffer()
        optimizer.zero_grad()
        losses = []
        if reference_models is None:
            step_fn = partial(
                forward_step,
                algorithm=run.algorithm,
                pdu_state=pdu_state,
            )
            losses = schedule(
                forward_step_func=step_fn,
                data_iterator=iter(step_batches),
                model=models,
                num_microbatches=num_microbatches,
                seq_length=run.data.sequence_length,
                micro_batch_size=physical_micro_batch,
                forward_only=False,
            )
        else:
            if rmu_state is not None:
                rmu_state.begin_optimizer_step()
                schedule(
                    forward_step_func=partial(
                        rmu_reference_forward_step,
                        state=rmu_state,
                    ),
                    data_iterator=iter(step_batches),
                    model=reference_models,
                    num_microbatches=num_microbatches,
                    seq_length=run.data.sequence_length,
                    micro_batch_size=physical_micro_batch,
                    forward_only=True,
                    collect_non_loss_data=True,
                )
                losses = schedule(
                    forward_step_func=partial(rmu_forward_step, state=rmu_state),
                    data_iterator=iter(step_batches),
                    model=models,
                    num_microbatches=num_microbatches,
                    seq_length=run.data.sequence_length,
                    micro_batch_size=physical_micro_batch,
                    forward_only=False,
                )
                rmu_state.assert_reference_queue_empty()
                success, grad_norm, _ = optimizer.step()
                metrics = aggregate_step_metrics(rmu_state.step_metrics)
                if not metrics:
                    metrics = aggregate_step_metrics(losses)
                finish_step(step, success, grad_norm, metrics, step_started_at)
                continue
            reference_step_fn = partial(
                reference_forward_step,
                algorithm=run.algorithm,
            )
            windows = reference_window_ranges(
                run.algorithm.name,
                num_microbatches,
                run.reference.logit_window_microbatches,
            )
            finalize_model_grads = model_config.finalize_model_grads_func
            try:
                for window_index, (start, end) in enumerate(windows):
                    window = step_batches[start:end]
                    window_size = end - start
                    reference_outputs = schedule(
                        forward_step_func=reference_step_fn,
                        data_iterator=iter(window),
                        model=reference_models,
                        num_microbatches=window_size,
                        seq_length=run.data.sequence_length,
                        micro_batch_size=run.training.micro_batch_size,
                        forward_only=True,
                        collect_non_loss_data=True,
                    )
                    step_fn = partial(
                        forward_step,
                        algorithm=run.algorithm,
                        reference_outputs=iter(reference_outputs),
                        loss_scale=window_size / num_microbatches,
                    )
                    # Multiple UNDIAL windows form one optimizer step. Finalize
                    # accumulated model-parallel gradients exactly once, after
                    # the last student window, rather than re-reducing earlier
                    # gradient buffers at every bounded-logit window.
                    is_last_window = window_index == len(windows) - 1
                    model_config.finalize_model_grads_func = (
                        finalize_model_grads if is_last_window else None
                    )
                    window_losses = schedule(
                        forward_step_func=step_fn,
                        data_iterator=iter(window),
                        model=models,
                        num_microbatches=window_size,
                        seq_length=run.data.sequence_length,
                        micro_batch_size=physical_micro_batch,
                        forward_only=False,
                    )
                    losses.extend(window_losses)
            finally:
                model_config.finalize_model_grads_func = finalize_model_grads
        success, grad_norm, _ = optimizer.step()
        metrics = aggregate_step_metrics(losses)
        finish_step(step, success, grad_norm, metrics, step_started_at)

    if dist.get_rank() == 0:
        stable = benchmark_records[10:] or benchmark_records
        step_times = [float(record["step_time_seconds"]) for record in stable]
        summary = {
            "algorithm": run.algorithm.name,
            "completed_steps": len(benchmark_records),
            "stable_step_range": [int(stable[0]["step"]), int(stable[-1]["step"])],
            "stable_step_time_median_seconds": statistics.median(step_times),
            "stable_step_time_mean_seconds": statistics.fmean(step_times),
            "stable_allocated_peak_gib": max(
                max(record["allocated_gib_per_rank"]) for record in stable
            ),
            "stable_reserved_peak_gib": max(
                max(record["reserved_gib_per_rank"]) for record in stable
            ),
            "stable_allocated_peak_gib_per_rank": [
                max(record["allocated_gib_per_rank"][rank] for record in stable)
                for rank in range(dist.get_world_size())
            ],
            "stable_reserved_peak_gib_per_rank": [
                max(record["reserved_gib_per_rank"][rank] for record in stable)
                for rank in range(dist.get_world_size())
            ],
        }
        print(f"benchmark_summary={summary}", flush=True)

    if save_weights and not run.training.save_every_epoch:
        final_output_dir = Path(run.training.output_dir)
        if run.training.save_final_checkpoint:
            final_output_dir /= f"checkpoint-step-{max_steps}"
        export_model_weights(final_output_dir)
    elif save_weights:
        if dist.get_rank() == 0:
            print(
                "save_every_epoch=true; model-only checkpoints were exported "
                "at completed epoch boundaries",
                flush=True,
            )
        dist.barrier()
    else:
        if dist.get_rank() == 0:
            print("save_weights=false; tokenizer and model export skipped", flush=True)
        dist.barrier()
    if progress_bar is not None:
        progress_bar.close()
    if wandb_run is not None:
        wandb_run.finish()
    dist.barrier()
    dist.destroy_process_group()
