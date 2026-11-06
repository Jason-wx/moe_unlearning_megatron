"""Measure the baseline retain-set NLL anchor for PDU's retain_loss_eps.

Loads a pretrained model on all visible GPUs (device_map="auto"), samples packs
from a retain cache (both ``xtuner_softpack_v1`` and ``document_chunk_v1``
formats are supported through ``CachedDocumentChunkDataset``), and computes the
token-pooled masked NLL with exactly the same semantics as the Megatron PDU
retain loss (``src/megatron_unlearning/forward_step.py:151-152``):

- labels right-shifted by one position and doc-start / padding positions set to
  -100, via ``collate_role_batch`` (``src/megatron_unlearning/batching.py``);
- cross entropy in fp32, token-pooled as ``sum(nll * mask) / mask.sum()``.

The result is the anchor that ``retain_loss_eps = (1 + tolerance) * anchor``
is calibrated against.

Usage from the repo root::

    PYTHONPATH=src envs/open-unlearning/bin/python -m tools.measure_retain_nll \
        --model interns2 \
        --output /path/to/retain_nll_anchor_interns2.json

    PYTHONPATH=src envs/open-unlearning/bin/python -m tools.measure_retain_nll \
        --model gemma4 --num-samples 3200
    PYTHONPATH=src envs/open-unlearning/bin/python -m tools.measure_retain_nll \
        --model qwen35 --num-samples 3200

gemma4/qwen35 caches hold ~1024-token document chunks (not 4096-token packs),
so scale ``--num-samples`` up ~4x for a comparable token count.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# The models live on shared storage; never reach out to the HF hub. Must be set
# before transformers is imported (imports below stay hub-free, transformers is
# imported lazily inside load_model).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from data.cached_pretraining import CachedDocumentChunkDataset  # noqa: E402
from megatron_unlearning.batching import (  # noqa: E402
    IGNORE_INDEX,
    collate_role_batch,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

MODEL_PRESETS = {
    "interns2": {
        "model_path": "/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/sgtm/internal/checkpoints/intern/Intern-S2-base05",
        "cache": REPO_ROOT / "data_cache/Xtuner/fineweb_edu_5b_interns2_softpack4096_seed42",
        "sequence_length": 4096,
        # xtuner_softpack_v1 meta carries the pad id; resolved at runtime.
        "pad_token_id": None,
    },
    "gemma4": {
        "model_path": "/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/sgtm/internal/checkpoints/gemma-26b-a4b",
        "cache": REPO_ROOT / "data_cache/gemma4/gemma4_fineweb_edu_5b_max1024_stride1024",
        "sequence_length": 1024,
        "pad_token_id": None,
    },
    "qwen35": {
        "model_path": "/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models-Qwen-Qwen3.5-35B-A3B-Base",
        "cache": REPO_ROOT / "data_cache/qwen35/qwen35_fineweb_edu_5b_max1024_stride1024",
        "sequence_length": 1024,
        "pad_token_id": None,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline retain-NLL anchor measurement (forward pass only)."
    )
    parser.add_argument("--model", required=True, choices=sorted(MODEL_PRESETS))
    parser.add_argument("--model-path", default=None, help="Override the preset model path.")
    parser.add_argument("--cache", default=None, help="Override the preset retain cache dir.")
    parser.add_argument(
        "--model-handler",
        default="AutoModelForCausalLM",
        help="transformers loader class (mirrors model.model_handler); "
        "newer architectures are registered under AutoModelForCausalLM in this env.",
    )
    parser.add_argument("--num-samples", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample-mode",
        choices=("random", "first"),
        default="random",
        help="random: uniform sample (seeded); first: first N chunks in training order.",
    )
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--pad-token-id", type=int, default=None)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument(
        "--ce-chunk",
        type=int,
        default=512,
        help="Positions per fp32 cross-entropy block (bounds the [B, chunk, V] fp32 copy).",
    )
    parser.add_argument("--output", default=None, help="Result JSON path.")
    parser.add_argument(
        "--expected",
        type=float,
        default=None,
        help="Reference anchor to compare against (printed only, e.g. 2.0006 for interns2).",
    )
    return parser.parse_args()


def resolve_pad_token_id(
    model_path: str, args: argparse.Namespace, dataset: CachedDocumentChunkDataset
) -> int:
    """CLI > cache meta (xtuner_softpack_v1) > tokenizer pad/eos."""
    if args.pad_token_id is not None:
        return args.pad_token_id
    meta_pad = dataset.meta.get("pad_token_id")
    if meta_pad is not None:
        print(f"[pad] using pad_token_id={meta_pad} from cache meta")
        return int(meta_pad)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    if pad is None:
        raise ValueError(
            "Cannot resolve a valid pad token id (needed so padded positions stay "
            "in-vocab even though they are masked out); pass --pad-token-id."
        )
    print(f"[pad] using pad_token_id={pad} from tokenizer")
    return int(pad)


def load_model(model_path: str, args: argparse.Namespace, dtype: torch.dtype):
    import transformers

    handler = getattr(transformers, args.model_handler)
    print(f"[model] loading {model_path} via {args.model_handler} ({args.dtype})")
    # use_cache stays default here: some native MoE __init__ signatures in this
    # transformers build reject it; the forward call below disables it instead.
    model = handler.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()
    n_gpus = torch.cuda.device_count()
    placed = sorted(
        {
            d
            for d in getattr(model, "hf_device_map", {}).values()
            if isinstance(d, int) or (isinstance(d, str) and d.startswith("cuda"))
        }
    )
    print(f"[model] visible GPUs: {n_gpus}, placed on: {placed}")
    return model


def input_device(model) -> torch.device:
    """First CUDA placement in module order (device_map hooks move the rest)."""
    for target in getattr(model, "hf_device_map", {}).values():
        if isinstance(target, int):
            return torch.device(f"cuda:{target}")
        if isinstance(target, str) and target.startswith("cuda"):
            return torch.device(target)
    return next(model.parameters()).device


def pad_attention_mask(examples, length: int) -> torch.Tensor:
    mask = torch.zeros((len(examples), length), dtype=torch.long)
    for row, item in enumerate(examples):
        am = item["attention_mask"]
        size = min(int(am.numel()), length)
        mask[row, :size] = am[:size].to(dtype=torch.long)
    return mask


def masked_ce_sum(
    logits: torch.Tensor, labels: torch.Tensor, loss_mask: torch.Tensor, chunk: int
) -> tuple[float, int]:
    """fp32 masked NLL over already next-token-aligned labels, in position blocks."""
    total = 0.0
    count = 0
    seq = logits.shape[1]
    for start in range(0, seq, chunk):
        end = min(start + chunk, seq)
        block = logits[:, start:end, :].float()
        ce = F.cross_entropy(
            block.transpose(1, 2),
            labels[:, start:end],
            ignore_index=IGNORE_INDEX,
            reduction="none",
        )
        m = loss_mask[:, start:end]
        total += float((ce * m).sum())
        count += int(m.sum())
    return total, count


def main() -> None:
    args = parse_args()
    preset = MODEL_PRESETS[args.model]
    model_path = args.model_path or preset["model_path"]
    cache_dir = Path(args.cache) if args.cache else preset["cache"]
    sequence_length = args.sequence_length or preset["sequence_length"]
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    dataset = CachedDocumentChunkDataset(cache_dir)
    num_chunks = len(dataset)
    print(
        f"[data] cache={cache_dir} format={dataset.cache_format} "
        f"chunks={num_chunks:,} seq_len={sequence_length}"
    )
    if args.num_samples > num_chunks:
        raise SystemExit(
            f"--num-samples {args.num_samples} exceeds cache size {num_chunks}"
        )

    if args.sample_mode == "random":
        indices = np.random.default_rng(args.seed).choice(
            num_chunks, size=args.num_samples, replace=False
        )
    else:
        indices = np.arange(args.num_samples)
    print(f"[data] mode={args.sample_mode} n={len(indices)} seed={args.seed}")

    pad_token_id = resolve_pad_token_id(model_path, args, dataset)
    model = load_model(model_path, args, dtype)
    device = input_device(model)

    batch_records = []
    total_sum = 0.0
    total_count = 0
    started = time.time()
    num_batches = (len(indices) + args.batch_size - 1) // args.batch_size
    log_every = max(1, num_batches // 20)

    with torch.no_grad():
        for batch_idx, start in enumerate(range(0, len(indices), args.batch_size)):
            batch_indices = indices[start : start + args.batch_size]
            examples = [dataset[int(i)] for i in batch_indices]
            # Roles are irrelevant here; collate pads, right-shifts labels and
            # derives loss_mask exactly like the training path.
            collated = collate_role_batch(
                examples, None, pad_token_id=pad_token_id, sequence_length=sequence_length
            )
            attention_2d = pad_attention_mask(examples, sequence_length)
            out = model(
                input_ids=collated["input_ids"].to(device),
                attention_mask=attention_2d.to(device),
                use_cache=False,
            )
            logits = out.logits  # [B, L, V]; last position is untrained padding tail
            labels = collated["labels"].to(logits.device)
            loss_mask = collated["loss_mask"].to(logits.device)
            batch_sum, batch_count = masked_ce_sum(logits, labels, loss_mask, args.ce_chunk)
            del out, logits

            total_sum += batch_sum
            total_count += batch_count
            batch_records.append(
                {
                    "batch": batch_idx,
                    "first_chunk_index": int(batch_indices[0]),
                    "nll": batch_sum / batch_count,
                    "num_tokens": batch_count,
                }
            )
            if batch_idx % log_every == 0 or batch_idx == num_batches - 1:
                elapsed = time.time() - started
                print(
                    f"[batch {batch_idx + 1}/{num_batches}] "
                    f"batch_nll={batch_sum / batch_count:.4f} "
                    f"pooled={total_sum / total_count:.4f} elapsed={elapsed:.0f}s"
                )

    batch_nlls = np.array([rec["nll"] for rec in batch_records])
    pooled_nll = total_sum / total_count
    summary = {
        "pooled_nll": pooled_nll,
        "mean_batch_nll": float(batch_nlls.mean()),
        "std_batch_nll": float(batch_nlls.std(ddof=1)) if len(batch_nlls) > 1 else 0.0,
        "se_mean": float(batch_nlls.std(ddof=1) / np.sqrt(len(batch_nlls)))
        if len(batch_nlls) > 1
        else 0.0,
        "min": float(batch_nlls.min()),
        "p25": float(np.percentile(batch_nlls, 25)),
        "median": float(np.percentile(batch_nlls, 50)),
        "p75": float(np.percentile(batch_nlls, 75)),
        "max": float(batch_nlls.max()),
    }
    result = {
        "description": "Baseline retain-NLL anchor for PDU retain_loss_eps calibration",
        "created": datetime.now(timezone.utc).isoformat(),
        "config": {
            "model_preset": args.model,
            "model_path": str(model_path),
            "model_handler": args.model_handler,
            "cache": str(cache_dir),
            "cache_format": dataset.cache_format,
            "cache_num_chunks": num_chunks,
            "num_samples": len(indices),
            "batch_size": args.batch_size,
            "seed": args.seed,
            "sample_mode": args.sample_mode,
            "sequence_length": sequence_length,
            "pad_token_id": pad_token_id,
            "dtype": args.dtype,
            "ce_chunk": args.ce_chunk,
            "num_gpus": torch.cuda.device_count(),
        },
        "summary": summary,
        "total_valid_tokens": total_count,
        "batches": batch_records,
        "sampled_chunk_indices": [int(i) for i in indices],
    }

    print("\n=== retain NLL anchor ===")
    for key, value in summary.items():
        print(f"{key:>16}: {value:.4f}")
    print(f"{'total_tokens':>16}: {total_count:,} in {num_batches} batches")
    print(f"{'elapsed_s':>16}: {time.time() - started:.0f}")
    if args.expected is not None:
        delta = pooled_nll - args.expected
        print(f"{'expected':>16}: {args.expected:.4f} (delta {delta:+.4f})")

    output = args.output or f"retain_nll_anchor_{args.model}.json"
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[out] wrote {output_path}")


if __name__ == "__main__":
    main()
