"""Writer for fully materialized, fixed-length XTuner soft-pack caches."""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from data.utils import IGNORE_INDEX
from data.xtuner_compat import build_expand_soft_pack_infos, proxy_attention_flops

PACKED_CACHE_FORMAT = "xtuner_softpack_v1"
TOKEN_IDS_NAME = "token_ids.uint32.bin"
LABELS_NAME = "labels.int32.bin"
ATTENTION_MASK_NAME = "attention_mask.uint8.bin"
CHUNKS_NAME = "chunks.jsonl"
META_NAME = "meta.json"
SUMMARY_NAME = "summary.json"


def _sample_document_indices(
    document_lengths: np.ndarray,
    *,
    sample_ratio: float,
    seed: int,
    filter_empty: bool,
) -> np.ndarray:
    if sample_ratio <= 0:
        raise ValueError("dataset.sample_ratio must be positive")
    sampled = np.arange(len(document_lengths), dtype=np.int64)
    if filter_empty:
        sampled = sampled[document_lengths != 0]
    target = int(len(sampled) * sample_ratio)
    if target <= 0:
        return sampled[:0]

    base_repeats = int(sample_ratio)
    repeated = np.tile(sampled, base_repeats) if base_repeats > 0 else sampled[:0]
    remaining = target - len(repeated)
    if remaining <= 0:
        return repeated[:target]

    rng = random.Random(seed)
    choices = rng.sample(range(len(sampled)), remaining)
    extra = sampled[np.asarray(choices, dtype=np.int64)]
    return np.concatenate([repeated, extra])


def _fit_pack(
    document_indices: Sequence[int],
    document_lengths: np.ndarray,
    *,
    pack_max_length: int,
) -> list[int]:
    kept = [int(index) for index in document_indices]
    total_tokens = sum(int(document_lengths[index]) for index in kept)
    while len(kept) > 1 and total_tokens > pack_max_length:
        removed = kept.pop()
        total_tokens -= int(document_lengths[removed])
    return kept


def _materialize_pack(
    document_indices: Sequence[int],
    get_document_tokens: Callable[[int], Sequence[int]],
    document_lengths: np.ndarray,
    *,
    pack_max_length: int,
    pad_token_id: int,
    bos_token_id: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[int], int]:
    kept = _fit_pack(
        document_indices,
        document_lengths,
        pack_max_length=pack_max_length,
    )
    content: list[int] = []
    starts: list[int] = []
    if bos_token_id is not None:
        # Anchor every packed sequence with a BOS so the training regime
        # matches how the model is prompted at evaluation time. Costs one
        # content slot per pack (the final token is truncated).
        content.append(int(bos_token_id))
    for index in kept:
        remaining = pack_max_length - len(content)
        if remaining <= 0:
            break
        starts.append(len(content))
        content.extend(int(token) for token in get_document_tokens(index)[:remaining])

    if not content:
        content = [pad_token_id]
        starts = [0]

    content_length = min(len(content), pack_max_length)
    input_ids = np.full(pack_max_length, pad_token_id, dtype=np.uint32)
    labels = np.full(pack_max_length, IGNORE_INDEX, dtype=np.int32)
    attention_mask = np.zeros(pack_max_length, dtype=np.uint8)
    content_array = np.asarray(content[:pack_max_length], dtype=np.uint32)
    input_ids[:content_length] = content_array
    labels[:content_length] = content_array.astype(np.int32)
    attention_mask[:content_length] = 1
    for start in starts:
        if start < content_length:
            labels[start] = IGNORE_INDEX
    return input_ids, labels, attention_mask, kept, starts, content_length


def write_xtuner_packed_cache(
    *,
    output_dir: Path,
    document_lengths: np.ndarray,
    get_document_tokens: Callable[[int], Sequence[int]],
    pad_token_id: int,
    seed: int,
    pack_max_length: int,
    pack_chunk_size: int,
    pack_workers: int,
    pack_extra_buffer_size: int,
    sample_ratio: float = 1.0,
    filter_empty: bool = True,
    bos_token_id: int | None = None,
    metadata: dict | None = None,
    show_progress: bool = True,
) -> dict:
    """Build deterministic XTuner packs and write fully padded training tensors."""

    if pack_max_length <= 0 or pack_chunk_size <= 0 or pack_workers <= 0:
        raise ValueError("pack lengths, chunk size, and worker count must be positive")
    if pack_extra_buffer_size < 0:
        raise ValueError("pack_extra_buffer_size must be nonnegative")
    document_lengths = np.asarray(document_lengths, dtype=np.int64)
    sampled_indices = _sample_document_indices(
        document_lengths,
        sample_ratio=sample_ratio,
        seed=seed,
        filter_empty=filter_empty,
    )
    if not len(sampled_indices):
        raise ValueError("dataset sampling produced no documents")

    sampled_lengths = document_lengths[sampled_indices]
    proxy_flops = np.asarray(
        [proxy_attention_flops(int(length)) for length in sampled_lengths],
        dtype=np.float64,
    )
    pack_infos = build_expand_soft_pack_infos(
        sampled_lengths,
        proxy_flops,
        seed=seed,
        pack_max_length=pack_max_length,
        pack_workers=pack_workers,
        pack_chunk_size=pack_chunk_size,
        pack_extra_buffer_size=pack_extra_buffer_size,
    )
    if not pack_infos:
        raise ValueError("soft packing produced no packs")

    output_dir.mkdir(parents=True, exist_ok=False)
    token_path = output_dir / TOKEN_IDS_NAME
    label_path = output_dir / LABELS_NAME
    attention_path = output_dir / ATTENTION_MASK_NAME
    chunks_path = output_dir / CHUNKS_NAME
    num_content_tokens = 0
    num_packed_documents = 0

    iterator = enumerate(pack_infos)
    if show_progress:
        iterator = enumerate(tqdm(pack_infos, desc="Writing XTuner soft packs"))

    with (
        token_path.open("wb") as token_handle,
        label_path.open("wb") as label_handle,
        attention_path.open("wb") as attention_handle,
        chunks_path.open("w", encoding="utf-8") as chunks_handle,
    ):
        for pack_index, info in iterator:
            virtual_indices = [int(index) for index in info["indices"]]
            original_indices = [int(sampled_indices[index]) for index in virtual_indices]
            inputs, labels, attention, kept, starts, content_length = _materialize_pack(
                original_indices,
                get_document_tokens,
                document_lengths,
                pack_max_length=pack_max_length,
                pad_token_id=pad_token_id,
                bos_token_id=bos_token_id,
            )
            inputs.tofile(token_handle)
            labels.tofile(label_handle)
            attention.tofile(attention_handle)
            chunks_handle.write(
                json.dumps(
                    {
                        "chunk_index": pack_index,
                        "index": pack_index,
                        "token_offset": pack_index * pack_max_length,
                        "token_length": pack_max_length,
                        "content_length": content_length,
                        "document_indices": kept,
                        "document_starts": starts,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            num_content_tokens += content_length
            num_packed_documents += len(kept)

    num_packs = len(pack_infos)
    num_stored_tokens = num_packs * pack_max_length
    payload = {
        **(metadata or {}),
        "cache_format": PACKED_CACHE_FORMAT,
        "token_ids_file": TOKEN_IDS_NAME,
        "token_ids_dtype": "uint32",
        "labels_file": LABELS_NAME,
        "labels_dtype": "int32",
        "attention_mask_file": ATTENTION_MASK_NAME,
        "attention_mask_dtype": "uint8",
        "chunks_file": CHUNKS_NAME,
        "num_chunks": num_packs,
        "num_tokens": num_stored_tokens,
        "num_content_tokens": num_content_tokens,
        "num_source_documents": len(document_lengths),
        "num_sampled_documents": len(sampled_indices),
        "num_packed_documents": num_packed_documents,
        "sequence_length": pack_max_length,
        "pad_to_max_length": True,
        "pad_token_id": int(pad_token_id),
        "seed": int(seed),
        "sample_ratio": float(sample_ratio),
        "filter_empty": bool(filter_empty),
        "pack_level": "soft",
        "pack_max_length": int(pack_max_length),
        "pack_chunk_size": int(pack_chunk_size),
        "pack_workers": int(pack_workers),
        "pack_extra_buffer_size": int(pack_extra_buffer_size),
        "global_pack": False,
        "bos_per_pack": bos_token_id is not None,
        "bos_token_id": None if bos_token_id is None else int(bos_token_id),
    }
    (output_dir / META_NAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "output_dir": str(output_dir),
        "num_packs": num_packs,
        "num_content_tokens": num_content_tokens,
        "num_stored_tokens": num_stored_tokens,
        "packing_efficiency": num_content_tokens / num_stored_tokens,
    }
    (output_dir / SUMMARY_NAME).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
