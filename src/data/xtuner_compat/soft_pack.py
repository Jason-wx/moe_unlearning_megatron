# Copyright (c) OpenMMLab. All rights reserved.
"""Locally vendored pure-text ExpandSoftPack packing behavior.

This module intentionally keeps XTuner's ordering and gap-filling semantics,
including its chunk-local search and ``torch.randperm`` seed behavior.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import shared_memory

import numpy as np
import torch


def closest_sum_indices(buffer, value):
    buffer = np.asarray(buffer)
    sorted_indices = np.argsort(buffer)
    closest_sum = 0
    closest_indices: list[int] = []
    for idx in sorted_indices:
        closest_sum += buffer[idx]
        if closest_sum <= value:
            closest_indices.append(int(idx))
        if closest_sum >= value:
            break
    return closest_indices


def get_pack_chunk_infos(
    inds,
    dataset_id,
    target,
    pack_extra_buffer_size,
    num_tokens=None,
    proxy_attn_flops=None,
    shm_name=None,
    shape=None,
    dtype=None,
    proxy_attn_flops_shm_name=None,
    proxy_attn_flops_dtype=None,
):
    token_shm = None
    flop_shm = None
    if num_tokens is None:
        token_shm = shared_memory.SharedMemory(name=shm_name)
        num_tokens = np.ndarray(shape, dtype=dtype, buffer=token_shm.buf)
        flop_shm = shared_memory.SharedMemory(name=proxy_attn_flops_shm_name)
        proxy_attn_flops = np.ndarray(
            shape,
            dtype=proxy_attn_flops_dtype,
            buffer=flop_shm.buf,
        )

    item_buffer: list[int] = []
    length_buffer: list[int] = []
    pack_proxy_attn_flops = 0.0
    pack_infos: list[dict] = []

    try:
        while inds:
            shuffled_index = inds.pop()
            if num_tokens[shuffled_index] + sum(length_buffer) <= target:
                item_buffer.append(shuffled_index)
                length_buffer.append(int(num_tokens[shuffled_index]))
                pack_proxy_attn_flops += float(proxy_attn_flops[shuffled_index])
                continue

            if item_buffer:
                if sum(length_buffer) != target and pack_extra_buffer_size > 0:
                    buffer_index = inds[-pack_extra_buffer_size:]
                    buffer = num_tokens[buffer_index]
                    closest_indices = closest_sum_indices(
                        buffer,
                        target - sum(length_buffer),
                    )
                    indices_to_remove = []
                    for closest_index in closest_indices:
                        indices_to_remove.append(
                            closest_index + len(inds) - len(buffer_index)
                        )
                        selected = int(buffer_index[closest_index])
                        item_buffer.append(selected)
                        length_buffer.append(int(num_tokens[selected]))
                        pack_proxy_attn_flops += float(proxy_attn_flops[selected])
                    for index in sorted(indices_to_remove, reverse=True):
                        inds.pop(index)

                pack_infos.append(
                    {
                        "dataset_id": dataset_id,
                        "indices": item_buffer,
                        # XTuner stores its cost proxy in the field named longest.
                        "longest": int(pack_proxy_attn_flops),
                    }
                )

            item_buffer = [int(shuffled_index)]
            length_buffer = [int(num_tokens[shuffled_index])]
            pack_proxy_attn_flops = float(proxy_attn_flops[shuffled_index])

        if item_buffer:
            pack_infos.append(
                {
                    "dataset_id": dataset_id,
                    "indices": item_buffer,
                    "longest": int(pack_proxy_attn_flops),
                }
            )
        return pack_infos
    finally:
        if token_shm is not None:
            token_shm.close()
        if flop_shm is not None:
            flop_shm.close()


def get_pack_infos_by_expand_soft_split(
    inds: list[int],
    dataset_id: int,
    num_tokens: np.ndarray,
    proxy_attn_flops: np.ndarray,
    pack_max_length: int,
    pack_workers: int = 8,
    pack_chunk_size: int = 10_000,
    pack_extra_buffer_size: int = 1000,
):
    if len(num_tokens) != len(proxy_attn_flops):
        raise ValueError("num_tokens and proxy_attn_flops must have equal lengths")
    chunks = [inds[i : i + pack_chunk_size] for i in range(0, len(inds), pack_chunk_size)]
    use_parallel = pack_workers > 1 and len(chunks) > max(2, pack_workers)
    if not use_parallel:
        pack_infos: list[dict] = []
        for chunk_indices in chunks:
            pack_infos.extend(
                get_pack_chunk_infos(
                    chunk_indices,
                    dataset_id,
                    pack_max_length,
                    pack_extra_buffer_size,
                    num_tokens,
                    proxy_attn_flops,
                )
            )
        return pack_infos

    token_shm = shared_memory.SharedMemory(create=True, size=num_tokens.nbytes)
    flop_shm = shared_memory.SharedMemory(create=True, size=proxy_attn_flops.nbytes)
    try:
        np.copyto(
            np.ndarray(num_tokens.shape, dtype=num_tokens.dtype, buffer=token_shm.buf),
            num_tokens,
        )
        np.copyto(
            np.ndarray(
                proxy_attn_flops.shape,
                dtype=proxy_attn_flops.dtype,
                buffer=flop_shm.buf,
            ),
            proxy_attn_flops,
        )
        worker = partial(
            get_pack_chunk_infos,
            dataset_id=dataset_id,
            target=pack_max_length,
            pack_extra_buffer_size=pack_extra_buffer_size,
            shm_name=token_shm.name,
            shape=num_tokens.shape,
            dtype=num_tokens.dtype,
            proxy_attn_flops_shm_name=flop_shm.name,
            proxy_attn_flops_dtype=proxy_attn_flops.dtype,
        )
        context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=pack_workers, mp_context=context) as executor:
            results = list(executor.map(worker, chunks))
        return [info for result in results for info in result]
    finally:
        token_shm.close()
        token_shm.unlink()
        flop_shm.close()
        flop_shm.unlink()


def build_expand_soft_pack_infos(
    num_tokens: np.ndarray,
    proxy_attn_flops: np.ndarray,
    *,
    seed: int,
    pack_max_length: int,
    pack_workers: int = 8,
    pack_chunk_size: int = 10_000,
    pack_extra_buffer_size: int = 100,
) -> list[dict]:
    """Build pack membership in the same deterministic order as XTuner."""

    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(len(num_tokens), generator=generator).tolist()
    pack_infos = get_pack_infos_by_expand_soft_split(
        indices,
        dataset_id=0,
        num_tokens=num_tokens,
        proxy_attn_flops=proxy_attn_flops,
        pack_max_length=pack_max_length,
        pack_workers=pack_workers,
        pack_chunk_size=pack_chunk_size,
        pack_extra_buffer_size=pack_extra_buffer_size,
    )
    flattened = [index for info in pack_infos for index in info["indices"]]
    if len(flattened) != len(num_tokens) or len(set(flattened)) != len(num_tokens):
        raise RuntimeError("soft packing did not assign every sample exactly once")
    return pack_infos
