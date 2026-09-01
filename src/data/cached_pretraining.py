from __future__ import annotations

import json
import mmap
from array import array
from pathlib import Path

import torch
from torch.utils.data import Dataset

from data.utils import IGNORE_INDEX


TOKEN_IDS_NAME = "token_ids.uint32.bin"
CHUNKS_NAME = "chunks.jsonl"
META_NAME = "meta.json"
PACKED_CACHE_FORMAT = "xtuner_softpack_v1"
LABELS_NAME = "labels.int32.bin"
ATTENTION_MASK_NAME = "attention_mask.uint8.bin"


def write_document_chunk_cache(
    *,
    output_dir: Path,
    chunk_token_ids: array,
    chunks: list[dict],
    meta: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    token_path = output_dir / TOKEN_IDS_NAME
    with token_path.open("wb") as handle:
        chunk_token_ids.tofile(handle)

    chunks_path = output_dir / CHUNKS_NAME
    with chunks_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False, sort_keys=True) + "\n")

    payload = {
        **meta,
        "cache_format": "document_chunk_v1",
        "token_ids_file": TOKEN_IDS_NAME,
        "token_ids_dtype": "uint32",
        "chunks_file": CHUNKS_NAME,
        "num_chunks": len(chunks),
        "num_tokens": len(chunk_token_ids),
    }
    (output_dir / META_NAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class CachedDocumentChunkDataset(Dataset):
    def __init__(self, cache_dir, **_unused):
        super().__init__()
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.is_dir():
            raise NotADirectoryError(f"Cached chunk directory does not exist: {self.cache_dir}")

        meta_path = self.cache_dir / META_NAME
        token_path = self.cache_dir / TOKEN_IDS_NAME
        chunks_path = self.cache_dir / CHUNKS_NAME
        for path in (meta_path, token_path, chunks_path):
            if not path.exists():
                raise FileNotFoundError(f"Missing cached chunk file: {path}")

        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.cache_format = self.meta.get("cache_format")
        if self.cache_format not in {"document_chunk_v1", PACKED_CACHE_FORMAT}:
            raise ValueError(
                f"Unsupported cache_format={self.cache_format!r} in {meta_path}"
            )

        token_itemsize = array("I").itemsize
        expected_bytes = token_path.stat().st_size
        if expected_bytes % token_itemsize != 0:
            raise ValueError(f"Token id cache has invalid byte size: {token_path}")
        # A retain cache can be tens of GiB. Loading it with array.fromfile in
        # every torchrun rank duplicates the entire file in host RAM (8 x 20
        # GiB for Intern-S2). A read-only mmap shares file-backed pages across
        # ranks and faults in only the sampled token ranges.
        self._token_handle = token_path.open("rb")
        self._token_mmap = mmap.mmap(
            self._token_handle.fileno(), 0, access=mmap.ACCESS_READ
        )
        self.chunk_token_ids = memoryview(self._token_mmap).cast("I")

        self._label_handle = None
        self._label_mmap = None
        self._attention_handle = None
        self._attention_mmap = None
        self.packed_labels = None
        self.packed_attention_mask = None
        if self.cache_format == PACKED_CACHE_FORMAT:
            label_path = self.cache_dir / self.meta.get("labels_file", LABELS_NAME)
            attention_path = self.cache_dir / self.meta.get(
                "attention_mask_file", ATTENTION_MASK_NAME
            )
            for path in (label_path, attention_path):
                if not path.is_file():
                    raise FileNotFoundError(f"Missing packed cache file: {path}")
            if label_path.stat().st_size % array("i").itemsize:
                raise ValueError(f"Packed label cache has invalid byte size: {label_path}")
            self._label_handle = label_path.open("rb")
            self._label_mmap = mmap.mmap(
                self._label_handle.fileno(), 0, access=mmap.ACCESS_READ
            )
            self.packed_labels = memoryview(self._label_mmap).cast("i")
            self._attention_handle = attention_path.open("rb")
            self._attention_mmap = mmap.mmap(
                self._attention_handle.fileno(), 0, access=mmap.ACCESS_READ
            )
            self.packed_attention_mask = memoryview(self._attention_mmap).cast("B")

        # Avoid expanding millions of JSON objects into per-rank Python dicts.
        # Compact numeric arrays preserve the exact file order and offsets.
        self._token_offsets = array("Q")
        self._token_lengths = array("I")
        self._indices = []
        with chunks_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    chunk = json.loads(line)
                    self._token_offsets.append(int(chunk["token_offset"]))
                    self._token_lengths.append(int(chunk["token_length"]))
                    self._indices.append(chunk["index"])

        if int(self.meta.get("num_chunks", -1)) != len(self._token_offsets):
            raise ValueError(
                f"Chunk count mismatch in {self.cache_dir}: "
                f"meta={self.meta.get('num_chunks')} actual={len(self._token_offsets)}"
            )
        if int(self.meta.get("num_tokens", -1)) != len(self.chunk_token_ids):
            raise ValueError(
                f"Token count mismatch in {self.cache_dir}: "
                f"meta={self.meta.get('num_tokens')} actual={len(self.chunk_token_ids)}"
            )
        if self.cache_format == PACKED_CACHE_FORMAT:
            if len(self.packed_labels) != len(self.chunk_token_ids):
                raise ValueError(
                    f"Label count mismatch in {self.cache_dir}: "
                    f"labels={len(self.packed_labels)} tokens={len(self.chunk_token_ids)}"
                )
            if len(self.packed_attention_mask) != len(self.chunk_token_ids):
                raise ValueError(
                    f"Attention-mask count mismatch in {self.cache_dir}: "
                    f"mask={len(self.packed_attention_mask)} tokens={len(self.chunk_token_ids)}"
                )

    def __len__(self):
        return len(self._token_offsets)

    def __getitem__(self, idx):
        start = int(self._token_offsets[idx])
        end = start + int(self._token_lengths[idx])
        input_ids = self.chunk_token_ids[start:end].tolist()
        if self.cache_format == PACKED_CACHE_FORMAT:
            labels = self.packed_labels[start:end].tolist()
            attention_mask = self.packed_attention_mask[start:end].tolist()
        else:
            labels = list(input_ids)
            if labels:
                labels[0] = IGNORE_INDEX
            attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "index": self._indices[idx],
        }
