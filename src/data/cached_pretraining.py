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
        if self.meta.get("cache_format") != "document_chunk_v1":
            raise ValueError(
                f"Unsupported cache_format={self.meta.get('cache_format')!r} in {meta_path}"
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

    def __len__(self):
        return len(self._token_offsets)

    def __getitem__(self, idx):
        start = int(self._token_offsets[idx])
        end = start + int(self._token_lengths[idx])
        input_ids = self.chunk_token_ids[start:end].tolist()
        labels = list(input_ids)
        if labels:
            labels[0] = IGNORE_INDEX

        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "index": self._indices[idx],
        }
