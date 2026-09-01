"""YAML-driven builder for deterministic, fully materialized XTuner soft packs."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import multiprocessing
import os
import queue
import shutil
import tempfile
from array import array
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import yaml
from tqdm.auto import tqdm

from data.build_document_chunk_cache import (
    make_jsonl_file_specs,
    make_parquet_file_specs,
    partition_specs,
)
from data.xtuner_compat.pretrain import extract_pretrain_text, prepare_pretrain_text
from data.xtuner_packed_cache import SUMMARY_NAME, write_xtuner_packed_cache

REPO_DIR = Path(__file__).resolve().parents[2]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Packed-cache YAML config.")
    parser.add_argument("--overwrite", action="store_true", help="Override output.overwrite.")
    parser.add_argument("--max-documents", type=int, default=None, help="Small-run override.")
    parser.add_argument(
        "--tokenize-workers",
        type=int,
        default=None,
        help="Override build.tokenize_workers for the current machine.",
    )
    return parser


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def load_build_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"cache build config must be a mapping: {path}")
    payload = _expand(payload)
    for section in ("tokenizer", "dataset", "packing", "output"):
        if not isinstance(payload.get(section), dict):
            raise TypeError(f"config must define a {section!r} mapping")
    return payload


def _resolve_data_files(value: str | list[str]) -> list[str]:
    patterns = [value] if isinstance(value, str) else value
    resolved: list[str] = []
    for pattern in patterns:
        path = Path(pattern)
        if path.is_dir():
            matches = [
                str(child)
                for child in sorted(path.iterdir())
                if child.is_file() and child.suffix.lower() in {".jsonl", ".json", ".parquet"}
            ]
        else:
            matches = sorted(glob.glob(pattern))
        resolved.extend(matches)
    resolved = list(dict.fromkeys(resolved))
    if not resolved:
        raise FileNotFoundError(f"no files matched dataset.files={value!r}")
    return resolved


def _tokenize_batch(tokenizer, texts: list[str]) -> list[list[int]]:
    try:
        return tokenizer(texts, add_special_tokens=False, verbose=False)["input_ids"]
    except TypeError:
        return [tokenizer.encode(text, add_special_tokens=False) for text in texts]


def _prepared_texts(records, *, tokenizer, text_key, add_eos_token, add_bos_token):
    return [
        prepare_pretrain_text(
            extract_pretrain_text(dict(record), text_key),
            tokenizer,
            add_eos_token=add_eos_token,
            add_bos_token=add_bos_token,
        )
        for record in records
    ]


def _write_documents(token_handle, length_handle, tokenized: list[list[int]]) -> tuple[int, int]:
    num_documents = 0
    num_tokens = 0
    for token_ids in tokenized:
        token_array = array("I", (int(token_id) for token_id in token_ids))
        token_array.tofile(token_handle)
        array("Q", [len(token_array)]).tofile(length_handle)
        num_documents += 1
        num_tokens += len(token_array)
    return num_documents, num_tokens


def _process_document_partition(worker_id, partition, settings, part_dir, progress_queue=None):
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        settings["model_path"],
        trust_remote_code=settings["trust_remote_code"],
    )
    part_dir = Path(part_dir)
    token_path = part_dir / f"part_{worker_id:05d}.documents.uint32.bin"
    length_path = part_dir / f"part_{worker_id:05d}.lengths.uint64.bin"
    batch_size = settings["tokenize_batch_size"]
    num_documents = 0
    num_tokens = 0

    def process_records(records, token_handle, length_handle):
        texts = _prepared_texts(
            records,
            tokenizer=tokenizer,
            text_key=settings["text_key"],
            add_eos_token=settings["add_eos_token"],
            add_bos_token=settings["add_bos_token"],
        )
        return _write_documents(token_handle, length_handle, _tokenize_batch(tokenizer, texts))

    with token_path.open("wb") as token_handle, length_path.open("wb") as length_handle:
        for spec in partition:
            if spec.get("format") == "json":
                with open(spec["path"], "rb") as jsonl_handle:
                    jsonl_handle.seek(spec["start_offset"])
                    rows_left = int(spec["num_rows"])
                    while rows_left > 0:
                        lines = [jsonl_handle.readline() for _ in range(min(batch_size, rows_left))]
                        records = [json.loads(line) for line in lines if line]
                        if not records:
                            raise EOFError(f"unexpected end of JSONL file: {spec['path']}")
                        docs, tokens = process_records(records, token_handle, length_handle)
                        num_documents += docs
                        num_tokens += tokens
                        rows_left -= docs
                        if progress_queue is not None:
                            progress_queue.put(docs)
                continue

            parquet_file = pq.ParquetFile(spec["path"])
            rows_left = int(spec["num_rows"])
            for row_group_index in range(parquet_file.metadata.num_row_groups):
                if rows_left <= 0:
                    break
                row_group_rows = parquet_file.metadata.row_group(row_group_index).num_rows
                rows_to_read = min(rows_left, row_group_rows)
                columns = [settings["text_key"]] if settings["text_key"] else None
                table = parquet_file.read_row_group(row_group_index, columns=columns)
                records = table.to_pylist()[:rows_to_read]
                for start in range(0, len(records), batch_size):
                    docs, tokens = process_records(
                        records[start : start + batch_size],
                        token_handle,
                        length_handle,
                    )
                    num_documents += docs
                    num_tokens += tokens
                    if progress_queue is not None:
                        progress_queue.put(docs)
                rows_left -= rows_to_read

    return {
        "worker_id": worker_id,
        "token_path": str(token_path),
        "length_path": str(length_path),
        "num_documents": num_documents,
        "num_tokens": num_tokens,
    }


def _merge_document_parts(staging_dir: Path, results: list[dict]) -> tuple[Path, np.ndarray]:
    token_path = staging_dir / "documents.uint32.bin"
    length_arrays = []
    with token_path.open("wb") as token_out:
        for result in sorted(results, key=lambda item: item["worker_id"]):
            with open(result["token_path"], "rb") as token_in:
                shutil.copyfileobj(token_in, token_out, length=16 * 1024 * 1024)
            length_arrays.append(np.fromfile(result["length_path"], dtype=np.uint64))
    lengths = np.concatenate(length_arrays).astype(np.int64, copy=False)
    return token_path, lengths


def _tokenizer_sha256(tokenizer) -> str:
    digest = hashlib.sha256()
    with tempfile.TemporaryDirectory(prefix="ou_tokenizer_hash_") as directory:
        tokenizer.save_pretrained(directory)
        for path in sorted(Path(directory).iterdir()):
            if not path.is_file():
                continue
            digest.update(path.name.encode("utf-8"))
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
    return digest.hexdigest()


def _assert_safe_generated_path(path: Path) -> None:
    resolved = path.resolve()
    if resolved in {Path("/"), REPO_DIR, REPO_DIR.parent, Path.home().resolve()}:
        raise ValueError(f"refusing destructive operation on unsafe path: {resolved}")
    if len(resolved.parts) < 4:
        raise ValueError(f"refusing destructive operation on broad path: {resolved}")


def _remove_generated_path(path: Path) -> None:
    _assert_safe_generated_path(path)
    shutil.rmtree(path)


def _make_partitions(files, dataset_format, workers, max_documents):
    if dataset_format == "json":
        specs = make_jsonl_file_specs(files, workers, max_documents)
        return partition_specs(specs, workers)
    if dataset_format == "parquet":
        specs = make_parquet_file_specs(files, max_documents)
        for spec in specs:
            spec["format"] = "parquet"
        return partition_specs(specs, workers)
    raise ValueError("dataset.format must be 'json' or 'parquet'")


class _LocalProgressQueue:
    def __init__(self, progress_bar):
        self.progress_bar = progress_bar

    def put(self, count):
        self.progress_bar.update(int(count))


def _drain_progress_queue(progress_queue, progress_bar) -> None:
    while True:
        try:
            count = progress_queue.get_nowait()
        except queue.Empty:
            return
        progress_bar.update(int(count))


def build_from_config(config: dict[str, Any], *, overwrite=False, max_documents=None) -> Path:
    import transformers

    tokenizer_cfg = config["tokenizer"]
    dataset_cfg = config["dataset"]
    packing_cfg = config["packing"]
    output_cfg = config["output"]
    build_cfg = config.get("build") or {}

    model_path = str(tokenizer_cfg["path"])
    trust_remote_code = bool(tokenizer_cfg.get("trust_remote_code", True))
    data_files = _resolve_data_files(dataset_cfg["files"])
    dataset_format = str(dataset_cfg.get("format", "json")).lower()
    text_key = dataset_cfg.get("text_key")
    if dataset_format == "parquet" and not text_key:
        raise ValueError("dataset.text_key is required for parquet input")
    if str(packing_cfg.get("pack_level", "soft")).lower() != "soft":
        raise ValueError("the xtuner_softpack_v1 builder requires packing.pack_level=soft")
    if not bool(packing_cfg.get("pack_to_max_length", True)):
        raise ValueError("xtuner_softpack_v1 requires packing.pack_to_max_length=true")
    if bool(packing_cfg.get("global_pack", False)):
        raise ValueError("one-dataset cache builds require packing.global_pack=false")

    output_dir = Path(output_cfg["directory"]).resolve()
    workspace = output_dir.with_name(f"{output_dir.name}.build-tmp")
    allow_overwrite = bool(overwrite or output_cfg.get("overwrite", False))
    if output_dir.exists():
        if not allow_overwrite:
            raise FileExistsError(f"cache already exists: {output_dir}")
        _remove_generated_path(output_dir)
    if workspace.exists():
        _remove_generated_path(workspace)
    part_dir = workspace / "parts"
    part_dir.mkdir(parents=True)

    tokenize_workers = max(1, int(build_cfg.get("tokenize_workers", 8)))
    tokenize_batch_size = max(1, int(build_cfg.get("tokenize_batch_size", 1024)))
    document_limit = max_documents
    if document_limit is None:
        document_limit = dataset_cfg.get("max_documents")
    document_limit = int(document_limit) if document_limit is not None else None
    if document_limit is not None and document_limit <= 0:
        raise ValueError("dataset.max_documents must be positive or null")
    partitions = _make_partitions(
        data_files,
        dataset_format,
        tokenize_workers,
        document_limit,
    )
    if not partitions:
        raise ValueError("input dataset produced no partitions")
    total_documents = sum(int(spec["num_rows"]) for partition in partitions for spec in partition)
    show_progress = bool(build_cfg.get("show_progress", True))

    settings = {
        "model_path": model_path,
        "trust_remote_code": trust_remote_code,
        "text_key": str(text_key) if text_key is not None else None,
        "add_eos_token": bool(packing_cfg.get("add_eos_token", True)),
        "add_bos_token": bool(packing_cfg.get("add_bos_token", False)),
        "tokenize_batch_size": tokenize_batch_size,
    }
    results = []
    try:
        with tqdm(
            total=total_documents,
            desc="Tokenizing CPT documents",
            unit="doc",
            disable=not show_progress,
        ) as tokenize_progress:
            if len(partitions) == 1:
                local_progress = _LocalProgressQueue(tokenize_progress)
                results.append(
                    _process_document_partition(
                        0,
                        partitions[0],
                        settings,
                        part_dir,
                        local_progress,
                    )
                )
            else:
                with multiprocessing.Manager() as manager:
                    progress_queue = manager.Queue()
                    with ProcessPoolExecutor(max_workers=len(partitions)) as executor:
                        pending = {
                            executor.submit(
                                _process_document_partition,
                                worker_id,
                                partition,
                                settings,
                                part_dir,
                                progress_queue,
                            )
                            for worker_id, partition in enumerate(partitions)
                        }
                        while pending:
                            completed, pending = wait(
                                pending,
                                timeout=0.2,
                                return_when=FIRST_COMPLETED,
                            )
                            _drain_progress_queue(progress_queue, tokenize_progress)
                            for future in completed:
                                results.append(future.result())
                        _drain_progress_queue(progress_queue, tokenize_progress)

        staging_dir = workspace / "staging"
        staging_dir.mkdir()
        document_token_path, document_lengths = _merge_document_parts(staging_dir, results)
        expected_tokens = int(document_lengths.sum())
        actual_tokens = document_token_path.stat().st_size // np.dtype(np.uint32).itemsize
        if expected_tokens != actual_tokens:
            raise ValueError(
                f"staged token count mismatch: lengths={expected_tokens} file={actual_tokens}"
            )

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        if tokenizer.pad_token_id is None:
            raise ValueError("tokenizer has no pad_token_id, eos_token_id, or unk_token_id")

        document_tokens = np.memmap(document_token_path, dtype=np.uint32, mode="r")
        offsets = np.empty(len(document_lengths) + 1, dtype=np.uint64)
        offsets[0] = 0
        np.cumsum(document_lengths, dtype=np.uint64, out=offsets[1:])

        def get_document_tokens(index: int):
            return document_tokens[int(offsets[index]) : int(offsets[index + 1])]

        cache_dir = workspace / "cache"
        seed = int(packing_cfg.get("seed", 42))
        source_stats = [
            {
                "path": str(Path(path).resolve()),
                "size": Path(path).stat().st_size,
                "mtime_ns": Path(path).stat().st_mtime_ns,
            }
            for path in data_files
        ]
        write_xtuner_packed_cache(
            output_dir=cache_dir,
            document_lengths=document_lengths,
            get_document_tokens=get_document_tokens,
            pad_token_id=int(tokenizer.pad_token_id),
            seed=seed,
            pack_max_length=int(packing_cfg.get("pack_max_length", 4096)),
            pack_chunk_size=int(packing_cfg.get("pack_chunk_size", 10_000)),
            pack_workers=int(packing_cfg.get("pack_workers", 4)),
            pack_extra_buffer_size=int(packing_cfg.get("pack_extra_buffer_size", 100)),
            sample_ratio=float(dataset_cfg.get("sample_ratio", 1.0)),
            filter_empty=bool(dataset_cfg.get("filter_empty", True)),
            metadata={
                "builder": "data.build_xtuner_packed_cache",
                "model_path": model_path,
                "tokenizer_class": tokenizer.__class__.__name__,
                "tokenizer_sha256": _tokenizer_sha256(tokenizer),
                "trust_remote_code": trust_remote_code,
                "data_files": source_stats,
                "dataset_format": dataset_format,
                "text_key": text_key,
                "max_documents": document_limit,
                "add_eos_token": settings["add_eos_token"],
                "add_bos_token": settings["add_bos_token"],
                "pack_to_max_length": True,
                "source_order_is_training_order": True,
                "runtime_shuffle_required": False,
            },
            show_progress=show_progress,
        )
        document_tokens._mmap.close()
        shutil.rmtree(part_dir)
        shutil.rmtree(staging_dir)
        cache_dir.rename(output_dir)
        workspace.rmdir()

        summary_path = output_dir / SUMMARY_NAME
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["output_dir"] = str(output_dir)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output_dir
    except Exception:
        if workspace.exists():
            _remove_generated_path(workspace)
        raise


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_build_config(args.config.resolve())
    if args.tokenize_workers is not None:
        if args.tokenize_workers <= 0:
            raise ValueError("--tokenize-workers must be positive")
        config.setdefault("build", {})["tokenize_workers"] = args.tokenize_workers
    output_dir = build_from_config(
        config,
        overwrite=args.overwrite,
        max_documents=args.max_documents,
    )
    print(f"Wrote XTuner soft-pack cache to {output_dir}")


if __name__ == "__main__":
    main()
