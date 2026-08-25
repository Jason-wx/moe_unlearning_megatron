from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from array import array
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq
from tqdm.auto import tqdm

from data.cached_pretraining import CHUNKS_NAME, META_NAME, TOKEN_IDS_NAME

REPO_DIR = Path(__file__).resolve().parents[2]
DATA_CACHE_ROOT = (REPO_DIR / "data_cache").resolve()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a document chunk cache in parallel over parquet or JSONL files."
    )
    parser.add_argument("--model-path", required=True, help="Tokenizer/model path.")
    parser.add_argument("--data-files", required=True, help="Input file or glob.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output cache directory.")
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--stride", default=None)
    parser.add_argument("--min-length", type=int, default=1)
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument("--no-add-special-tokens", action="store_true")
    parser.add_argument("--tokenize-batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--dataset-path",
        choices=("parquet", "json"),
        default="parquet",
        help="Input format. JSON input must contain one object per line (JSONL).",
    )
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-documents", type=int, default=None)
    return parser


def normalize_stride(value):
    if value in (None, "", "none", "None", "null", "Null"):
        return None
    return int(value)


def assert_safe_cache_path(path: Path) -> None:
    path = path.resolve()
    try:
        path.relative_to(DATA_CACHE_ROOT)
    except ValueError as exc:
        raise ValueError(f"Refusing to remove path outside data_cache: {path}") from exc
    if path == DATA_CACHE_ROOT:
        raise ValueError(f"Refusing to remove data_cache root: {path}")
    if path in (REPO_DIR, REPO_DIR.parent, Path("/")):
        raise ValueError(f"Refusing to remove unsafe path: {path}")


def safe_rmtree(path: Path) -> None:
    assert_safe_cache_path(path)
    shutil.rmtree(path)


def resolve_data_files(data_files):
    resolved = sorted(glob.glob(data_files))
    if not resolved:
        raise FileNotFoundError(f"No files matched data-files glob: {data_files}")
    return resolved


def make_parquet_file_specs(paths, max_documents=None):
    specs = []
    next_index = 0
    remaining = max_documents
    for path in paths:
        rows = pq.ParquetFile(path).metadata.num_rows
        if remaining is not None:
            if remaining <= 0:
                break
            rows = min(rows, remaining)
            remaining -= rows
        specs.append({"path": str(path), "start_index": next_index, "num_rows": rows})
        next_index += rows
    return specs


def make_jsonl_file_specs(paths, workers, max_documents=None):
    """Split JSONL inputs at line boundaries into approximately equal byte ranges."""
    specs = []
    next_index = 0
    remaining = max_documents
    target_parts_per_file = max(1, workers // max(1, len(paths)))

    for path in paths:
        if remaining is not None and remaining <= 0:
            break
        path = Path(path)
        file_size = path.stat().st_size
        target_bytes = max(1, (file_size + target_parts_per_file - 1) // target_parts_per_file)
        segment_start_offset = 0
        segment_start_index = next_index
        segment_rows = 0

        with path.open("rb") as handle:
            while True:
                if remaining is not None and remaining <= 0:
                    break
                line = handle.readline()
                if not line:
                    break
                segment_rows += 1
                next_index += 1
                if remaining is not None:
                    remaining -= 1
                position = handle.tell()
                reached_boundary = position - segment_start_offset >= target_bytes
                reached_limit = remaining is not None and remaining <= 0
                if reached_boundary or reached_limit:
                    specs.append(
                        {
                            "path": str(path),
                            "start_index": segment_start_index,
                            "num_rows": segment_rows,
                            "start_offset": segment_start_offset,
                            "end_offset": position,
                            "format": "json",
                        }
                    )
                    segment_start_offset = position
                    segment_start_index = next_index
                    segment_rows = 0

        if segment_rows:
            specs.append(
                {
                    "path": str(path),
                    "start_index": segment_start_index,
                    "num_rows": segment_rows,
                    "start_offset": segment_start_offset,
                    "end_offset": file_size,
                    "format": "json",
                }
            )
    return specs


def partition_specs(specs, workers):
    workers = max(1, min(workers, len(specs)))
    total_rows = sum(spec["num_rows"] for spec in specs)
    target_rows = max(1, (total_rows + workers - 1) // workers)
    partitions = []
    current = []
    current_rows = 0
    for spec in specs:
        if current and current_rows >= target_rows and len(partitions) < workers - 1:
            partitions.append(current)
            current = []
            current_rows = 0
        current.append(spec)
        current_rows += spec["num_rows"]
    if current:
        partitions.append(current)
    return partitions


def content_length(tokenizer, max_length, add_special_tokens):
    if not add_special_tokens:
        return max_length
    if hasattr(tokenizer, "build_inputs_with_special_tokens"):
        num_special_tokens = len(tokenizer.build_inputs_with_special_tokens([]))
    else:
        num_special_tokens = len(tokenizer("", add_special_tokens=True)["input_ids"])
    return max(1, max_length - num_special_tokens)


def with_special_tokens(tokenizer, token_ids, add_special_tokens):
    if not add_special_tokens:
        return token_ids
    if hasattr(tokenizer, "build_inputs_with_special_tokens"):
        return tokenizer.build_inputs_with_special_tokens(token_ids)
    return token_ids


def tokenize_documents(tokenizer, texts):
    try:
        return tokenizer(
            [str(text) for text in texts],
            add_special_tokens=False,
            verbose=False,
        )["input_ids"]
    except TypeError:
        return [tokenizer(str(text), add_special_tokens=False)["input_ids"] for text in texts]


def write_token_ids(handle, token_ids):
    token_array = array("I")
    token_array.extend(int(token_id) for token_id in token_ids)
    token_array.tofile(handle)
    return len(token_array)


def process_partition(worker_id, partition, args_dict, part_dir):
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args_dict["model_path"],
        trust_remote_code=args_dict["trust_remote_code"],
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_length = args_dict["max_length"]
    stride = args_dict["stride"]
    min_length = args_dict["min_length"]
    drop_last = args_dict["drop_last"]
    add_special_tokens = args_dict["add_special_tokens"]
    text_key = args_dict["text_key"]
    batch_size = args_dict["tokenize_batch_size"]
    chunk_content_length = content_length(tokenizer, max_length, add_special_tokens)

    part_dir = Path(part_dir)
    token_path = part_dir / f"part_{worker_id:05d}.{TOKEN_IDS_NAME}"
    chunks_path = part_dir / f"part_{worker_id:05d}.{CHUNKS_NAME}"
    num_chunks = 0
    num_tokens = 0
    num_documents = 0

    with (
        token_path.open("wb") as token_handle,
        chunks_path.open("w", encoding="utf-8") as chunks_handle,
    ):
        for spec in partition:
            if spec.get("format") == "json":
                with open(spec["path"], "rb") as jsonl_handle:
                    jsonl_handle.seek(spec["start_offset"])
                    doc_index = spec["start_index"]
                    rows_left = spec["num_rows"]
                    while rows_left > 0:
                        lines = [jsonl_handle.readline() for _ in range(min(batch_size, rows_left))]
                        records = [json.loads(line) for line in lines if line]
                        texts = []
                        for record in records:
                            if text_key not in record:
                                raise KeyError(
                                    f"text key {text_key!r} not found in JSONL record "
                                    f"at document index {doc_index + len(texts)}"
                                )
                            texts.append(record[text_key])
                        tokenized_batch = tokenize_documents(tokenizer, texts)
                        for batch_offset, token_ids in enumerate(tokenized_batch):
                            global_doc_index = doc_index + batch_offset
                            for start in range(0, len(token_ids), stride):
                                chunk_length = min(chunk_content_length, len(token_ids) - start)
                                if chunk_length < min_length:
                                    continue
                                if drop_last and chunk_length < chunk_content_length:
                                    continue
                                input_ids = with_special_tokens(
                                    tokenizer,
                                    token_ids[start : start + chunk_length],
                                    add_special_tokens,
                                )
                                token_offset = num_tokens
                                written = write_token_ids(token_handle, input_ids)
                                chunk = {
                                    "index": global_doc_index,
                                    "chunk_index": num_chunks,
                                    "token_offset": token_offset,
                                    "token_length": written,
                                }
                                chunks_handle.write(
                                    json.dumps(chunk, ensure_ascii=False, sort_keys=True) + "\n"
                                )
                                num_chunks += 1
                                num_tokens += written
                                if start + chunk_content_length >= len(token_ids):
                                    break
                        processed = len(records)
                        if processed == 0:
                            raise EOFError(
                                f"Unexpected end of JSONL file while reading {spec['path']}"
                            )
                        doc_index += processed
                        rows_left -= processed
                        num_documents += processed
                continue

            pf = pq.ParquetFile(spec["path"])
            rows_left = spec["num_rows"]
            doc_index = spec["start_index"]
            for row_group_index in range(pf.metadata.num_row_groups):
                if rows_left <= 0:
                    break
                row_group_rows = pf.metadata.row_group(row_group_index).num_rows
                rows_to_read = min(rows_left, row_group_rows)
                table = pf.read_row_group(row_group_index, columns=[text_key])
                texts = table.column(text_key).to_pylist()[:rows_to_read]
                for batch_start in range(0, len(texts), batch_size):
                    text_batch = texts[batch_start : batch_start + batch_size]
                    tokenized_batch = tokenize_documents(tokenizer, text_batch)
                    for batch_offset, token_ids in enumerate(tokenized_batch):
                        global_doc_index = doc_index + batch_start + batch_offset
                        for start in range(0, len(token_ids), stride):
                            chunk_length = min(chunk_content_length, len(token_ids) - start)
                            if chunk_length < min_length:
                                continue
                            if drop_last and chunk_length < chunk_content_length:
                                continue
                            input_ids = with_special_tokens(
                                tokenizer,
                                token_ids[start : start + chunk_length],
                                add_special_tokens,
                            )
                            token_offset = num_tokens
                            written = write_token_ids(token_handle, input_ids)
                            chunk = {
                                "index": global_doc_index,
                                "chunk_index": num_chunks,
                                "token_offset": token_offset,
                                "token_length": written,
                            }
                            chunks_handle.write(
                                json.dumps(chunk, ensure_ascii=False, sort_keys=True) + "\n"
                            )
                            num_chunks += 1
                            num_tokens += written
                            if start + chunk_content_length >= len(token_ids):
                                break
                rows_left -= rows_to_read
                doc_index += rows_to_read
                num_documents += rows_to_read

    return {
        "worker_id": worker_id,
        "token_path": str(token_path),
        "chunks_path": str(chunks_path),
        "num_chunks": num_chunks,
        "num_tokens": num_tokens,
        "num_documents": num_documents,
    }


def merge_parts(output_dir, part_results):
    token_offset = 0
    chunk_index = 0
    token_path = output_dir / TOKEN_IDS_NAME
    chunks_path = output_dir / CHUNKS_NAME

    with token_path.open("wb") as token_out, chunks_path.open("w", encoding="utf-8") as chunks_out:
        for result in sorted(part_results, key=lambda item: item["worker_id"]):
            with open(result["token_path"], "rb") as token_in:
                shutil.copyfileobj(token_in, token_out, length=1024 * 1024 * 16)
            with open(result["chunks_path"], "r", encoding="utf-8") as chunks_in:
                for line in chunks_in:
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    chunk["token_offset"] = int(chunk["token_offset"]) + token_offset
                    chunk["chunk_index"] = chunk_index
                    chunks_out.write(json.dumps(chunk, ensure_ascii=False, sort_keys=True) + "\n")
                    chunk_index += 1
            token_offset += int(result["num_tokens"])

    return chunk_index, token_offset


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = args.output_dir.resolve()
    tmp_dir = output_dir.with_name(output_dir.name + ".tmp")
    part_dir = tmp_dir / "parts"

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Cache already exists: {output_dir}. Pass --overwrite to replace it."
            )
        safe_rmtree(output_dir)
    if tmp_dir.exists():
        safe_rmtree(tmp_dir)
    part_dir.mkdir(parents=True)

    max_length = int(args.max_length)
    stride = normalize_stride(args.stride) or max_length
    min_length = int(args.min_length)
    tokenize_batch_size = int(args.tokenize_batch_size)
    workers = int(args.workers)
    if (
        max_length <= 0
        or stride <= 0
        or min_length <= 0
        or tokenize_batch_size <= 0
        or workers <= 0
    ):
        raise ValueError(
            "max_length, stride, min_length, tokenize_batch_size, and workers must be positive"
        )

    paths = resolve_data_files(args.data_files)
    if args.dataset_path == "json":
        specs = make_jsonl_file_specs(paths, workers, args.max_documents)
        partitions = [[spec] for spec in specs]
    else:
        specs = make_parquet_file_specs(paths, args.max_documents)
        for spec in specs:
            spec["format"] = "parquet"
        partitions = partition_specs(specs, workers)
    args_dict = {
        "model_path": args.model_path,
        "trust_remote_code": args.trust_remote_code,
        "text_key": args.text_key,
        "max_length": max_length,
        "stride": stride,
        "min_length": min_length,
        "drop_last": bool(args.drop_last),
        "add_special_tokens": not args.no_add_special_tokens,
        "tokenize_batch_size": tokenize_batch_size,
    }

    print(
        f"Processing {sum(spec['num_rows'] for spec in specs)} documents from "
        f"{len(specs)} {args.dataset_path} input partitions with "
        f"{len(partitions)} workers"
    )
    part_results = []
    with ProcessPoolExecutor(max_workers=len(partitions)) as executor:
        futures = [
            executor.submit(process_partition, worker_id, partition, args_dict, part_dir)
            for worker_id, partition in enumerate(partitions)
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Finished workers"):
            result = future.result()
            part_results.append(result)
            print(
                f"worker={result['worker_id']} docs={result['num_documents']} "
                f"chunks={result['num_chunks']} tokens={result['num_tokens']}"
            )

    tmp_dir.mkdir(parents=True, exist_ok=True)
    num_chunks, num_tokens = merge_parts(tmp_dir, part_results)

    meta = {
        "model_path": str(args.model_path),
        "tokenizer_class": "AutoTokenizer",
        "data_files": args.data_files,
        "dataset_path": args.dataset_path,
        "split": "train",
        "text_key": args.text_key,
        "max_length": max_length,
        "stride": stride,
        "min_length": min_length,
        "drop_last": bool(args.drop_last),
        "add_special_tokens": not args.no_add_special_tokens,
        "tokenize_batch_size": tokenize_batch_size,
        "workers": len(partitions),
        "max_documents": args.max_documents,
        "cwd": os.getcwd(),
        "cache_format": "document_chunk_v1",
        "token_ids_file": TOKEN_IDS_NAME,
        "token_ids_dtype": "uint32",
        "chunks_file": CHUNKS_NAME,
        "num_chunks": num_chunks,
        "num_tokens": num_tokens,
    }
    (tmp_dir / META_NAME).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (tmp_dir / "summary.json").write_text(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "num_chunks": num_chunks,
                "num_tokens": num_tokens,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    safe_rmtree(part_dir)
    tmp_dir.rename(output_dir)

    print(f"Wrote document chunk cache to {output_dir}")
    print(f"num_chunks={num_chunks} num_tokens={num_tokens}")


if __name__ == "__main__":
    main()
