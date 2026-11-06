import json
import tempfile
import unittest
from array import array
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from data.build_xtuner_packed_cache import build_from_config
from data.cached_pretraining import CachedDocumentChunkDataset, write_document_chunk_cache
from data.collators import DataCollatorForSupervisedDataset
from data.xtuner_compat.soft_pack import build_expand_soft_pack_infos
from data.xtuner_packed_cache import write_xtuner_packed_cache
from megatron_unlearning.batching import collate_role_batch


class _TokenizerStub:
    pad_token_id = 2


class XtunerSoftPackTest(unittest.TestCase):
    def test_yaml_builder_path_runs_on_a_tiny_jsonl_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer_path = root / "tokenizer"
            backend = Tokenizer(
                WordLevel(
                    {
                        "[UNK]": 0,
                        "<pad>": 1,
                        "<eos>": 2,
                        "alpha": 3,
                        "beta": 4,
                        "gamma": 5,
                    },
                    unk_token="[UNK]",
                )
            )
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=backend,
                unk_token="[UNK]",
                pad_token="<pad>",
                eos_token="<eos>",
            )
            tokenizer.save_pretrained(tokenizer_path)
            data_path = root / "data.jsonl"
            data_path.write_text(
                "".join(
                    json.dumps({"content": text}) + "\n"
                    for text in ("alpha beta", "gamma", "alpha gamma beta")
                ),
                encoding="utf-8",
            )
            output_dir = root / "packed"
            built = build_from_config(
                {
                    "tokenizer": {"path": str(tokenizer_path), "trust_remote_code": False},
                    "dataset": {
                        "files": str(data_path),
                        "format": "json",
                        "text_key": "content",
                        "sample_ratio": 1.0,
                    },
                    "packing": {
                        "seed": 42,
                        "pack_level": "soft",
                        "pack_max_length": 8,
                        "pack_to_max_length": True,
                        "pack_chunk_size": 100,
                        "pack_workers": 1,
                        "pack_extra_buffer_size": 2,
                        "global_pack": False,
                        "add_eos_token": True,
                        "add_bos_token": False,
                    },
                    "build": {"tokenize_workers": 1, "tokenize_batch_size": 2},
                    "output": {"directory": str(output_dir), "overwrite": False},
                }
            )
            self.assertEqual(built, output_dir)
            dataset = CachedDocumentChunkDataset(output_dir)
            self.assertGreater(len(dataset), 0)
            self.assertTrue(
                all(len(dataset[index]["input_ids"]) == 8 for index in range(len(dataset)))
            )

    def test_local_soft_pack_order_is_seeded_and_assigns_every_document(self):
        lengths = np.asarray([3, 4, 2, 10, 3], dtype=np.int64)
        flops = lengths.astype(np.float64)
        first = build_expand_soft_pack_infos(
            lengths,
            flops,
            seed=42,
            pack_max_length=8,
            pack_workers=1,
            pack_chunk_size=100,
            pack_extra_buffer_size=2,
        )
        second = build_expand_soft_pack_infos(
            lengths,
            flops,
            seed=42,
            pack_max_length=8,
            pack_workers=1,
            pack_chunk_size=100,
            pack_extra_buffer_size=2,
        )
        self.assertEqual(first, second)
        assigned = [index for pack in first for index in pack["indices"]]
        self.assertEqual(sorted(assigned), list(range(len(lengths))))

    def test_parallel_pack_matches_serial_pack(self):
        lengths = np.asarray([(index % 7) + 1 for index in range(80)], dtype=np.int64)
        flops = lengths.astype(np.float64)
        common = {
            "seed": 19,
            "pack_max_length": 16,
            "pack_chunk_size": 5,
            "pack_extra_buffer_size": 2,
        }
        serial = build_expand_soft_pack_infos(lengths, flops, pack_workers=1, **common)
        parallel = build_expand_soft_pack_infos(lengths, flops, pack_workers=2, **common)
        self.assertEqual(parallel, serial)

    def test_packed_cache_round_trip_preserves_boundaries_padding_and_long_truncation(self):
        documents = [
            [10, 11, 2],
            [20, 21, 22, 2],
            [30, 2],
            [40, 41, 42, 43, 44, 45, 46, 47, 48, 2],
        ]
        lengths = np.asarray([len(document) for document in documents], dtype=np.int64)
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory) / "packed"
            write_xtuner_packed_cache(
                output_dir=cache_dir,
                document_lengths=lengths,
                get_document_tokens=lambda index: documents[index],
                pad_token_id=2,
                seed=7,
                pack_max_length=8,
                pack_chunk_size=100,
                pack_workers=1,
                pack_extra_buffer_size=2,
                show_progress=False,
            )
            dataset = CachedDocumentChunkDataset(cache_dir)
            chunks = [
                json.loads(line)
                for line in (cache_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            ]

            assigned = [index for chunk in chunks for index in chunk["document_indices"]]
            self.assertEqual(sorted(assigned), list(range(len(documents))))
            for index, chunk in enumerate(chunks):
                item = dataset[index]
                self.assertEqual(item["input_ids"].shape, torch.Size([8]))
                self.assertEqual(item["labels"].shape, torch.Size([8]))
                self.assertEqual(item["attention_mask"].shape, torch.Size([8]))
                content_length = chunk["content_length"]
                self.assertTrue(torch.all(item["attention_mask"][:content_length] == 1))
                self.assertTrue(torch.all(item["attention_mask"][content_length:] == 0))
                self.assertTrue(torch.all(item["labels"][content_length:] == -100))
                for start in chunk["document_starts"]:
                    self.assertEqual(item["labels"][start].item(), -100)

            long_pack_index = next(
                index for index, chunk in enumerate(chunks) if 3 in chunk["document_indices"]
            )
            self.assertEqual(dataset[long_pack_index]["input_ids"].tolist(), documents[3][:8])

    def test_document_chunk_v1_behavior_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory) / "legacy"
            write_document_chunk_cache(
                output_dir=cache_dir,
                chunk_token_ids=array("I", [7, 8, 9]),
                chunks=[{"index": 4, "token_offset": 0, "token_length": 3}],
                meta={},
            )
            item = CachedDocumentChunkDataset(cache_dir)[0]
            self.assertEqual(item["input_ids"].tolist(), [7, 8, 9])
            self.assertEqual(item["labels"].tolist(), [-100, 8, 9])
            self.assertEqual(item["attention_mask"].tolist(), [1, 1, 1])

    def test_collators_use_explicit_mask_when_pad_and_eos_ids_are_equal(self):
        instance = {
            "input_ids": torch.tensor([10, 2, 20, 2, 2]),
            "labels": torch.tensor([-100, 2, -100, 2, -100]),
            "attention_mask": torch.tensor([1, 1, 1, 1, 0]),
        }
        hf_batch = DataCollatorForSupervisedDataset(_TokenizerStub())([instance])
        self.assertEqual(hf_batch["attention_mask"].tolist(), [[True, True, True, True, False]])

        megatron_batch = collate_role_batch(
            [instance],
            None,
            pad_token_id=2,
            sequence_length=5,
        )
        self.assertEqual(megatron_batch["loss_mask"].tolist(), [[True, False, True, False, False]])
        self.assertFalse(megatron_batch["attention_mask"][0, 0, 3, 1].item())
        self.assertTrue(megatron_batch["attention_mask"][0, 0, 3, 4].item())


if __name__ == "__main__":
    unittest.main()
