import json
import tempfile
import unittest
from array import array
from pathlib import Path
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

import data.build_document_chunk_cache as builder

BOS_ID = 2
WORDS = [f"w{index}" for index in range(10)]
VOCAB = {"[UNK]": 0, "<pad>": 1, "<bos>": BOS_ID, "<eos>": 3}
VOCAB.update({word: 4 + index for index, word in enumerate(WORDS)})
# Document lengths straddle chunk boundaries: 7 and 9 tokens split into several
# chunks, 3 and 1 fit in one, the empty document yields no chunk.
DOCUMENTS = [WORDS[:7], WORDS[3:6], WORDS[9:], [], WORDS[:9]]


def _save_bos_tokenizer(path: Path) -> list[list[int]]:
    """Save a tokenizer that, like Gemma4, only adds its BOS on encode."""
    backend = Tokenizer(WordLevel(VOCAB, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="<bos> $A", special_tokens=[("<bos>", BOS_ID)]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    tokenizer.save_pretrained(path)
    return [[VOCAB[word] for word in document] for document in DOCUMENTS]


def _read_cache(cache_dir: Path) -> tuple[dict, list[dict], list[list[int]]]:
    meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
    chunks = [
        json.loads(line)
        for line in (cache_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    tokens = array("I")
    with (cache_dir / "token_ids.uint32.bin").open("rb") as handle:
        tokens.frombytes(handle.read())
    chunk_tokens = [
        tokens[chunk["token_offset"] : chunk["token_offset"] + chunk["token_length"]].tolist()
        for chunk in chunks
    ]
    return meta, chunks, chunk_tokens


class DocumentChunkCacheTest(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        self.tokenizer_path = self.root / "tokenizer"
        self.document_tokens = _save_bos_tokenizer(self.tokenizer_path)
        self.data_path = self.root / "data.parquet"
        pq.write_table(
            pa.table({"text": [" ".join(document) for document in DOCUMENTS]}), self.data_path
        )
        # safe_rmtree only removes paths under DATA_CACHE_ROOT; scope it to this test.
        patcher = mock.patch.object(builder, "DATA_CACHE_ROOT", self.root.resolve())
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._directory.cleanup)

    def _build(self, name: str, *extra: str) -> Path:
        output_dir = self.root / name
        builder.main(
            [
                "--model-path",
                str(self.tokenizer_path),
                "--data-files",
                str(self.data_path),
                "--output-dir",
                str(output_dir),
                "--max-length",
                "4",
                "--workers",
                "1",
                *extra,
            ]
        )
        return output_dir

    def test_tokenizer_fixture_only_adds_bos_on_encode(self):
        tokenizer = PreTrainedTokenizerFast.from_pretrained(self.tokenizer_path)
        self.assertEqual(tokenizer("w1", add_special_tokens=True)["input_ids"], [BOS_ID, 5])
        self.assertEqual(builder.special_token_affixes(tokenizer), ([BOS_ID], []))

    def test_every_chunk_starts_with_bos_and_no_token_is_dropped(self):
        meta, chunks, chunk_tokens = _read_cache(self._build("bos"))

        # BOS takes one of the 4 slots, so the default stride is the 3 content tokens.
        self.assertEqual(meta["stride"], 3)
        for tokens in chunk_tokens:
            self.assertEqual(tokens[0], BOS_ID)
            self.assertLessEqual(len(tokens), 4)
        for index, document in enumerate(self.document_tokens):
            content = [
                token
                for chunk, tokens in zip(chunks, chunk_tokens)
                if chunk["index"] == index
                for token in tokens[1:]
            ]
            self.assertEqual(content, document, f"document {index}")

    def test_chunks_without_special_tokens_are_contiguous(self):
        meta, chunks, chunk_tokens = _read_cache(self._build("plain", "--no-add-special-tokens"))

        self.assertEqual(meta["stride"], 4)
        self.assertNotIn(BOS_ID, [token for tokens in chunk_tokens for token in tokens])
        for index, document in enumerate(self.document_tokens):
            content = [
                token
                for chunk, tokens in zip(chunks, chunk_tokens)
                if chunk["index"] == index
                for token in tokens
            ]
            self.assertEqual(content, document, f"document {index}")

    def test_stride_that_would_drop_tokens_is_rejected_before_overwrite(self):
        existing = self.root / "existing"
        existing.mkdir()
        (existing / "meta.json").write_text("{}", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "would be dropped"):
            self._build("existing", "--stride", "4", "--overwrite")
        self.assertTrue((existing / "meta.json").exists())


class SpecialTokenAffixTest(unittest.TestCase):
    class _PrefixSuffixTokenizer:
        """Encode-only tokenizer (no build_inputs_with_special_tokens) adding BOS and EOS."""

        def __call__(self, text, add_special_tokens=True, **_unused):
            ids = [10 + len(word) for word in text.split()]
            return {"input_ids": [BOS_ID, *ids, 3] if add_special_tokens else ids}

    def test_prefix_and_suffix_wrap_chunk_and_reserve_slots(self):
        tokenizer = self._PrefixSuffixTokenizer()
        affixes = builder.special_token_affixes(tokenizer)

        self.assertEqual(affixes, ([BOS_ID], [3]))
        self.assertEqual(builder.content_length(tokenizer, 10, True, affixes), 8)
        self.assertEqual(
            builder.with_special_tokens(tokenizer, [7, 8], True, affixes), [BOS_ID, 7, 8, 3]
        )
        self.assertEqual(builder.with_special_tokens(tokenizer, [7, 8], False, affixes), [7, 8])


if __name__ == "__main__":
    unittest.main()
