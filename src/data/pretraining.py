from array import array

import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from data.utils import (
    IGNORE_INDEX,
    load_hf_dataset,
    add_dataset_index,
    preprocess_pretraining_instance,
)


class CompletionDataset(Dataset):
    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        prefix_key="prompt",
        text_key="text",
        max_length=2048,
        predict_with_generate=False,
        insert_space=False,
    ):
        super(CompletionDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = load_hf_dataset(**hf_args)
        self.data = add_dataset_index(self.data)
        # if either key does not exist in dataset, it is taken as ""
        self.prefix_key = prefix_key
        self.text_key = text_key
        self.predict_with_generate = predict_with_generate
        self.insert_space = insert_space

    def __len__(self):
        return len(self.data)

    def _process_sample(self, prefix, text_content, index=-1):
        tokenized_data = preprocess_pretraining_instance(
            self.tokenizer,
            prefix,
            text_content,
            self.max_length,
            self.predict_with_generate,
            self.insert_space,
        )
        item_dct = {
            "input_ids": tokenized_data["input_ids"],
            "labels": tokenized_data["labels"],
            "attention_mask": tokenized_data["attention_mask"],
        }
        if index != -1:
            item_dct["index"] = index
        return item_dct

    def __getitem__(self, idx):
        pref = self.data[idx].get(self.prefix_key, "")
        text_content = self.data[idx].get(self.text_key, "")
        index = self.data[idx]["index"]
        item = self._process_sample(pref, text_content, index)
        return item


class TruncatedTextDataset(Dataset):
    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        text_key="text",
        max_length=512,
    ):
        super(TruncatedTextDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.data = load_hf_dataset(**hf_args)
        self.data = add_dataset_index(self.data)
        self.text_key = text_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text_content = self.data[idx].get(self.text_key, "")
        tokenized_data = self.tokenizer(
            str(text_content),
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = tokenized_data["input_ids"]
        labels = list(input_ids)
        if labels:
            labels[0] = IGNORE_INDEX

        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(
                tokenized_data.get("attention_mask", [1] * len(input_ids))
            ),
            "index": self.data[idx]["index"],
        }


class DocumentChunkDataset(Dataset):
    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        text_key="text",
        max_length=1024,
        stride=None,
        min_length=1,
        drop_last=False,
        add_special_tokens=True,
        tokenize_batch_size=64,
    ):
        super(DocumentChunkDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.stride = int(stride) if stride not in (None, "") else self.max_length
        self.min_length = int(min_length)
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if self.min_length <= 0:
            raise ValueError("min_length must be positive")
        self.drop_last = self._as_bool(drop_last)
        self.add_special_tokens = self._as_bool(add_special_tokens)
        self.tokenize_batch_size = int(tokenize_batch_size)
        if self.tokenize_batch_size <= 0:
            raise ValueError("tokenize_batch_size must be positive")
        self.data = load_hf_dataset(**hf_args)
        self.data = add_dataset_index(self.data)
        self.text_key = text_key
        self.chunk_token_ids = array("I")
        self.chunks = self._build_chunks()

    @staticmethod
    def _as_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes", "y")
        return bool(value)

    def _with_special_tokens(self, token_ids):
        if not self.add_special_tokens:
            return token_ids
        if hasattr(self.tokenizer, "build_inputs_with_special_tokens"):
            return self.tokenizer.build_inputs_with_special_tokens(token_ids)
        return token_ids

    def _content_length(self):
        if not self.add_special_tokens:
            return self.max_length
        if hasattr(self.tokenizer, "build_inputs_with_special_tokens"):
            num_special_tokens = len(self.tokenizer.build_inputs_with_special_tokens([]))
        else:
            num_special_tokens = len(
                self.tokenizer("", add_special_tokens=True)["input_ids"]
            )
        return max(1, self.max_length - num_special_tokens)

    def _tokenize_document(self, text):
        try:
            return self.tokenizer(
                str(text),
                add_special_tokens=False,
                verbose=False,
            )["input_ids"]
        except TypeError:
            return self.tokenizer(str(text), add_special_tokens=False)["input_ids"]

    def _tokenize_documents(self, texts):
        try:
            return self.tokenizer(
                [str(text) for text in texts],
                add_special_tokens=False,
                verbose=False,
            )["input_ids"]
        except TypeError:
            return [self._tokenize_document(text) for text in texts]

    def _build_chunks(self):
        content_length = self._content_length()
        chunks = []
        progress = tqdm(
            range(0, len(self.data), self.tokenize_batch_size),
            desc=(
                f"Building document chunks "
                f"(max_length={self.max_length}, stride={self.stride}, "
                f"tokenize_batch_size={self.tokenize_batch_size})"
            ),
            total=(len(self.data) + self.tokenize_batch_size - 1)
            // self.tokenize_batch_size,
            dynamic_ncols=True,
        )
        for batch_start in progress:
            batch_end = min(batch_start + self.tokenize_batch_size, len(self.data))
            text_batch = [
                self.data[doc_idx].get(self.text_key, "")
                for doc_idx in range(batch_start, batch_end)
            ]
            tokenized_batch = self._tokenize_documents(text_batch)
            for doc_idx, token_ids in zip(range(batch_start, batch_end), tokenized_batch):
                for start in range(0, len(token_ids), self.stride):
                    chunk_length = min(content_length, len(token_ids) - start)
                    if chunk_length < self.min_length:
                        continue
                    if self.drop_last and chunk_length < content_length:
                        continue
                    input_ids = self._with_special_tokens(
                        token_ids[start : start + chunk_length]
                    )
                    token_offset = len(self.chunk_token_ids)
                    self.chunk_token_ids.extend(input_ids)
                    chunks.append(
                        {
                            "index": self.data[doc_idx]["index"],
                            "chunk_index": len(chunks),
                            "token_offset": token_offset,
                            "token_length": len(input_ids),
                        }
                    )
                    if start + content_length >= len(token_ids):
                        break
            progress.set_postfix(chunks=len(chunks), refresh=False)
        return chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        start = chunk["token_offset"]
        end = start + chunk["token_length"]
        input_ids = self.chunk_token_ids[start:end].tolist()
        labels = list(input_ids)
        if labels:
            labels[0] = IGNORE_INDEX

        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "index": chunk["index"],
        }


class PretrainingDataset(Dataset):
    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        text_key="text",
        max_length=2048,
        chunk_batch_size=128,
    ):
        super(PretrainingDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.chunk_batch_size = int(chunk_batch_size)
        self.chunks = self._chunk_raw_text(load_hf_dataset(**hf_args)[text_key])

    def _chunk_raw_text(self, raw_text):
        separator_ids = self.tokenizer("\n\n", add_special_tokens=False)["input_ids"]
        chunks = []
        token_buffer = []
        text_batch = []
        seen_text = False

        def flush_text_batch():
            nonlocal token_buffer, text_batch, seen_text
            if not text_batch:
                return

            batch_text = "\n\n".join(text_batch)
            batch_token_ids = self.tokenizer(batch_text, add_special_tokens=False)[
                "input_ids"
            ]
            if seen_text and batch_token_ids:
                token_buffer.extend(separator_ids)
            token_buffer.extend(batch_token_ids)
            seen_text = True
            text_batch = []

            while len(token_buffer) >= self.max_length:
                chunks.append(self.tokenizer.decode(token_buffer[: self.max_length]))
                token_buffer = token_buffer[self.max_length :]

        for text in raw_text:
            if text is None:
                continue
            text_batch.append(str(text))
            if len(text_batch) >= self.chunk_batch_size:
                flush_text_batch()

        flush_text_batch()
        if token_buffer:
            chunks.append(self.tokenizer.decode(token_buffer))
        return chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return preprocess_pretraining_instance(
            self.tokenizer, "", self.chunks[idx], self.max_length
        )
