.PHONY: quality style test

PYTHON ?= envs/open-unlearning/bin/python
RUFF_CACHE_DIR ?= /tmp/open-unlearning-ruff
check_dirs := src/open_unlearning src/megatron_unlearning src/data/build_document_chunk_cache.py tests

quality:
	PYTHONPATH=src RUFF_CACHE_DIR=$(RUFF_CACHE_DIR) $(PYTHON) -m ruff check $(check_dirs)
	PYTHONPATH=src RUFF_CACHE_DIR=$(RUFF_CACHE_DIR) $(PYTHON) -m ruff format --check $(check_dirs)

style:
	PYTHONPATH=src RUFF_CACHE_DIR=$(RUFF_CACHE_DIR) $(PYTHON) -m ruff check $(check_dirs) --fix
	PYTHONPATH=src RUFF_CACHE_DIR=$(RUFF_CACHE_DIR) $(PYTHON) -m ruff format $(check_dirs)

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v
