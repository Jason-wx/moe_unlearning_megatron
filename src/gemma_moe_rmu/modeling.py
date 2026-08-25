import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
import transformers
import deepspeed
from omegaconf import DictConfig, OmegaConf
from transformers.integrations import HfDeepSpeedConfig

logger = logging.getLogger(__name__)


def _dtype_from_config(dtype: Any):
    if dtype in (torch.bfloat16, torch.float16, torch.float32):
        return dtype
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    if dtype == "float32" or dtype is None:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def _disable_cache(model):
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False


def get_gemma_moe_models(model_cfg: DictConfig):
    model_args = OmegaConf.to_container(model_cfg.model_args, resolve=True)
    tokenizer_args = OmegaConf.to_container(model_cfg.tokenizer_args, resolve=True)
    model_path = model_args.pop("pretrained_model_name_or_path")
    dtype_cfg = model_args.pop("dtype", model_args.pop("torch_dtype", None))
    dtype = _dtype_from_config(dtype_cfg)
    model_args.setdefault("trust_remote_code", True)
    tokenizer_args.setdefault("trust_remote_code", True)

    model_handler = model_cfg.get("model_handler", "AutoModelForMultimodalLM")
    if not hasattr(transformers, model_handler):
        raise RuntimeError(
            f"transformers.{model_handler} is unavailable. Gemma 4 requires "
            "transformers>=5.5.0."
        )
    model_cls = getattr(transformers, model_handler)
    deepspeed_config = (
        Path(__file__).resolve().parents[2]
        / "configs/accelerate/zero_stage3_offload_config.json"
    )
    if not deepspeed_config.is_file():
        raise FileNotFoundError(f"DeepSpeed config not found: {deepspeed_config}")

    # torchrun sets RANK/WORLD_SIZE before entering this function, while the
    # process group would normally be initialized later by Trainer.  ZeRO-3
    # must know the real world size before model construction to partition it.
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not torch.distributed.is_initialized():
        deepspeed.init_distributed(dist_backend="nccl")

    # HfDeepSpeedConfig must be created and kept alive before from_pretrained.
    # It makes Transformers construct parameters inside deepspeed.zero.Init,
    # so each rank loads only its ZeRO-3 shard instead of a complete model.
    # The Accelerate/Trainer config intentionally contains ``"auto"`` values.
    # Trainer resolves them later, but from_pretrained enters zero.Init before
    # Trainer exists.  DeepSpeed 0.15 therefore needs concrete values here.
    init_ds_config = json.loads(deepspeed_config.read_text(encoding="utf-8"))
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    micro_batch_size = 8
    gradient_accumulation_steps = 2
    init_ds_config["train_micro_batch_size_per_gpu"] = micro_batch_size
    init_ds_config["gradient_accumulation_steps"] = gradient_accumulation_steps
    init_ds_config["train_batch_size"] = (
        world_size * micro_batch_size * gradient_accumulation_steps
    )
    zero_config = init_ds_config["zero_optimization"]
    zero_config["reduce_bucket_size"] = 50_000_000
    zero_config["stage3_prefetch_bucket_size"] = 45_000_000
    zero_config["stage3_param_persistence_threshold"] = 50_000
    hf_deepspeed_config = HfDeepSpeedConfig(init_ds_config)

    logger.info(
        "Loading ZeRO-3 sharded Gemma active model with %s from %s",
        model_cls.__name__,
        model_path,
    )
    try:
        model = model_cls.from_pretrained(model_path, dtype=dtype, **model_args)
        logger.info("Loading ZeRO-3 sharded Gemma reference model from %s", model_path)
        ref_model = model_cls.from_pretrained(model_path, dtype=dtype, **model_args)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load Gemma 4 model {model_path} with {model_cls.__name__}. "
            "Make sure transformers>=5.5.0 is installed."
        ) from exc

    _disable_cache(model)
    _disable_cache(ref_model)
    # Keep the weak-reference target alive for the complete model lifecycle.
    model._hf_deepspeed_config_guard = hf_deepspeed_config
    ref_model._hf_deepspeed_config_guard = hf_deepspeed_config

    tokenizer = transformers.AutoTokenizer.from_pretrained(**tokenizer_args)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, ref_model, tokenizer
