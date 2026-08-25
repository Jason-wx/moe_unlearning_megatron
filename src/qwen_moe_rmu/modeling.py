import logging
from typing import Any

import torch
import transformers
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def _dtype_from_config(dtype: Any):
    if dtype in (torch.bfloat16, torch.float16, torch.float32):
        return dtype
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    if dtype is None:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def _get_auto_model_class(model_handler: str | None):
    if model_handler:
        if not hasattr(transformers, model_handler):
            raise ValueError(
                f"transformers.{model_handler} is not available. "
                "Upgrade transformers or choose another model_handler."
            )
        return getattr(transformers, model_handler)

    for candidate in (
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "AutoModelForCausalLM",
    ):
        if hasattr(transformers, candidate):
            return getattr(transformers, candidate)

    raise ValueError("No suitable transformers AutoModel class is available.")


def get_qwen_moe_model(model_cfg: DictConfig):
    model_args = OmegaConf.to_container(model_cfg.model_args, resolve=True)
    tokenizer_args = OmegaConf.to_container(model_cfg.tokenizer_args, resolve=True)

    model_path = model_args.pop("pretrained_model_name_or_path")
    dtype_cfg = model_args.pop("dtype", model_args.pop("torch_dtype", None))
    dtype = _dtype_from_config(dtype_cfg)
    model_args.setdefault("trust_remote_code", True)
    tokenizer_args.setdefault("trust_remote_code", True)

    model_handler = model_cfg.get("model_handler", None)
    model_cls = _get_auto_model_class(model_handler)
    logger.info("Loading Qwen MoE model with %s from %s", model_cls.__name__, model_path)

    try:
        model = model_cls.from_pretrained(
            model_path,
            dtype=dtype,
            **model_args,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load {model_path} with {model_cls.__name__}. "
            "For Qwen3.5 MoE, make sure the installed transformers version "
            "supports qwen3_5_moe."
        ) from exc
    if hasattr(model, "config"):
        model.config.use_cache = False
        if hasattr(model.config, "text_config"):
            model.config.text_config.use_cache = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(**tokenizer_args)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
