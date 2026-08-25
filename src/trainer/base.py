# Modified from https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/trainer.py

import logging
import os
import re
from typing import Any, Dict, List, Optional, Union

from torch.utils.data import Dataset
from transformers import Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

logger = logging.getLogger(__name__)

# When using custom evaluators without an eval dataset, pass a dummy value
# to prevent Trainer from raising on eval_dataset=None when eval_strategy is set
_EVAL_PLACEHOLDER = "_EVAL_PLACEHOLDER"


class FinetuneTrainer(Trainer):
    def __init__(
        self,
        evaluators=None,
        template_args=None,
        trainable_param_mode=None,
        log_trainable_params=True,
        *args,
        **kwargs,
    ):
        self.evaluators = evaluators
        self.template_args = template_args
        if trainable_param_mode is not None:
            self._configure_trainable_parameters(
                kwargs["model"], trainable_param_mode, log_trainable_params
            )
        if kwargs.get("eval_dataset") is None and evaluators:
            kwargs["eval_dataset"] = _EVAL_PLACEHOLDER
        super().__init__(*args, **kwargs)

    @staticmethod
    def _configure_trainable_parameters(model, mode, log_trainable_params):
        if mode != "moe_experts":
            raise ValueError(f"Unsupported trainable_param_mode: {mode}")

        model_type = str(getattr(getattr(model, "config", None), "model_type", ""))
        is_gemma4 = "gemma4" in model_type.lower() or "gemma4" in type(model).__name__.lower()
        if is_gemma4:
            # Gemma 4 exposes sparse experts under `.experts` and its shared/dense
            # expert as the ordinary `.mlp` branch.
            patterns = (
                r"model\.language_model\.layers\.\d+\.experts\.(gate_up_proj|down_proj)",
                r"model\.language_model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)\.weight",
            )
        else:
            # Qwen3.5 and Intern-S2 share the same MoE parameter layout. Deliberately
            # exclude `gate` and `shared_expert_gate`, which are routers, not experts.
            patterns = (
                r"model\.language_model\.layers\.\d+\.mlp\.experts\.(gate_up_proj|down_proj)",
                r"model\.language_model\.layers\.\d+\.mlp\.shared_expert\.(gate_proj|up_proj|down_proj)\.weight",
            )

        compiled = tuple(re.compile(pattern) for pattern in patterns)
        matched = []
        total_numel = 0
        trainable_numel = 0
        for name, parameter in model.named_parameters():
            parameter.requires_grad = any(pattern.fullmatch(name) for pattern in compiled)
            total_numel += parameter.numel()
            if parameter.requires_grad:
                matched.append(name)
                trainable_numel += parameter.numel()

        if not matched:
            raise ValueError(
                f"No parameters matched trainable_param_mode={mode}; "
                f"model_type={model_type}, patterns={patterns}"
            )
        if log_trainable_params:
            logger.info(
                "Expert-only training enabled: %d tensors, %d/%d parameters trainable (%.4f%%).",
                len(matched),
                trainable_numel,
                total_numel,
                100.0 * trainable_numel / total_numel,
            )
            for name in matched[:80]:
                logger.info("trainable expert: %s", name)
            if len(matched) > 80:
                logger.info("... %d more trainable expert tensors omitted", len(matched) - 80)

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        trial: Dict[str, Any] = None,
    ) -> Dict[str, float]:
        # Run a custom evaluator and save results
        if self.evaluators and self.accelerator.is_local_main_process:
            if self.accelerator.num_processes != 1:
                logger.warning(
                    "Custom evaluator can be run with this Trainer only when a single accelerator process is running."
                )
                return {}

            run_dir = self._get_output_dir(trial=trial)
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            output_dir = os.path.join(run_dir, checkpoint_folder, "evals")
            os.makedirs(output_dir, exist_ok=True)
            eval_metrics = {}
            for _, evaluator in self.evaluators.items():
                eval_args = {
                    "output_dir": output_dir,
                    "template_args": self.template_args,
                    "model": self.model,
                    "tokenizer": self.processing_class,
                }
                eval_metrics.update(evaluator.evaluate(**eval_args))
            self.log(eval_metrics)
            return eval_metrics

        if eval_dataset is None or eval_dataset == _EVAL_PLACEHOLDER:
            return {}
        # Run the default HF Trainer evaluate method when eval dataset is provided
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
