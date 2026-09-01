# Modified from https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/trainer.py

import logging
import os
from typing import Any

from torch.utils.data import Dataset, SequentialSampler
from transformers import Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from open_unlearning.trainable_parameters import (
    configure_hf_trainable_parameters,
)

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
        shuffle_train=True,
        *args,
        **kwargs,
    ):
        self.evaluators = evaluators
        self.template_args = template_args
        self.shuffle_train = bool(shuffle_train)
        if trainable_param_mode is not None:
            self._configure_trainable_parameters(
                kwargs["model"], trainable_param_mode, log_trainable_params
            )
        if kwargs.get("eval_dataset") is None and evaluators:
            kwargs["eval_dataset"] = _EVAL_PLACEHOLDER
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self, train_dataset=None):
        if self.shuffle_train:
            return super()._get_train_sampler(train_dataset)
        dataset = self.train_dataset if train_dataset is None else train_dataset
        return SequentialSampler(dataset)

    @staticmethod
    def _configure_trainable_parameters(model, mode, log_trainable_params):
        summary = configure_hf_trainable_parameters(model, mode)
        if log_trainable_params:
            logger.info(
                "Trainable mode %s enabled for %s: %d tensors, %d/%d parameters "
                "trainable (%.4f%%); category_tensors=%s; category_numel=%s.",
                summary.mode,
                summary.family,
                len(summary.names),
                summary.trainable_numel,
                summary.total_numel,
                100.0 * summary.trainable_numel / summary.total_numel,
                summary.category_tensors,
                summary.category_numel,
            )
            for category, name in summary.names[:80]:
                logger.info("trainable %s: %s", category, name)
            if len(summary.names) > 80:
                logger.info("... %d more trainable tensors omitted", len(summary.names) - 80)

    def evaluate(
        self,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
        trial: dict[str, Any] | None = None,
    ) -> dict[str, float]:
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
            for evaluator in self.evaluators.values():
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
