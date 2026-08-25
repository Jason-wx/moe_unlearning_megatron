import os
import sys

import hydra
from omegaconf import DictConfig

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data import get_collators, get_data  # noqa: E402
from evals import get_evaluators  # noqa: E402
from gemma_moe_rmu.modeling import get_gemma_moe_models  # noqa: E402
from gemma_moe_rmu.trainer import GemmaMoeRMUTrainer  # noqa: E402
from trainer import load_trainer_args  # noqa: E402
from trainer.utils import seed_everything  # noqa: E402


@hydra.main(version_base=None, config_path="../../configs", config_name="gemma_moe_rmu/unlearn")
def main(cfg: DictConfig):
    seed_everything(cfg.trainer.args.seed)
    model, ref_model, tokenizer = get_gemma_moe_models(cfg.model)
    template_args = cfg.model.template_args
    data = get_data(
        cfg.data,
        mode=cfg.get("mode", "unlearn"),
        tokenizer=tokenizer,
        template_args=template_args,
    )
    collator = get_collators(cfg.collator, tokenizer=tokenizer)

    evaluators = None
    if cfg.get("eval", None):
        evaluators = get_evaluators(
            eval_cfgs=cfg.eval,
            template_args=template_args,
            model=model,
            tokenizer=tokenizer,
        )

    trainer_args = load_trainer_args(cfg.trainer.args, data.get("train"))
    trainer = GemmaMoeRMUTrainer(
        model=model,
        train_dataset=data.get("train"),
        eval_dataset=data.get("eval"),
        processing_class=tokenizer,
        data_collator=collator,
        args=trainer_args,
        evaluators=evaluators,
        template_args=template_args,
        ref_model=ref_model,
        **cfg.trainer.method_args,
    )
    if trainer_args.do_train:
        trainer.train()
        # This Gemma RMU entry point intentionally saves only the final model.
        # Optimizer/scheduler/RNG and Trainer state are not persisted.
        trainer.save_model(trainer_args.output_dir)
    if trainer_args.do_eval:
        trainer.evaluate(metric_key_prefix="eval")


if __name__ == "__main__":
    main()
