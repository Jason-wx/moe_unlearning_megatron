import os
import sys

import hydra
from omegaconf import DictConfig

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data import get_collators, get_data  # noqa: E402
from evals import get_evaluators  # noqa: E402
from qwen_moe_rmu.modeling import get_qwen_moe_model  # noqa: E402
from qwen_moe_rmu.trainer import QwenMoeRMUTrainer  # noqa: E402
from trainer import load_trainer_args  # noqa: E402
from trainer.utils import seed_everything  # noqa: E402


@hydra.main(version_base=None, config_path="../../configs", config_name="qwen_moe_rmu/unlearn")
def main(cfg: DictConfig):
    seed_everything(cfg.trainer.args.seed)
    mode = cfg.get("mode", "unlearn")

    model, tokenizer = get_qwen_moe_model(cfg.model)
    template_args = cfg.model.template_args

    data = get_data(cfg.data, mode=mode, tokenizer=tokenizer, template_args=template_args)
    collator = get_collators(cfg.collator, tokenizer=tokenizer)

    evaluators = None
    if cfg.get("eval", None):
        evaluators = get_evaluators(
            eval_cfgs=cfg.eval,
            template_args=template_args,
            model=model,
            tokenizer=tokenizer,
        )

    trainer_args = load_trainer_args(cfg.trainer.args, data.get("train", None))
    trainer = QwenMoeRMUTrainer(
        model=model,
        train_dataset=data.get("train", None),
        eval_dataset=data.get("eval", None),
        processing_class=tokenizer,
        data_collator=collator,
        args=trainer_args,
        evaluators=evaluators,
        template_args=template_args,
        **cfg.trainer.method_args,
    )

    if trainer_args.do_train:
        trainer.train()
        trainer.save_state()
        trainer.save_model(trainer_args.output_dir)

    if trainer_args.do_eval:
        trainer.evaluate(metric_key_prefix="eval")


if __name__ == "__main__":
    main()

