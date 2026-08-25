"""Small command surface for both supported training backends."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .configuration import load_experiment
from .runtime_paths import RuntimePaths


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _load(args: argparse.Namespace):
    components = {
        kind: value
        for kind in ("model", "dataset", "algorithm", "profile")
        if (value := getattr(args, kind, None))
    }
    overrides = {}
    if getattr(args, "name", None):
        overrides["name"] = args.name
    if getattr(args, "output_dir", None):
        overrides.setdefault("training", {})["output_dir"] = str(args.output_dir)
    if getattr(args, "train_steps", None) is not None:
        overrides.setdefault("training", {})["train_steps"] = args.train_steps
    if getattr(args, "train_epochs", None) is not None:
        training = overrides.setdefault("training", {})
        training["train_steps"] = -1
        training["num_train_epochs"] = args.train_epochs
    if getattr(args, "save_at_steps", None):
        overrides.setdefault("training", {})["save_steps"] = sorted(set(args.save_at_steps))
    if getattr(args, "save_at_epoch_end", False):
        training = overrides.setdefault("training", {})
        training["save_every_epoch"] = True
        training["save_final_checkpoint"] = False
    if getattr(args, "save_only_at_end", False):
        training = overrides.setdefault("training", {})
        training["save_every_epoch"] = False
        training["save_final_checkpoint"] = True
    return load_experiment(
        args.config,
        components=components or None,
        overrides=overrides or None,
    )


def _train(args: argparse.Namespace) -> int:
    from .backends import launch

    config = _load(args)
    return launch(config, args.backend, dry_run=args.dry_run)


def _inspect(args: argparse.Namespace) -> int:
    config = _load(args)
    if args.backend:
        config.validate(backend=args.backend, check_paths=args.check_paths)
    else:
        config.validate(check_paths=args.check_paths)
    if args.format == "json":
        print(json.dumps(config.resolved(), indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print(yaml.safe_dump(config.resolved(), sort_keys=False, allow_unicode=True), end="")
    return 0


def _paths(args: argparse.Namespace) -> int:
    del args
    print(yaml.safe_dump(RuntimePaths.from_environment().as_dict(), sort_keys=False), end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ou", description="OpenUnlearning CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="train with DeepSpeed or Megatron")
    train.add_argument("--backend", choices=("deepspeed", "megatron"), required=True)
    train.add_argument("--config", type=Path, required=True)
    _add_component_arguments(train)
    train.add_argument("--dry-run", action="store_true")
    train.set_defaults(handler=_train)

    inspect = subparsers.add_parser("inspect", help="resolve and validate a config")
    inspect.add_argument("--config", type=Path, required=True)
    _add_component_arguments(inspect)
    inspect.add_argument("--backend", choices=("deepspeed", "megatron"))
    inspect.add_argument("--check-paths", action="store_true")
    inspect.add_argument("--format", choices=("yaml", "json"), default="yaml")
    inspect.set_defaults(handler=_inspect)

    paths = subparsers.add_parser("paths", help="show resolved runtime paths")
    paths.set_defaults(handler=_paths)
    return parser


def _add_component_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", help="replace the model component")
    parser.add_argument("--dataset", help="replace the dataset component")
    parser.add_argument("--algorithm", help="replace the algorithm component")
    parser.add_argument("--profile", help="replace the training profile component")
    parser.add_argument("--name", help="override the generated experiment name")
    parser.add_argument("--output-dir", type=Path, help="override training.output_dir")
    length = parser.add_mutually_exclusive_group()
    length.add_argument(
        "--train-steps",
        type=_positive_int,
        help="override the total number of optimizer steps (takes precedence over epochs)",
    )
    length.add_argument(
        "--train-epochs",
        type=_positive_int,
        help="train complete epochs and disable a configured fixed step limit",
    )
    parser.add_argument(
        "--save-at-step",
        dest="save_at_steps",
        action="append",
        type=_positive_int,
        help="export a checkpoint after this optimizer step; may be repeated (Megatron)",
    )
    save_mode = parser.add_mutually_exclusive_group()
    save_mode.add_argument(
        "--save-at-epoch-end",
        action="store_true",
        help="export a checkpoint after every completed epoch",
    )
    save_mode.add_argument(
        "--save-only-at-end",
        action="store_true",
        help="disable epoch checkpoints and export one checkpoint after the final step",
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
