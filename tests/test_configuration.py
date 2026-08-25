from __future__ import annotations

import os
import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir

from open_unlearning.backends.deepspeed import build_command
from open_unlearning.backends.megatron import to_megatron_mapping
from open_unlearning.cli import _load, build_parser
from open_unlearning.configuration import load_experiment
from open_unlearning.runtime_paths import RuntimePaths

REPO = Path(__file__).resolve().parents[1]
EXPERIMENT = REPO / "configs" / "experiments" / "interns2_pure_npo.yaml"


class RuntimePathTests(unittest.TestCase):
    def test_repository_layout_is_the_default(self) -> None:
        paths = RuntimePaths.from_environment(repo=REPO, environment={})
        self.assertEqual(paths.cache, REPO / "data_cache")
        self.assertEqual(paths.checkpoints, REPO / "saves")
        self.assertEqual(paths.runs, REPO / "outputs")
        self.assertEqual(paths.evaluations, REPO / "evaluations")
        self.assertEqual(paths.env, REPO / "envs" / "open-unlearning")

    def test_external_runtime_layout(self) -> None:
        runtime = REPO.parent / "open-unlearning-runtime"
        paths = RuntimePaths.from_environment(
            repo=REPO, environment={"OU_RUNTIME_ROOT": str(runtime)}
        )
        self.assertEqual(paths.cache, runtime / "caches")
        self.assertEqual(paths.checkpoints, runtime / "checkpoints")
        self.assertEqual(paths.runs, runtime / "runs")
        self.assertEqual(paths.evaluations, runtime / "evaluations")


class SharedConfigurationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = {
            "OU_REPO_ROOT": str(REPO),
            "OU_RUNTIME_ROOT": str(REPO.parent / "runtime-test"),
            "OU_MEGATRON_CHECKPOINT_ROOT": str(REPO.parent / "megatron-checkpoints"),
            "INTERNS2_MODEL_PATH": "/models/interns2",
        }
        self.config = load_experiment(EXPERIMENT, environment=self.environment)

    def test_both_backends_validate(self) -> None:
        self.config.validate(backend="deepspeed")
        self.config.validate(backend="megatron")

    def test_all_model_examples_support_both_backends(self) -> None:
        for path in sorted((REPO / "configs" / "experiments").glob("*_pure_npo.yaml")):
            with self.subTest(experiment=path.name):
                config = load_experiment(path, environment=self.environment)
                config.validate(backend="deepspeed")
                config.validate(backend="megatron")

    def test_components_can_replace_repeated_algorithm_scripts(self) -> None:
        config = load_experiment(
            EXPERIMENT,
            environment=self.environment,
            components={"algorithm": "grad_diff", "profile": "wmdp_moe_default"},
        )
        self.assertEqual(config.algorithm_name, "grad_diff")
        self.assertEqual(config.name, "interns2-grad-diff-wmdp-bio")
        self.assertTrue(config.payload["training"]["output_dir"].endswith(config.name))
        config.validate(backend="deepspeed")
        config.validate(backend="megatron")

    def test_backend_capability_matrix_rejects_unsupported_pairs(self) -> None:
        token_npo = load_experiment(
            EXPERIMENT,
            environment=self.environment,
            components={"algorithm": "token_npo"},
        )
        with self.assertRaisesRegex(ValueError, "deepspeed does not support"):
            token_npo.validate(backend="deepspeed")
        token_npo.validate(backend="megatron")

        rmu = load_experiment(
            EXPERIMENT,
            environment=self.environment,
            components={"algorithm": "rmu"},
        )
        with self.assertRaisesRegex(ValueError, "deepspeed does not support"):
            rmu.validate(backend="deepspeed")
        rmu.validate(backend="megatron")

    def test_deepspeed_command_is_forget_only_npo(self) -> None:
        previous = os.environ.copy()
        os.environ.update(self.environment)
        try:
            command = build_command(self.config)
        finally:
            os.environ.clear()
            os.environ.update(previous)
        joined = " ".join(command)
        self.assertIn("trainer=NPO", command)
        self.assertIn("~data.retain", command)
        self.assertIn("gradient_accumulation_steps=4", joined)
        self.assertIn("/deepspeed", joined)

    def test_deepspeed_command_composes_in_legacy_hydra(self) -> None:
        previous = os.environ.copy()
        os.environ.update(self.environment)
        try:
            command = build_command(self.config)
        finally:
            os.environ.clear()
            os.environ.update(previous)
        marker = command.index("--config-name=unlearn.yaml")
        with initialize_config_dir(config_dir=str(REPO / "configs"), version_base=None):
            merged = compose(config_name="unlearn.yaml", overrides=command[marker + 1 :])
        self.assertEqual(merged.trainer.handler, "NPO")
        self.assertIsNone(merged.data.get("retain"))
        self.assertEqual(merged.trainer.args.gradient_accumulation_steps, 4)
        self.assertEqual(merged.trainer.method_args.beta, 0.1)

    def test_megatron_mapping_is_forget_only_npo(self) -> None:
        mapping = to_megatron_mapping(self.config)
        self.assertEqual(mapping["algorithm"]["name"], "npo")
        self.assertIsNone(mapping["data"]["retain_cache"])
        self.assertEqual(mapping["training"]["micro_batch_size"], 16)
        self.assertEqual(mapping["training"]["global_batch_size"], 128)
        self.assertEqual(
            mapping["training"]["output_dir"],
            str(REPO.parent / "megatron-checkpoints" / self.config.name),
        )

    def test_cli_can_limit_steps_and_save_only_the_final_checkpoint(self) -> None:
        previous = os.environ.copy()
        os.environ.update(self.environment)
        try:
            args = build_parser().parse_args(
                [
                    "train",
                    "--backend",
                    "megatron",
                    "--config",
                    str(EXPERIMENT),
                    "--train-steps",
                    "300",
                    "--save-only-at-end",
                    "--output-dir",
                    "saves/interns2-pure-npo-300steps",
                    "--dry-run",
                ]
            )
            config = _load(args)
        finally:
            os.environ.clear()
            os.environ.update(previous)

        mapping = to_megatron_mapping(config)
        self.assertEqual(mapping["training"]["train_steps"], 300)
        self.assertFalse(mapping["training"]["save_every_epoch"])
        self.assertTrue(mapping["training"]["save_final_checkpoint"])
        self.assertEqual(
            mapping["training"]["output_dir"],
            str(REPO.parent / "megatron-checkpoints" / "interns2-pure-npo-300steps"),
        )

    def test_cli_can_save_step_300_and_continue_to_epoch_end(self) -> None:
        previous = os.environ.copy()
        os.environ.update(self.environment)
        try:
            args = build_parser().parse_args(
                [
                    "train",
                    "--backend",
                    "megatron",
                    "--config",
                    str(EXPERIMENT),
                    "--train-epochs",
                    "1",
                    "--save-at-step",
                    "300",
                    "--save-at-epoch-end",
                    "--dry-run",
                ]
            )
            config = _load(args)
        finally:
            os.environ.clear()
            os.environ.update(previous)

        mapping = to_megatron_mapping(config)
        self.assertEqual(mapping["training"]["train_steps"], -1)
        self.assertEqual(mapping["training"]["num_train_epochs"], 1)
        self.assertEqual(mapping["training"]["save_steps"], [300])
        self.assertTrue(mapping["training"]["save_every_epoch"])
        self.assertFalse(mapping["training"]["save_final_checkpoint"])


if __name__ == "__main__":
    unittest.main()
