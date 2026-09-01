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
DEEPSPEED_EXPERIMENT = (
    REPO / "configs" / "training" / "deepspeed" / "interns2" / "pure_npo.yaml"
)
MEGATRON_EXPERIMENT = (
    REPO / "configs" / "training" / "megatron" / "interns2" / "pure_npo.yaml"
)
WGA_EXPERIMENT = (
    REPO / "configs" / "training" / "megatron" / "interns2" / "wga.yaml"
)
SATIMP_EXPERIMENT = (
    REPO / "configs" / "training" / "megatron" / "interns2" / "satimp.yaml"
)
RMU_EXPERIMENT = REPO / "configs" / "training" / "megatron" / "interns2" / "rmu.yaml"
HYDRA_BRIDGE = REPO / "configs" / "internal" / "deepspeed"


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
        self.deepspeed_config = load_experiment(
            DEEPSPEED_EXPERIMENT, environment=self.environment
        )
        self.megatron_config = load_experiment(
            MEGATRON_EXPERIMENT, environment=self.environment
        )

    def test_backend_specific_configs_validate(self) -> None:
        self.deepspeed_config.validate(backend="deepspeed")
        self.megatron_config.validate(backend="megatron")
        with self.assertRaisesRegex(ValueError, "not enabled"):
            self.deepspeed_config.validate(backend="megatron")
        with self.assertRaisesRegex(ValueError, "not enabled"):
            self.megatron_config.validate(backend="deepspeed")

    def test_all_training_configs_validate_their_single_backend(self) -> None:
        for path in sorted((REPO / "configs" / "training").rglob("*.yaml")):
            with self.subTest(experiment=path.name):
                config = load_experiment(path, environment=self.environment)
                self.assertEqual(len(config.payload["backends"]), 1)
                config.validate(backend=next(iter(config.payload["backends"])))

    def test_training_config_is_self_contained(self) -> None:
        self.assertNotIn("defaults", self.deepspeed_config.payload)
        self.assertEqual(
            set(self.deepspeed_config.payload),
            {"name", "model", "data", "algorithm", "training", "backends"},
        )
        self.assertIn("deepspeed_config", self.deepspeed_config.backend("deepspeed"))
        self.assertIn("parallel", self.megatron_config.backend("megatron"))

    def test_backend_capability_matrix_rejects_unsupported_pairs(self) -> None:
        rmu = load_experiment(RMU_EXPERIMENT, environment=self.environment)
        with self.assertRaisesRegex(ValueError, "not enabled"):
            rmu.validate(backend="deepspeed")
        rmu.validate(backend="megatron")

    def test_deepspeed_command_is_forget_only_pure_npo(self) -> None:
        previous = os.environ.copy()
        os.environ.update(self.environment)
        try:
            command = build_command(self.deepspeed_config)
        finally:
            os.environ.clear()
            os.environ.update(previous)
        joined = " ".join(command)
        self.assertIn("trainer=PureNPO", command)
        self.assertIn("~data.retain", command)
        self.assertIn("gradient_accumulation_steps=4", joined)
        self.assertIn("/deepspeed", joined)

    def test_deepspeed_command_composes_in_legacy_hydra(self) -> None:
        previous = os.environ.copy()
        os.environ.update(self.environment)
        try:
            command = build_command(self.deepspeed_config)
        finally:
            os.environ.clear()
            os.environ.update(previous)
        marker = command.index("--config-name=unlearn.yaml")
        with initialize_config_dir(config_dir=str(HYDRA_BRIDGE), version_base=None):
            merged = compose(config_name="unlearn.yaml", overrides=command[marker + 1 :])
        self.assertEqual(merged.trainer.handler, "PureNPO")
        self.assertIsNone(merged.data.get("retain"))
        self.assertEqual(merged.trainer.args.gradient_accumulation_steps, 4)
        self.assertEqual(merged.trainer.method_args.beta, 0.1)

    def test_all_supported_deepspeed_algorithms_compose_in_minimal_bridge(self) -> None:
        root = REPO / "configs" / "training" / "deepspeed" / "interns2"
        for path in sorted(root.glob("*.yaml")):
            with self.subTest(config=path.name):
                config = load_experiment(path, environment=self.environment)
                command = build_command(config)
                marker = command.index("--config-name=unlearn.yaml")
                with initialize_config_dir(config_dir=str(HYDRA_BRIDGE), version_base=None):
                    merged = compose(
                        config_name="unlearn.yaml",
                        overrides=command[marker + 1 :],
                    )
                self.assertTrue(merged.trainer.handler)

    def test_megatron_mapping_is_forget_only_pure_npo(self) -> None:
        mapping = to_megatron_mapping(self.megatron_config)
        self.assertEqual(mapping["algorithm"]["name"], "pure_npo")
        self.assertIsNone(mapping["data"]["retain_cache"])
        self.assertEqual(mapping["training"]["micro_batch_size"], 4)
        self.assertEqual(mapping["training"]["global_batch_size"], 4)
        self.assertEqual(mapping["training"]["train_steps"], -1)
        self.assertEqual(mapping["training"]["num_train_epochs"], 1)
        self.assertEqual(mapping["data"]["sequence_length"], 4096)
        self.assertFalse(mapping["data"]["shuffle"])
        self.assertEqual(mapping["training"]["trainable_param_mode"], "full")
        self.assertEqual(mapping["optimizer"]["lr_scheduler_type"], "cosine")
        self.assertEqual(mapping["optimizer"]["lr_warmup_ratio"], 0.03)
        self.assertEqual(mapping["optimizer"]["lr_warmup_init"], 0.0)
        self.assertEqual(mapping["optimizer"]["lr_decay_steps"], "auto")
        self.assertEqual(
            mapping["training"]["output_dir"],
            str(
                Path(self.megatron_config.backend("megatron")["output_root"])
                / self.megatron_config.name
            ),
        )

    def test_five_npo_family_config_names_remain_distinct(self) -> None:
        expected = ("npo", "pure_npo", "token_npo", "simnpo", "token_simnpo")
        for backend in ("deepspeed", "megatron"):
            root = REPO / "configs" / "training" / backend / "interns2"
            actual = tuple(
                load_experiment(root / f"{name}.yaml", environment=self.environment).algorithm_name
                for name in expected
            )
            self.assertEqual(actual, expected)

    def test_wga_is_forget_only_and_uses_selected_optimizer_settings(self) -> None:
        config = load_experiment(WGA_EXPERIMENT, environment=self.environment)
        config.validate(backend="megatron")
        self.assertFalse(config.uses_retain)

        mapping = to_megatron_mapping(config)
        self.assertIsNone(mapping["data"]["retain_cache"])
        self.assertEqual(mapping["algorithm"]["beta"], 0.5)
        self.assertEqual(mapping["algorithm"]["alpha"], 0.0)
        self.assertEqual(mapping["optimizer"]["learning_rate"], 1.0e-6)
        self.assertEqual(mapping["optimizer"]["gradient_clip"], 1.0)
        self.assertEqual(mapping["training"]["save_steps"], [])
        self.assertTrue(mapping["training"]["save_every_epoch"])
        self.assertFalse(mapping["training"]["save_final_checkpoint"])

    def test_satimp_is_forget_only_and_uses_selected_settings(self) -> None:
        config = load_experiment(SATIMP_EXPERIMENT, environment=self.environment)
        config.validate(backend="megatron")
        self.assertFalse(config.uses_retain)

        mapping = to_megatron_mapping(config)
        self.assertIsNone(mapping["data"]["retain_cache"])
        self.assertEqual(mapping["algorithm"]["beta1"], 2.0)
        self.assertEqual(mapping["algorithm"]["beta2"], 0.1)
        self.assertEqual(mapping["algorithm"]["alpha"], 0.0)
        self.assertEqual(mapping["optimizer"]["learning_rate"], 4.0e-6)
        self.assertEqual(mapping["optimizer"]["gradient_clip"], 1.0)
        self.assertEqual(mapping["training"]["train_steps"], -1)
        self.assertFalse(mapping["training"]["save_final_checkpoint"])
        self.assertTrue(mapping["training"]["save_every_epoch"])

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
                    str(MEGATRON_EXPERIMENT),
                    "--train-steps",
                    "300",
                    "--save-only-at-end",
                    "--output-name",
                    "interns2-pure-npo-300steps",
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
            str(
                Path(config.backend("megatron")["output_root"])
                / "interns2-pure-npo-300steps"
            ),
        )

    def test_megatron_output_name_cannot_escape_output_root(self) -> None:
        for unsafe_name in ("../escape", "/tmp/escape", "nested/escape", ".."):
            with self.subTest(output_name=unsafe_name):
                with self.assertRaisesRegex(ValueError, "safe single directory name"):
                    load_experiment(
                        MEGATRON_EXPERIMENT,
                        environment=self.environment,
                        overrides={"training": {"output_name": unsafe_name}},
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
                    str(MEGATRON_EXPERIMENT),
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
