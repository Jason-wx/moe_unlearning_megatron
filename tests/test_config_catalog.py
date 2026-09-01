from __future__ import annotations

import unittest
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


class ConfigCatalogTests(unittest.TestCase):
    def test_all_yaml_files_are_mappings(self) -> None:
        for path in sorted((REPO / "configs").rglob("*.yaml")):
            with self.subTest(path=path.relative_to(REPO)), path.open(encoding="utf-8") as handle:
                self.assertIsInstance(yaml.safe_load(handle), dict)

    def test_component_names_match_catalog(self) -> None:
        for path in sorted((REPO / "configs" / "algorithms").glob("*.yaml")):
            with path.open(encoding="utf-8") as handle:
                algorithm = yaml.safe_load(handle)["algorithm"]
            self.assertIn("name", algorithm)
            self.assertIn("deepspeed_trainer", algorithm)

    def test_deepspeed_world_size_matches_accelerate(self) -> None:
        for path in sorted((REPO / "configs" / "backends" / "deepspeed").glob("*.yaml")):
            with path.open(encoding="utf-8") as handle:
                backend = yaml.safe_load(handle)["backends"]["deepspeed"]
            accelerate_path = REPO / backend["accelerate_config"]
            with accelerate_path.open(encoding="utf-8") as handle:
                accelerate = yaml.safe_load(handle)
            self.assertEqual(backend["world_size"], accelerate["num_processes"])

    def test_all_training_configs_use_unit_gradient_clipping(self) -> None:
        for path in sorted((REPO / "configs" / "training").rglob("*.yaml")):
            with self.subTest(path=path.relative_to(REPO)), path.open(encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
            self.assertEqual(config["training"]["gradient_clip"], 1.0)
            deepspeed = config["backends"].get("deepspeed")
            if deepspeed is not None:
                self.assertEqual(
                    deepspeed["deepspeed_config"]["gradient_clipping"],
                    1.0,
                )

    def test_training_catalog_has_one_eight_gpu_config_per_algorithm(self) -> None:
        training_root = REPO / "configs" / "training"
        for backend_dir in sorted(path for path in training_root.iterdir() if path.is_dir()):
            for model_dir in sorted(path for path in backend_dir.iterdir() if path.is_dir()):
                algorithms = []
                for path in sorted(model_dir.glob("*.yaml")):
                    with path.open(encoding="utf-8") as handle:
                        config = yaml.safe_load(handle)
                    algorithms.append(config["algorithm"]["name"])
                    backend = config["backends"][backend_dir.name]
                    world_size = (
                        backend["parallel"]["world_size"]
                        if backend_dir.name == "megatron"
                        else backend["world_size"]
                    )
                    self.assertEqual(world_size, 8, path.relative_to(REPO))
                    if backend_dir.name == "megatron":
                        self.assertEqual(config["training"]["lr_scheduler_type"], "cosine")
                        self.assertEqual(config["training"]["lr_warmup_ratio"], 0.03)
                        self.assertEqual(config["training"]["lr_warmup_init"], 0.0)
                        self.assertEqual(config["training"]["lr_decay_steps"], "auto")
                        self.assertIn("output_name", config["training"])
                        self.assertNotIn("output_dir", config["training"])
                        self.assertEqual(
                            backend["output_root"],
                            "/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/wangxiao1/"
                            "checkpoints/unlearn/megatron/v3",
                            path.relative_to(REPO),
                        )
                self.assertEqual(
                    len(algorithms),
                    len(set(algorithms)),
                    f"duplicate algorithm configs in {model_dir.relative_to(REPO)}",
                )

        interns2 = training_root / "megatron" / "interns2"
        self.assertEqual(len(list(interns2.glob("*.yaml"))), 13)

    def test_interns2_megatron_empirical_hyperparameters_are_locked(self) -> None:
        expected = {
            "ceu": (1.0e-6, {"gamma": 1.0, "ignore_first_n_tokens": 1}),
            "grad_ascent": (1.0e-6, {"gamma": 1.0}),
            "grad_diff": (1.0e-6, {"gamma": 1.0, "alpha": 1.0}),
            "npo": (1.0e-6, {"beta": 0.1, "gamma": 1.0}),
            "pdu": (
                1.0e-6,
                {
                    "gamma": 1.0,
                    "alpha": 100.0,
                    "retain_loss_eps": 0.3,
                    "primal_dual": True,
                    "dual_step_size": 5.0,
                },
            ),
            "pure_npo": (1.0e-6, {"beta": 0.1, "gamma": 1.0}),
            "rmu": (
                1.0e-5,
                {
                    "gamma": 1.0,
                    "alpha": 350.0,
                    "target_layer": 9,
                    "train_layers": [7, 8, 9],
                    "steering_coeff": 300.0,
                    "control_norm_mode": "fixed",
                    "control_scale_coeff": 1.0,
                },
            ),
            "satimp": (
                4.0e-6,
                {"beta1": 2.0, "beta2": 0.1, "gamma": 1.0, "alpha": 0.0},
            ),
            "simnpo": (
                1.0e-6,
                {"beta": 0.1, "delta": 0.0, "gamma": 1.0, "alpha": 0.1},
            ),
            "token_npo": (1.0e-6, {"beta": 0.1, "gamma": 1.0}),
            "token_simnpo": (
                1.0e-6,
                {"beta": 0.1, "delta": 0.0, "gamma": 1.0, "alpha": 0.1},
            ),
            "undial": (1.0e-6, {"beta": 3.0, "gamma": 1.0, "alpha": 0.0}),
            "wga": (1.0e-6, {"beta": 0.5, "gamma": 1.0, "alpha": 0.0}),
        }
        root = REPO / "configs" / "training" / "megatron" / "interns2"
        for algorithm, (learning_rate, algorithm_fields) in expected.items():
            with self.subTest(algorithm=algorithm), (root / f"{algorithm}.yaml").open(
                encoding="utf-8"
            ) as handle:
                config = yaml.safe_load(handle)
            self.assertEqual(config["training"]["learning_rate"], learning_rate)
            for field, value in algorithm_fields.items():
                self.assertEqual(config["algorithm"][field], value)

if __name__ == "__main__":
    unittest.main()
