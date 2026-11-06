from __future__ import annotations

import unittest
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
FINAL_CONFIGS = REPO / "configs" / "training" / "megatron" / "final-training-configs"
DEEPSPEED_CONFIGS = REPO / "configs" / "training" / "deepspeed"
# Only the directories that are part of the repository; early per-model configs
# under configs/training/megatron/<model>/ stay on cluster storage (.gitignore).
YAML_ROOTS = (
    REPO / "configs" / "data_processing",
    REPO / "configs" / "internal",
    FINAL_CONFIGS,
    DEEPSPEED_CONFIGS,
    REPO / "tests" / "fixtures",
)
FINAL_ALGORITHMS = {
    "CEU": "ceu",
    "GradAscent": "grad_ascent",
    "GradDiff": "grad_diff",
    "PDU": "pdu",
    "PureNPO": "pure_npo",
    "RMU": "rmu",
    "SatImp": "satimp",
    "SimNPO": "simnpo",
    "WGA": "wga",
}
FINAL_EPOCHS = {"bio": 3, "cyber": 10}


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _final_configs() -> list[Path]:
    # Trace.yaml belongs to a different codebase and is kept only as a record.
    return sorted(p for p in FINAL_CONFIGS.glob("*/*/*.yaml") if p.name != "Trace.yaml")


class ConfigCatalogTests(unittest.TestCase):
    def test_all_yaml_files_are_mappings(self) -> None:
        for root in YAML_ROOTS:
            for path in sorted(root.rglob("*.yaml")):
                with self.subTest(path=path.relative_to(REPO)):
                    self.assertIsInstance(_load(path), dict)

    def test_final_configs_cover_nine_algorithms_per_model_and_domain(self) -> None:
        domain_dirs = sorted(path for path in FINAL_CONFIGS.glob("*/*") if path.is_dir())
        self.assertEqual(
            [str(path.relative_to(FINAL_CONFIGS)) for path in domain_dirs],
            [f"{model}/{domain}" for model in ("gemma", "intern", "qwen") for domain in ("bio", "cyber")],
        )
        for domain_dir in domain_dirs:
            with self.subTest(directory=str(domain_dir.relative_to(REPO))):
                stems = {path.stem for path in domain_dir.glob("*.yaml")} - {"Trace"}
                self.assertEqual(stems, set(FINAL_ALGORITHMS))
                for stem, algorithm in FINAL_ALGORITHMS.items():
                    config = _load(domain_dir / f"{stem}.yaml")
                    self.assertEqual(config["algorithm"]["name"], algorithm)

    def test_final_configs_share_one_training_plan(self) -> None:
        for path in _final_configs():
            domain = path.parent.name
            with self.subTest(path=path.relative_to(REPO)):
                config = _load(path)
                training = config["training"]
                megatron = config["backends"]["megatron"]
                parallel = megatron["parallel"]
                self.assertEqual(list(config["backends"]), ["megatron"])
                self.assertEqual(training["global_batch_size"], 32)
                self.assertEqual(training["num_train_epochs"], FINAL_EPOCHS[domain])
                self.assertEqual(training["gradient_clip"], 1.0)
                self.assertEqual(training["lr_scheduler_type"], "cosine")
                self.assertEqual(training["lr_warmup_ratio"], 0.05)
                self.assertEqual(training["lr_warmup_init"], 0.0)
                self.assertEqual(training["lr_decay_steps"], "auto")
                self.assertEqual(training["seed"], 42)
                self.assertEqual(training["trainable_param_mode"], "full")
                self.assertIn("output_name", training)
                self.assertNotIn("output_dir", training)
                self.assertEqual(config["data"]["sequence_length"], 4096)
                self.assertFalse(config["data"]["shuffle"])
                self.assertEqual(parallel["world_size"], 8)
                self.assertEqual(parallel["pipeline_parallel"], 1)
                self.assertEqual(parallel["context_parallel"], 1)

    def test_final_configs_evaluate_their_own_wmdp_domain_online(self) -> None:
        for path in _final_configs():
            with self.subTest(path=path.relative_to(REPO)):
                evaluation = _load(path)["backends"]["megatron"]["evaluation"]
                self.assertTrue(evaluation["enabled"])
                self.assertEqual(
                    list(evaluation["tasks"]), [f"wmdp_{path.parent.name}", "mmlu"]
                )

    def test_deepspeed_configs_use_eight_gpus_and_unit_gradient_clipping(self) -> None:
        algorithms = []
        for path in sorted(DEEPSPEED_CONFIGS.rglob("*.yaml")):
            with self.subTest(path=path.relative_to(REPO)):
                config = _load(path)
                backend = config["backends"]["deepspeed"]
                algorithms.append(config["algorithm"]["name"])
                self.assertEqual(backend["world_size"], 8)
                self.assertEqual(config["training"]["gradient_clip"], 1.0)
                self.assertEqual(backend["deepspeed_config"]["gradient_clipping"], 1.0)
        self.assertEqual(len(algorithms), len(set(algorithms)))


if __name__ == "__main__":
    unittest.main()
