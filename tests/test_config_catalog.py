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


if __name__ == "__main__":
    unittest.main()
