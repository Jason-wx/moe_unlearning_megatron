import sys
import types
import unittest
from unittest.mock import patch

from megatron_unlearning.config import from_mapping
from megatron_unlearning.model_config import build_mcore_config_dict


class ModelConfigTest(unittest.TestCase):
    @staticmethod
    def _run_config(first=7, last=30, recompute=None):
        return from_mapping(
            {
                "model": {"path": "/model", "num_experts": 256},
                "data": {
                    "forget_cache": "/forget",
                    "retain_cache": "/retain",
                    "sequence_length": 1024,
                },
                "parallel": {
                    "world_size": 6,
                    "tensor_parallel": 2,
                    "pipeline_parallel": 3,
                    "expert_parallel": 2,
                    "expert_tensor_parallel": 1,
                    "sequence_parallel": True,
                    "activation_recompute": recompute,
                    "num_layers_in_first_pipeline_stage": first,
                    "num_layers_in_last_pipeline_stage": last,
                },
                "training": {"micro_batch_size": 16, "global_batch_size": 128},
                "algorithm": {"name": "rmu"},
            }
        )

    @staticmethod
    def _fake_modules():
        bridge = types.ModuleType("mcore_bridge")
        bridge.hf_to_mcore_config = lambda _: {
            "num_layers": 40,
            "recompute_granularity": "full",
        }

        transformers = types.ModuleType("transformers")

        class AutoConfig:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return object()

        transformers.AutoConfig = AutoConfig
        return {"mcore_bridge": bridge, "transformers": transformers}

    def test_uneven_pipeline_fields_reach_mcore_config(self):
        with patch.dict(sys.modules, self._fake_modules()):
            result = build_mcore_config_dict(self._run_config())
        self.assertEqual(result["pipeline_model_parallel_size"], 3)
        self.assertEqual(result["num_layers_in_first_pipeline_stage"], 7)
        self.assertEqual(result["num_layers_in_last_pipeline_stage"], 30)

    def test_invalid_uneven_pipeline_partition_is_rejected(self):
        with patch.dict(sys.modules, self._fake_modules()):
            with self.assertRaisesRegex(ValueError, "middle pipeline stages"):
                build_mcore_config_dict(self._run_config(first=14, last=26))

    def test_recompute_keys_are_stripped_by_default(self):
        with patch.dict(sys.modules, self._fake_modules()):
            result = build_mcore_config_dict(self._run_config())
        self.assertNotIn("recompute_granularity", result)
        self.assertNotIn("recompute_method", result)

    def test_full_activation_recompute_is_applied(self):
        run = self._run_config(recompute="full")
        with patch.dict(sys.modules, self._fake_modules()):
            result = build_mcore_config_dict(run)
        self.assertEqual(result["recompute_granularity"], "full")
        self.assertEqual(result["recompute_method"], "uniform")
        self.assertEqual(result["recompute_num_layers"], 1)

    def test_selective_activation_recompute_is_applied(self):
        run = self._run_config(recompute="selective")
        with patch.dict(sys.modules, self._fake_modules()):
            result = build_mcore_config_dict(run)
        self.assertEqual(result["recompute_granularity"], "selective")
        self.assertNotIn("recompute_method", result)

    def test_invalid_activation_recompute_is_rejected(self):
        run = self._run_config(recompute="block")
        with patch.dict(sys.modules, self._fake_modules()):
            with self.assertRaisesRegex(ValueError, "activation_recompute"):
                build_mcore_config_dict(run)


if __name__ == "__main__":
    unittest.main()
