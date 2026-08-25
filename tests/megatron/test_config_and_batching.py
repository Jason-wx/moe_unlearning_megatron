import tempfile
import unittest
from pathlib import Path

import torch

from megatron_unlearning.batching import collate_role_batch
from megatron_unlearning.config import algorithm_uses_retain, from_mapping, load_config
from megatron_unlearning.forward_step import (
    PDUState,
    reduce_reference_outputs,
    reduce_role_logits,
    reduce_role_losses,
)
from megatron_unlearning.rmu import (
    RMUHookState,
    RMUTargetLayerReached,
    find_local_transformer_layer,
    local_pipeline_position,
    local_sequence_mask,
)
from megatron_unlearning.runtime import (
    _freeze_reference_parameters,
    _set_trainable_parameters,
    aggregate_step_metrics,
    copy_hf_checkpoint_metadata,
    reference_window_ranges,
)
from open_unlearning.backends.megatron import to_megatron_mapping
from open_unlearning.configuration import load_experiment


class ConfigTest(unittest.TestCase):
    def test_hf_checkpoint_metadata_is_copied(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source = root / "source"
            output = root / "checkpoint"
            source.mkdir()
            (source / "config.json").write_text('{"model_type": "test"}', encoding="utf-8")
            (source / "generation_config.json").write_text('{"temperature": 1.0}', encoding="utf-8")

            copy_hf_checkpoint_metadata(source, output)

            self.assertEqual(
                (output / "config.json").read_text(encoding="utf-8"),
                '{"model_type": "test"}',
            )
            self.assertEqual(
                (output / "generation_config.json").read_text(encoding="utf-8"),
                '{"temperature": 1.0}',
            )

    def test_hf_checkpoint_metadata_requires_config(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source = root / "source"
            source.mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "source config"):
                copy_hf_checkpoint_metadata(source, root / "checkpoint")

    def _mapping(self):
        return {
            "model": {"path": "/model", "num_experts": 256},
            "data": {"forget_cache": "/forget", "sequence_length": 8},
            "parallel": {
                "world_size": 8,
                "tensor_parallel": 4,
                "pipeline_parallel": 2,
                "context_parallel": 1,
                "expert_parallel": 4,
                "expert_tensor_parallel": 1,
                "sequence_parallel": True,
            },
            "training": {"micro_batch_size": 1, "global_batch_size": 32},
            "algorithm": {"name": "grad_ascent"},
        }

    def test_valid_standard_eight_gpu_layout(self):
        config = from_mapping(self._mapping())
        layout = config.validate(check_paths=False)
        self.assertEqual(layout.tensor_parallel, 4)
        self.assertEqual(layout.pipeline_parallel, 2)
        self.assertEqual(layout.expert_parallel, 4)
        self.assertEqual(layout.data_parallel_size, 1)
        self.assertEqual(layout.expert_data_parallel_size, 1)

    def test_grad_diff_requires_retain(self):
        mapping = self._mapping()
        mapping["algorithm"]["name"] = "grad_diff"
        with self.assertRaisesRegex(ValueError, "retain_cache"):
            from_mapping(mapping).validate(check_paths=False)

    def test_invalid_global_batch_is_rejected(self):
        mapping = self._mapping()
        mapping["training"].update(micro_batch_size=3, global_batch_size=10)
        with self.assertRaisesRegex(ValueError, "global_batch_size"):
            from_mapping(mapping).validate(check_paths=False)

    def test_training_epochs_must_be_positive(self):
        mapping = self._mapping()
        mapping["training"]["num_train_epochs"] = 0
        with self.assertRaisesRegex(ValueError, "num_train_epochs"):
            from_mapping(mapping).validate(check_paths=False)

    def test_checkpoint_steps_must_be_positive_and_unique(self):
        mapping = self._mapping()
        mapping["training"]["save_steps"] = [300, 300]
        with self.assertRaisesRegex(ValueError, "duplicates"):
            from_mapping(mapping).validate(check_paths=False)

        mapping["training"]["save_steps"] = [0]
        with self.assertRaisesRegex(ValueError, "positive"):
            from_mapping(mapping).validate(check_paths=False)

    def test_megatron_wandb_is_forced_offline(self):
        config = from_mapping(self._mapping())
        self.assertEqual(config.training.wandb_mode, "offline")
        self.assertEqual(config.training.wandb_project, "open-unlearning-megatron")

        mapping = self._mapping()
        mapping["training"]["wandb_mode"] = "online"
        with self.assertRaisesRegex(ValueError, "offline mode"):
            from_mapping(mapping).validate(check_paths=False)

    def test_token_npo_train_eval_config(self):
        shared = load_experiment(
            "configs/experiments/interns2_pure_npo.yaml",
            components={"algorithm": "token_npo"},
        )
        config = from_mapping(to_megatron_mapping(shared))
        self.assertEqual(config.optimizer.learning_rate, 1.0e-6)
        self.assertEqual(config.optimizer.adam_beta1, 0.9)
        self.assertEqual(config.optimizer.adam_beta2, 0.999)
        self.assertEqual(config.optimizer.adam_eps, 1.0e-8)
        self.assertEqual(config.optimizer.gradient_clip, 0.0)
        self.assertEqual(config.training.micro_batch_size, 16)
        self.assertEqual(config.training.global_batch_size, 128)
        self.assertEqual(config.training.train_steps, 1344)
        self.assertEqual(config.training.num_train_epochs, 1)
        self.assertEqual(config.training.seed, 42)
        self.assertTrue(config.training.save_every_epoch)

    def test_full_logit_algorithm_accepts_standard_tp(self):
        mapping = self._mapping()
        mapping["algorithm"]["name"] = "ceu"
        layout = from_mapping(mapping).validate(check_paths=False)
        self.assertEqual(layout.tensor_parallel, 4)

    def test_token_npo_is_reference_based_and_does_not_require_retain(self):
        mapping = self._mapping()
        mapping["algorithm"] = {"name": "token_npo", "beta": 0.1}
        config = from_mapping(mapping)
        self.assertFalse(algorithm_uses_retain(config.algorithm))
        config.validate(check_paths=False)

    def test_reference_must_follow_student_layout_and_be_frozen(self):
        mapping = self._mapping()
        mapping["reference"] = {"allocation": "dedicated_rank"}
        with self.assertRaisesRegex(ValueError, "same TP/PP/CP/EP/ETP"):
            from_mapping(mapping).validate(check_paths=False)

        mapping = self._mapping()
        mapping["reference"] = {"frozen": False}
        with self.assertRaisesRegex(ValueError, "must be frozen"):
            from_mapping(mapping).validate(check_paths=False)

    def test_gradient_checkpointing_is_not_a_training_option(self):
        mapping = self._mapping()
        mapping["training"]["recompute"] = "uniform"
        with self.assertRaisesRegex(TypeError, "recompute"):
            from_mapping(mapping)

    def test_undial_only_requires_retain_when_alpha_is_nonzero(self):
        mapping = self._mapping()
        mapping["algorithm"] = {"name": "undial", "alpha": 0.0}
        config = from_mapping(mapping)
        self.assertFalse(algorithm_uses_retain(config.algorithm))
        config.validate(check_paths=False)

        mapping["algorithm"]["alpha"] = 0.1
        config = from_mapping(mapping)
        self.assertTrue(algorithm_uses_retain(config.algorithm))
        with self.assertRaisesRegex(ValueError, "retain_cache"):
            config.validate(check_paths=False)

    def test_rmu_config_requires_valid_target_and_retain_cache(self):
        mapping = self._mapping()
        mapping["data"]["retain_cache"] = "/retain"
        mapping["training"]["trainable_param_mode"] = "rmu_experts"
        mapping["algorithm"] = {
            "name": "rmu",
            "target_layer": 9,
            "train_layers": [7, 8, 9],
        }
        config = from_mapping(mapping)
        layout = config.validate(check_paths=False)
        self.assertEqual(layout.pipeline_parallel, 2)

        mapping["algorithm"]["train_layers"] = [7, 10]
        with self.assertRaisesRegex(ValueError, "after target_layer"):
            from_mapping(mapping).validate(check_paths=False)

    def test_yaml_inheritance_deep_merges_algorithm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "base.yaml").write_text(
                "model: {path: /model, num_experts: 8}\n"
                "data: {forget_cache: /forget, retain_cache: /retain}\n"
                "algorithm: {name: grad_diff, alpha: 1.0}\n",
                encoding="utf-8",
            )
            (root / "child.yaml").write_text(
                "extends: base.yaml\nalgorithm: {name: simnpo, alpha: 0.1}\n",
                encoding="utf-8",
            )
            config = load_config(root / "child.yaml")
            self.assertEqual(config.algorithm.name, "simnpo")
            self.assertEqual(config.algorithm.alpha, 0.1)
            self.assertEqual(config.data.retain_cache, "/retain")


class BatchingTest(unittest.TestCase):
    @staticmethod
    def _example(values):
        tensor = torch.tensor(values)
        labels = tensor.clone()
        labels[0] = -100
        return {"input_ids": tensor, "labels": labels}

    def test_grad_diff_uses_one_role_tagged_batch(self):
        batch = collate_role_batch(
            [self._example([1, 2, 3])],
            [self._example([4, 5])],
            pad_token_id=0,
            sequence_length=4,
        )
        self.assertEqual(batch["input_ids"].shape, (2, 4))
        self.assertEqual(batch["roles"].tolist(), [0, 1])
        self.assertEqual(batch["loss_mask"].sum(dim=1).tolist(), [2, 1])
        self.assertEqual(batch["activation_mask"].sum(dim=1).tolist(), [2, 1])
        self.assertEqual(batch["labels"].tolist(), [[2, 3, -100, -100], [5, -100, -100, -100]])

        token_losses = torch.tensor([[2.0, 4.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]])
        loss, metrics = reduce_role_losses(
            token_losses,
            batch["loss_mask"],
            batch["roles"],
            algorithm="grad_diff",
            gamma=1.0,
            alpha=0.5,
        )
        self.assertAlmostEqual(loss.item(), -1.5)
        self.assertAlmostEqual(metrics["forget_nll"].item(), 3.0)
        self.assertAlmostEqual(metrics["retain_loss"].item(), 3.0)

    def test_optimizer_step_metrics_aggregate_microbatches(self):
        metrics = aggregate_step_metrics(
            [
                {
                    "total_loss": torch.tensor(1.0),
                    "ceu_valid_tokens": torch.tensor(5),
                    "forget_valid_tokens": torch.tensor(7),
                    "retain_preference": torch.tensor(2.0),
                },
                {
                    "total_loss": torch.tensor(3.0),
                    "ceu_valid_tokens": torch.tensor(6),
                    "forget_valid_tokens": torch.tensor(5),
                    "retain_preference": torch.tensor(4.0),
                },
            ]
        )
        self.assertEqual(metrics["total_loss"].item(), 2.0)
        self.assertEqual(metrics["ceu_valid_tokens"].item(), 11.0)
        self.assertEqual(metrics["forget_valid_tokens"].item(), 12.0)
        self.assertEqual(metrics["retain_preference"].item(), 4.0)


class TrainableParameterTest(unittest.TestCase):
    class FakeModel:
        def __init__(self):
            names = (
                "decoder.layers.0.mlp.router.weight",
                "decoder.layers.0.mlp.experts.linear_fc1.weight",
                "decoder.layers.0.mlp.shared_experts.linear_fc1.weight",
                "decoder.layers.1.mlp.linear_fc1.weight",
                "decoder.layers.1.self_attention.linear_qkv.weight",
            )
            self.parameters = [(name, torch.nn.Parameter(torch.ones(1))) for name in names]

        def named_parameters(self):
            return iter(self.parameters)

    def test_qwen_trains_routed_and_shared_experts_only(self):
        model = self.FakeModel()
        _set_trainable_parameters([model], "moe_experts", model_type="qwen3_5_moe")
        trainable = {name for name, value in model.parameters if value.requires_grad}
        self.assertEqual(
            trainable,
            {
                "decoder.layers.0.mlp.experts.linear_fc1.weight",
                "decoder.layers.0.mlp.shared_experts.linear_fc1.weight",
            },
        )

    def test_gemma_also_trains_dense_mlp_blocks(self):
        model = self.FakeModel()
        _set_trainable_parameters([model], "moe_experts", model_type="gemma4")
        trainable = {name for name, value in model.parameters if value.requires_grad}
        self.assertIn("decoder.layers.1.mlp.linear_fc1.weight", trainable)
        self.assertNotIn("decoder.layers.0.mlp.router.weight", trainable)


class StatefulObjectiveTest(unittest.TestCase):
    def test_pdu_updates_nonnegative_dual_preference(self):
        logits = torch.tensor(
            [
                [[2.0, 0.0], [1.0, 1.0]],
                [[0.0, 2.0], [2.0, 0.0]],
            ],
            requires_grad=True,
        )
        labels = torch.tensor([[1, -100], [1, 0]])
        mask = labels.ne(-100)
        roles = torch.tensor([0, 1])
        config = from_mapping(
            {
                "model": {"path": "/model", "num_experts": 2},
                "data": {"forget_cache": "/forget", "retain_cache": "/retain"},
                "algorithm": {
                    "name": "pdu",
                    "alpha": 1.0,
                    "retain_loss_eps": 0.0,
                    "dual_step_size": 2.0,
                },
            }
        ).algorithm
        state = PDUState(retain_preference=1.0)
        loss, metrics = reduce_role_logits(
            logits, labels, mask, roles, config=config, pdu_state=state
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(state.retain_preference, 1.0)
        self.assertAlmostEqual(metrics["retain_preference"].item(), 1.0)


class ReferenceAlgorithmTest(unittest.TestCase):
    def test_reference_parameters_are_frozen_and_eval_only(self):
        model = torch.nn.Linear(2, 2)
        _freeze_reference_parameters([model])
        self.assertFalse(model.training)
        self.assertTrue(all(not value.requires_grad for value in model.parameters()))

    def test_npo_reduction_uses_reference_token_losses(self):
        current = torch.tensor([[2.0, 4.0]], requires_grad=True)
        reference = torch.tensor([[1.0, 1.0]])
        batch = {
            "roles": torch.tensor([0]),
            "loss_mask": torch.tensor([[True, True]]),
            "labels": torch.tensor([[1, 2]]),
        }
        config = from_mapping(
            {
                "model": {"path": "/model", "num_experts": 4},
                "data": {"forget_cache": "/forget"},
                "algorithm": {"name": "npo", "beta": 0.1, "gamma": 2.0},
            }
        ).algorithm
        loss, metrics = reduce_reference_outputs(current, reference, batch, config=config)
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(metrics["current_sequence_nll"].item(), 3.0)
        loss.backward()
        self.assertTrue(torch.all(current.grad < 0))

    def test_token_npo_reduction_keeps_per_token_reference_differences(self):
        current = torch.tensor([[2.0, 10.0]], requires_grad=True)
        reference = torch.tensor([[1.0, 1.0]])
        batch = {
            "roles": torch.tensor([0]),
            "loss_mask": torch.tensor([[True, True]]),
            "labels": torch.tensor([[1, 2]]),
        }
        config = from_mapping(
            {
                "model": {"path": "/model", "num_experts": 4},
                "data": {"forget_cache": "/forget"},
                "algorithm": {"name": "token_npo", "beta": 0.1},
            }
        ).algorithm
        loss, metrics = reduce_reference_outputs(current, reference, batch, config=config)
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(metrics["token_npo_logit"].item(), 0.5)
        self.assertIn("token_gradient_weight", metrics)
        loss.backward()
        self.assertTrue(torch.all(current.grad < 0))

    def test_reference_windows_bound_undial_logits_only(self):
        self.assertEqual(reference_window_ranges("npo", 8, 2), [(0, 8)])
        self.assertEqual(reference_window_ranges("token_npo", 8, 2), [(0, 8)])
        self.assertEqual(
            reference_window_ranges("undial", 5, 2),
            [(0, 2), (2, 4), (4, 5)],
        )


class RMUHookTest(unittest.TestCase):
    class FakeLayer(torch.nn.Module):
        def __init__(self, layer_number):
            super().__init__()
            self.layer_number = layer_number
            self.mlp = torch.nn.Module()
            self.mlp.experts = torch.nn.Linear(2, 2)
            self.self_attention = torch.nn.Linear(2, 2)

    class FakeMCoreModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = torch.nn.Module()
            self.decoder.layers = torch.nn.ModuleList(
                [RMUHookTest.FakeLayer(number) for number in (8, 9, 10, 11)]
            )

    class WrappedMCoreModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = RMUHookTest.FakeMCoreModel()

    def test_global_target_layer_and_train_layers(self):
        model = self.FakeMCoreModel()
        target = find_local_transformer_layer([model], target_layer=9)
        self.assertEqual(target.layer_number, 10)
        self.assertEqual(local_pipeline_position([model], target_layer=9), "target")
        _set_trainable_parameters(
            [model],
            "rmu_experts",
            model_type="qwen3_5_moe",
            train_layers=[7, 8, 9],
        )
        trainable = {name for name, value in model.named_parameters() if value.requires_grad}
        self.assertTrue(trainable)
        self.assertTrue(all("mlp.experts" in name for name in trainable))
        self.assertTrue(all("decoder.layers.3" not in name for name in trainable))

    def test_wrapped_model_finds_nested_decoder(self):
        model = self.WrappedMCoreModel()
        target = find_local_transformer_layer([model], target_layer=9)
        self.assertEqual(target.layer_number, 10)
        self.assertEqual(local_pipeline_position([model], target_layer=9), "target")
        _set_trainable_parameters(
            [model],
            "rmu_experts",
            model_type="qwen3_5_moe",
            train_layers=[9],
        )
        trainable = {name for name, value in model.named_parameters() if value.requires_grad}
        self.assertTrue(trainable)
        self.assertTrue(
            all("language_model.decoder.layers.2.mlp.experts" in name for name in trainable)
        )

    def test_local_mask_uses_sequence_first_activation(self):
        mask = torch.tensor([[True, False, True], [False, True, True]])
        activation = torch.zeros(3, 2, 4)
        local = local_sequence_mask(mask, activation, sequence_parallel=True)
        self.assertTrue(torch.equal(local, mask.transpose(0, 1)))

    def test_hook_injects_rmu_gradient_without_changing_forward_value(self):
        algorithm = from_mapping(
            {
                "model": {"path": "/model", "num_experts": 4},
                "data": {"forget_cache": "/forget", "retain_cache": "/retain"},
                "algorithm": {
                    "name": "rmu",
                    "gamma": 1.0,
                    "alpha": 2.0,
                    "steering_coeff": 3.0,
                    "control_norm_mode": "fixed",
                },
            }
        ).algorithm
        state = RMUHookState(
            algorithm=algorithm,
            sequence_parallel=False,
            seed=42,
            loss_scale=0.25,
        )
        reference = torch.zeros(3, 2, 4)
        with self.assertRaises(RMUTargetLayerReached):
            state.reference_hook(None, None, reference)
        state.begin_student_batch(
            {
                "roles": torch.tensor([0, 1]),
                "loss_mask": torch.tensor([[True, True, False], [True, False, False]]),
                "activation_mask": torch.tensor([[False, True, True], [False, True, False]]),
            }
        )
        activation = torch.ones(3, 2, 4, requires_grad=True)
        try:
            state.student_hook(None, None, activation)
        except RMUTargetLayerReached as reached:
            replaced = reached.activation
        else:
            self.fail("RMU student hook did not stop at the target layer")
        self.assertTrue(torch.equal(replaced, activation))
        self.assertIsNone(replaced._base)
        (replaced.sum() * 0.0).backward()
        self.assertGreater(activation.grad.abs().sum().item(), 0.0)
        self.assertFalse(state.reference_activations)
        self.assertIn("rmu_loss", state.latest_metrics)


if __name__ == "__main__":
    unittest.main()
