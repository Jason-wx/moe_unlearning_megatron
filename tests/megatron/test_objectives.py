import sys
import unittest
from pathlib import Path

import torch

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from megatron_unlearning.objectives import (
    ceu_objective,
    grad_ascent_objective,
    grad_diff_objective,
    masked_mean,
    npo_objective,
    parallel_rmu_mse,
    pdu_forget_objective,
    pure_npo_objective,
    rmu_mse_statistics,
    satimp_objective,
    simnpo_objective,
    token_npo_objective,
    token_simnpo_objective,
    undial_objective,
    wga_objective,
)
from megatron_unlearning.parallel import ParallelLayout


class ObjectiveTests(unittest.TestCase):
    def test_masked_mean_ignores_padding(self):
        values = torch.tensor([[1.0, 2.0, 99.0]])
        mask = torch.tensor([[True, True, False]])
        self.assertAlmostEqual(masked_mean(values, mask).item(), 1.5)

    def test_grad_ascent_has_negative_nll_gradient(self):
        nll = torch.tensor([[1.0, 2.0]], requires_grad=True)
        loss = grad_ascent_objective(nll, torch.ones_like(nll, dtype=torch.bool))
        loss.backward()
        self.assertTrue(torch.all(nll.grad < 0))

    def test_grad_diff_combines_expected_components(self):
        forget = torch.tensor([[2.0, 4.0]])
        retain = torch.tensor([[1.0, 3.0]])
        mask = torch.ones_like(forget, dtype=torch.bool)
        loss, metrics = grad_diff_objective(forget, mask, retain, mask, gamma=2.0, alpha=3.0)
        self.assertAlmostEqual(metrics["forget_loss"].item(), -3.0)
        self.assertAlmostEqual(metrics["retain_loss"].item(), 2.0)
        self.assertAlmostEqual(loss.item(), 0.0)

    def test_simnpo_saturates_as_forget_nll_increases(self):
        mask = torch.ones(1, 2, dtype=torch.bool)
        low, _ = simnpo_objective(torch.zeros(1, 2), mask, beta=0.1)
        high, _ = simnpo_objective(torch.full((1, 2), 20.0), mask, beta=0.1)
        self.assertLess(high.item(), low.item())

    def test_simnpo_reduces_sequence_before_logistic_transform(self):
        nll = torch.tensor([[0.0, 10.0]])
        mask = torch.ones_like(nll, dtype=torch.bool)
        sequence_loss, _ = simnpo_objective(nll, mask, beta=0.1)
        token_loss, _ = token_simnpo_objective(nll, mask, beta=0.1)
        expected = -20.0 * torch.nn.functional.logsigmoid(torch.tensor(0.5))
        self.assertAlmostEqual(sequence_loss.item(), expected.item())
        self.assertNotAlmostEqual(sequence_loss.item(), token_loss.item())

    def test_wga_uses_detached_probability_weight(self):
        nll = torch.tensor([[1.0, 2.0]], requires_grad=True)
        mask = torch.ones_like(nll, dtype=torch.bool)
        loss, metrics = wga_objective(nll, mask, beta=1.0)
        expected = -(torch.exp(-nll.detach()) * nll).mean()
        self.assertTrue(torch.allclose(loss, expected))
        self.assertGreater(metrics["forget_weight"], 0)

    def test_satimp_weight_is_finite(self):
        nll = torch.tensor([[0.2, 2.0]], requires_grad=True)
        mask = torch.ones_like(nll, dtype=torch.bool)
        loss, metrics = satimp_objective(nll, mask, beta1=5.0, beta2=1.0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(nll.grad).all())
        self.assertGreaterEqual(metrics["forget_weight"], 0)
        self.assertAlmostEqual(metrics["token_probability_mean"].item(), torch.exp(-nll.detach()).mean().item())
        self.assertLessEqual(metrics["token_probability_p10"], metrics["token_probability_p50"])
        self.assertLessEqual(metrics["token_probability_p50"], metrics["token_probability_p90"])
        self.assertLessEqual(metrics["forget_weight_p10"], metrics["forget_weight_p50"])
        self.assertLessEqual(metrics["forget_weight_p50"], metrics["forget_weight_p90"])
        self.assertGreaterEqual(metrics["forget_weight_effective_token_fraction"], 0.0)
        self.assertLessEqual(metrics["forget_weight_effective_token_fraction"], 1.0)

    def test_ceu_pushes_down_the_correct_logit(self):
        logits = torch.tensor([[[2.0, 1.0, 0.0], [0.5, 1.5, 0.0]]], requires_grad=True)
        labels = torch.tensor([[0, 1]])
        loss, _ = ceu_objective(logits, labels, ignore_first_n_tokens=0)
        loss.backward()
        self.assertGreater(logits.grad[0, 0, 0], 0)
        self.assertGreater(logits.grad[0, 1, 1], 0)

    def test_pdu_logit_dispersion(self):
        logits = torch.tensor([[[1.0, 3.0], [2.0, 2.0]]])
        mask = torch.tensor([[True, False]])
        self.assertAlmostEqual(pdu_forget_objective(logits, mask).item(), 1.0)

    def test_pure_npo_uses_length_normalized_sequence_nll(self):
        current = torch.tensor([[2.0, 2.0, 50.0], [4.0, 4.0, 4.0]])
        reference = torch.ones_like(current)
        mask = torch.tensor([[True, True, False], [True, True, True]])
        loss, metrics = pure_npo_objective(current, reference, mask, beta=0.1)
        self.assertAlmostEqual(metrics["current_sequence_nll"].item(), 3.0)
        self.assertAlmostEqual(metrics["reference_sequence_nll"].item(), 1.0)
        self.assertTrue(torch.isfinite(loss))

    def test_original_npo_uses_summed_sequence_nll(self):
        current = torch.tensor([[2.0, 2.0, 50.0], [4.0, 4.0, 4.0]])
        reference = torch.ones_like(current)
        mask = torch.tensor([[True, True, False], [True, True, True]])
        loss, metrics = npo_objective(current, reference, mask, beta=0.1)
        self.assertAlmostEqual(metrics["current_sequence_nll"].item(), 8.0)
        self.assertAlmostEqual(metrics["reference_sequence_nll"].item(), 2.5)
        self.assertTrue(torch.isfinite(loss))

    def test_pure_npo_skips_individually_empty_sequences(self):
        current = torch.tensor([[2.0, 2.0], [99.0, 99.0]])
        reference = torch.tensor([[1.0, 1.0], [88.0, 88.0]])
        mask = torch.tensor([[True, True], [False, False]])
        loss, metrics = pure_npo_objective(current, reference, mask, beta=0.1)
        expected = -20.0 * torch.nn.functional.logsigmoid(torch.tensor(0.1))
        self.assertAlmostEqual(loss.item(), expected.item())
        self.assertEqual(metrics["valid_sequences"].item(), 1)
        self.assertEqual(metrics["skipped_empty_sequences"].item(), 1)
        self.assertAlmostEqual(metrics["current_sequence_nll"].item(), 2.0)

    def test_pure_npo_rejects_an_entirely_empty_batch(self):
        values = torch.ones(2, 3)
        mask = torch.zeros_like(values, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "no valid sequences"):
            pure_npo_objective(values, values, mask, beta=0.1)

    def test_token_npo_applies_sigmoid_before_masked_mean(self):
        current = torch.tensor([[2.0, 10.0, 99.0]], requires_grad=True)
        reference = torch.ones_like(current)
        mask = torch.tensor([[True, True, False]])
        loss, metrics = token_npo_objective(current, reference, mask, beta=0.1)
        expected = (-20.0 * torch.nn.functional.logsigmoid(torch.tensor([0.1, 0.9]))).mean()
        sequence_loss, _ = pure_npo_objective(current, reference, mask, beta=0.1)
        self.assertAlmostEqual(loss.item(), expected.item())
        self.assertNotAlmostEqual(loss.item(), sequence_loss.item())
        self.assertAlmostEqual(metrics["current_token_nll"].item(), 6.0)
        self.assertAlmostEqual(metrics["reference_token_nll"].item(), 1.0)
        self.assertAlmostEqual(metrics["token_npo_logit"].item(), 0.5)
        loss.backward()
        self.assertTrue(torch.all(current.grad[mask] < 0))
        self.assertEqual(current.grad[~mask].abs().sum().item(), 0.0)

    def test_undial_matches_full_vocab_soft_target_cross_entropy(self):
        student = torch.tensor([[[1.0, 0.0, -1.0], [0.5, 1.0, -0.5]]], requires_grad=True)
        teacher = torch.tensor([[[2.0, 0.0, 1.0], [0.0, 2.0, 1.0]]])
        labels = torch.tensor([[0, 1]])
        mask = torch.tensor([[True, False]])
        loss, metrics = undial_objective(student, teacher, labels, mask, beta=2.0)

        adjusted_teacher = teacher.clone()
        adjusted_teacher[0, 0, 0] -= 2.0
        expected = (
            -(adjusted_teacher.softmax(dim=-1) * student.log_softmax(dim=-1))
            .sum(dim=-1)[mask]
            .mean()
        )
        self.assertTrue(torch.allclose(loss, expected))
        self.assertEqual(metrics["undial_valid_tokens"].item(), 1)
        loss.backward()
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_rmu_statistics_are_additive(self):
        activation = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        target = torch.zeros_like(activation)
        numerator, count = rmu_mse_statistics(activation, target, torch.tensor([[True, False]]))
        self.assertAlmostEqual(numerator.item(), 5.0)
        self.assertAlmostEqual(count.item(), 2.0)

    def test_parallel_rmu_mse_matches_full_activation_mean(self):
        activation = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]], requires_grad=True)
        target = torch.zeros_like(activation)
        mask = torch.tensor([[True], [False]])
        loss, metrics = parallel_rmu_mse(activation, target, mask)
        self.assertAlmostEqual(loss.item(), 2.5)
        self.assertEqual(metrics["rmu_element_count"].item(), 2)
        loss.backward()
        self.assertTrue(torch.allclose(activation.grad[1], torch.zeros_like(activation.grad[1])))


class ParallelLayoutTests(unittest.TestCase):
    def test_valid_eight_gpu_layout(self):
        layout = ParallelLayout(
            world_size=8,
            tensor_parallel=4,
            pipeline_parallel=2,
            context_parallel=1,
            expert_parallel=4,
            expert_tensor_parallel=1,
            sequence_parallel=True,
            num_experts=256,
        ).validate()
        self.assertEqual(layout.data_parallel_size, 1)
        self.assertEqual(layout.expert_data_parallel_size, 1)

    def test_invalid_dense_parallel_product(self):
        with self.assertRaises(ValueError):
            ParallelLayout(world_size=8, tensor_parallel=3).validate()

    def test_sequence_parallel_requires_tp(self):
        with self.assertRaises(ValueError):
            ParallelLayout(
                world_size=8,
                tensor_parallel=1,
                pipeline_parallel=1,
                expert_parallel=1,
                sequence_parallel=True,
            ).validate()


if __name__ == "__main__":
    unittest.main()
