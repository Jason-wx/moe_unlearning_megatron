from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from open_unlearning.trainable_parameters import (
    ATTENTION,
    EXPERT,
    FULL,
    GEMMA4_FAMILY,
    QWEN35_FAMILY,
    ROUTER,
    classify_hf_parameter,
    classify_mcore_layer_parameter,
    configure_hf_trainable_parameters,
    model_family,
    trainable_categories,
)


class QwenHFModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3_5_moe")
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        layer.self_attn.q_proj = torch.nn.Linear(1, 1, bias=False)
        layer.self_attn.q_norm = torch.nn.LayerNorm(1)
        layer.linear_attn = torch.nn.Module()
        layer.linear_attn.in_proj_qkvz = torch.nn.Linear(1, 1, bias=False)
        layer.linear_attn.norm = torch.nn.LayerNorm(1)
        layer.input_layernorm = torch.nn.LayerNorm(1)
        layer.post_attention_layernorm = torch.nn.LayerNorm(1)
        layer.mlp = torch.nn.Module()
        layer.mlp.experts = torch.nn.Linear(1, 1, bias=False)
        layer.mlp.shared_expert = torch.nn.Linear(1, 1, bias=False)
        layer.mlp.gate = torch.nn.Linear(1, 1, bias=False)
        layer.mlp.shared_expert_gate = torch.nn.Linear(1, 1, bias=False)
        self.model.language_model.layers = torch.nn.ModuleList([layer])
        self.model.language_model.norm = torch.nn.LayerNorm(1)
        self.model.visual = torch.nn.Linear(1, 1, bias=False)
        self.lm_head = torch.nn.Linear(1, 1, bias=False)


class TrainableParameterClassificationTest(unittest.TestCase):
    def test_supported_model_families_are_explicit(self):
        self.assertEqual(model_family("qwen3_5_moe"), QWEN35_FAMILY)
        self.assertEqual(model_family("qwen3_5_moe", "InternS2Model"), QWEN35_FAMILY)
        self.assertEqual(model_family("gemma4"), GEMMA4_FAMILY)
        with self.assertRaisesRegex(ValueError, "supports only"):
            model_family("llama")

    def test_full_is_exact_union_of_three_categories(self):
        self.assertEqual(trainable_categories(FULL), {ATTENTION, EXPERT, ROUTER})
        with self.assertRaisesRegex(ValueError, "unsupported trainable_param_mode"):
            trainable_categories("all")

    def test_qwen_hf_categories_exclude_block_norms_and_mtp(self):
        prefix = "model.language_model.layers.3."
        cases = {
            "self_attn.q_proj.weight": ATTENTION,
            "self_attn.q_norm.weight": ATTENTION,
            "linear_attn.in_proj_qkvz.weight": ATTENTION,
            "linear_attn.norm.weight": ATTENTION,
            "input_layernorm.weight": None,
            "post_attention_layernorm.weight": None,
            "mlp.experts.gate_up_proj": EXPERT,
            "mlp.shared_expert.down_proj.weight": EXPERT,
            "mlp.gate.weight": ROUTER,
            "mlp.shared_expert_gate.weight": ROUTER,
        }
        for relative, expected in cases.items():
            with self.subTest(parameter=relative):
                self.assertEqual(
                    classify_hf_parameter(prefix + relative, family=QWEN35_FAMILY),
                    expected,
                )
        self.assertIsNone(
            classify_hf_parameter(
                "mtp.layers.0.mlp.experts.gate_up_proj", family=QWEN35_FAMILY
            )
        )

    def test_gemma_hf_mlp_is_shared_expert_and_router_scales_are_router(self):
        prefix = "model.language_model.layers.5."
        cases = {
            "self_attn.q_proj.weight": ATTENTION,
            "self_attn.k_norm.weight": ATTENTION,
            "post_attention_layernorm.weight": None,
            "pre_feedforward_layernorm_2.weight": None,
            "mlp.gate_proj.weight": EXPERT,
            "experts.gate_up_proj": EXPERT,
            "router.proj.weight": ROUTER,
            "router.scale": ROUTER,
            "router.per_expert_scale": ROUTER,
        }
        for relative, expected in cases.items():
            with self.subTest(parameter=relative):
                self.assertEqual(
                    classify_hf_parameter(prefix + relative, family=GEMMA4_FAMILY),
                    expected,
                )

    def test_mcore_fused_block_norms_remain_frozen(self):
        self.assertIsNone(
            classify_mcore_layer_parameter(
                "self_attention.linear_qkv.layer_norm_weight", family=QWEN35_FAMILY
            )
        )
        self.assertIsNone(
            classify_mcore_layer_parameter(
                "self_attention.in_proj.layer_norm_weight", family=QWEN35_FAMILY
            )
        )
        self.assertEqual(
            classify_mcore_layer_parameter(
                "self_attention.q_layernorm.weight", family=QWEN35_FAMILY
            ),
            ATTENTION,
        )
        self.assertIsNone(
            classify_mcore_layer_parameter(
                "mlp.linear_fc1.layer_norm_weight", family=GEMMA4_FAMILY
            )
        )

    def test_qwen_hf_attention_mode_changes_only_attention_internals(self):
        model = QwenHFModel()
        configure_hf_trainable_parameters(model, ATTENTION)
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        self.assertTrue(any("self_attn.q_proj" in name for name in trainable))
        self.assertTrue(any("self_attn.q_norm" in name for name in trainable))
        self.assertTrue(any("linear_attn.norm" in name for name in trainable))
        self.assertFalse(any("input_layernorm" in name for name in trainable))
        self.assertFalse(any("post_attention_layernorm" in name for name in trainable))
        self.assertFalse(any("mlp." in name for name in trainable))
        self.assertFalse(any("visual" in name or "lm_head" in name for name in trainable))

    def test_qwen_hf_full_mode_is_union_but_still_excludes_block_norms(self):
        model = QwenHFModel()
        configure_hf_trainable_parameters(model, FULL)
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        self.assertTrue(any("self_attn" in name for name in trainable))
        self.assertTrue(any("mlp.experts" in name for name in trainable))
        self.assertTrue(any("mlp.gate" in name for name in trainable))
        self.assertFalse(any("input_layernorm" in name for name in trainable))
        self.assertFalse(any("post_attention_layernorm" in name for name in trainable))
        self.assertFalse(any("visual" in name or "lm_head" in name for name in trainable))


if __name__ == "__main__":
    unittest.main()
