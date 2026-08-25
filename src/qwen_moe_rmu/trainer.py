import copy
import logging
import re
from typing import Iterable

import torch
from accelerate.utils import is_deepspeed_available
from trainer.unlearn.base import UnlearnTrainer
from trainer.utils import compute_kl_divergence

if is_deepspeed_available():
    import deepspeed
else:
    deepspeed = None

logger = logging.getLogger(__name__)


class QwenMoeRMUTrainer(UnlearnTrainer):
    """RMU trainer variant for Qwen MoE models.

    This class is a standalone adaptation instead of a registry replacement for
    the default RMU trainer. It keeps non-selected parameters frozen after
    optimizer creation to avoid unnecessary gradients on large MoE checkpoints.
    """

    def __init__(
        self,
        module_regex: str = r"model\.language_model\.layers\.9",
        trainable_params_regex: Iterable[str] = (
            r"model\.language_model\.layers\.(7|8|9)\.mlp\.experts\.(gate_up_proj|down_proj)",
            r"model\.language_model\.layers\.(7|8|9)\.mlp\.shared_expert\.(gate_proj|up_proj|down_proj)\.weight",
        ),
        gamma: float = 1.0,
        alpha: float = 1200.0,
        steering_coeff: float = 6.5,
        control_norm_mode: str = "ref_activation_l2",
        control_scale_coeff: float = 1.0,
        retain_loss_type: str = "EMBED_DIFF",
        log_trainable_params: bool = True,
        ref_model=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.module_regex = module_regex
        self.trainable_params_regex = list(trainable_params_regex)
        self.gamma = gamma
        self.alpha = alpha
        self.steering_coeff = steering_coeff
        self.control_norm_mode = control_norm_mode
        self.control_scale_coeff = control_scale_coeff
        self.retain_loss_type = retain_loss_type
        self.log_trainable_params = log_trainable_params
        self.control_vec = None
        if ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)
        else:
            self.ref_model = self._prepare_preloaded_ref_model(ref_model)

        # Validate early against the unwrapped model when possible. DeepSpeed can
        # still wrap later; compute_loss resolves modules again from the active model.
        self._get_matching_module(self.model, self.module_regex)
        self._get_matching_module(self.ref_model, self.module_regex)

    def _prepare_ref_model(self, model):
        ref_model = copy.deepcopy(model).to(self.accelerator.device)
        return self._prepare_preloaded_ref_model(ref_model)

    def _prepare_preloaded_ref_model(self, ref_model):
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        if self.is_deepspeed_enabled:
            ref_model = self._prepare_deepspeed(ref_model)
        else:
            ref_model = self.accelerator.prepare_model(ref_model, evaluation_mode=True)
        return ref_model

    def _unwrap_model(self, model):
        if deepspeed is not None and isinstance(model, deepspeed.DeepSpeedEngine):
            return model.module
        return model

    def _get_matching_module(self, model, module_regex):
        model = self._unwrap_model(model)
        matched = {
            name: module
            for name, module in model.named_modules()
            if re.fullmatch(module_regex, name)
        }
        if len(matched) > 1:
            raise ValueError(
                f"More than one module matched {module_regex}: {list(matched)}"
            )
        if not matched:
            sample = [name for name, _ in list(model.named_modules())[:30]]
            raise ValueError(
                f"No module matched {module_regex}. First module names: {sample}"
            )
        return next(iter(matched.values()))

    def _set_requires_grad_all(self, model, requires_grad: bool):
        for param in model.parameters():
            param.requires_grad = requires_grad

    def _set_trainable_params(self, model):
        matched = []
        for name, param in self._unwrap_model(model).named_parameters():
            if any(re.fullmatch(pattern, name) for pattern in self.trainable_params_regex):
                param.requires_grad = True
                matched.append(name)
            else:
                param.requires_grad = False
        if not matched:
            raise ValueError(
                "No trainable parameters matched trainable_params_regex: "
                f"{self.trainable_params_regex}"
            )
        if self.log_trainable_params and self.accelerator.is_local_main_process:
            logger.info("Matched %d trainable parameter tensors.", len(matched))
            for name in matched[:80]:
                logger.info("trainable: %s", name)
            if len(matched) > 80:
                logger.info("... %d more trainable tensors omitted", len(matched) - 80)
        return matched

    def create_optimizer(self):
        self._set_requires_grad_all(self.model, False)
        self._set_trainable_params(self.model)
        return super().create_optimizer()

    def forward_with_cache(self, model, inputs, module_regex, no_grad=True):
        class StopForward(Exception):
            pass

        cache = []
        module = self._get_matching_module(model, module_regex)
        forward_inputs = {
            key: value for key, value in inputs.items() if key != "labels"
        }
        forward_inputs.setdefault("logits_to_keep", 1)

        def hook(_module, _input, output):
            cache.append(output[0] if isinstance(output, tuple) else output)
            raise StopForward

        handle = module.register_forward_hook(hook)
        outputs = None
        try:
            with torch.set_grad_enabled(not no_grad):
                outputs = model(**forward_inputs)
        except StopForward:
            pass
        finally:
            handle.remove()
        if not cache:
            raise RuntimeError(f"Forward hook did not capture output for {module_regex}")
        return cache[0], outputs

    def get_fixed_control_vector(self, dim, dtype, device):
        if self.control_vec is None or self.control_vec.shape[-1] != dim:
            random_vector = torch.rand(1, 1, dim, dtype=dtype, device=device)
            self.control_vec = (
                random_vector / torch.norm(random_vector) * self.steering_coeff
            )
        return self.control_vec.to(dtype=dtype, device=device)

    def get_unit_control_vector(self, activation):
        random_vector = torch.rand(
            1,
            1,
            activation.shape[-1],
            dtype=activation.dtype,
            device=activation.device,
        )
        return random_vector / random_vector.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def get_control_target(self, forget_inputs, model_acts):
        if self.control_norm_mode == "fixed":
            return self.get_fixed_control_vector(
                model_acts.shape[-1], dtype=model_acts.dtype, device=model_acts.device
            ).expand_as(model_acts)

        unit_control = self.get_unit_control_vector(model_acts).expand_as(model_acts)

        if self.control_norm_mode == "ref_activation_l2":
            ref_acts, _ = self.forward_with_cache(
                self.ref_model, forget_inputs, self.module_regex, no_grad=True
            )
            target_norm = ref_acts.to(model_acts.device).norm(dim=-1, keepdim=True)
        elif self.control_norm_mode == "batch_activation_l2":
            target_norm = model_acts.detach().norm(dim=-1, keepdim=True)
        else:
            raise ValueError(
                "Unsupported control_norm_mode="
                f"{self.control_norm_mode}. Expected fixed, ref_activation_l2, "
                "or batch_activation_l2."
            )

        return unit_control * target_norm * self.control_scale_coeff

    def compute_activation_loss(self, activation1, activation2, mask):
        squared_diff = torch.nn.functional.mse_loss(
            activation1, activation2, reduction="none"
        )
        expanded_mask = mask.unsqueeze(-1).expand_as(squared_diff)
        squared_diff_sum = (squared_diff * expanded_mask).mean(dim=2).sum(dim=1)
        num_tokens = mask.sum(dim=-1).clamp_min(1)
        return (squared_diff_sum / num_tokens).mean()

    def _trim_inputs(self, inputs):
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": inputs["labels"],
        }

    def compute_retain_loss(self, model, retain_inputs):
        if self.retain_loss_type == "EMBED_DIFF":
            model_acts, _ = self.forward_with_cache(
                model, retain_inputs, self.module_regex, no_grad=False
            )
            ref_acts, _ = self.forward_with_cache(
                self.ref_model, retain_inputs, self.module_regex, no_grad=True
            )
            mask = retain_inputs["labels"] != -100
            return self.compute_activation_loss(
                model_acts, ref_acts.to(model_acts.device), mask
            )
        if self.retain_loss_type == "NLL":
            return model(**retain_inputs).loss
        if self.retain_loss_type == "KL":
            kl_loss, _ = compute_kl_divergence(model, self.ref_model, retain_inputs)
            return kl_loss
        raise NotImplementedError(f"Unsupported retain_loss_type: {self.retain_loss_type}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        forget_inputs = self._trim_inputs(inputs["forget"])
        model_acts, forget_outputs = self.forward_with_cache(
            model, forget_inputs, self.module_regex, no_grad=False
        )

        control_vec = forget_inputs.get("control_vec", None)
        if control_vec is None:
            control_vec = self.get_control_target(forget_inputs, model_acts)
        else:
            control_vec = control_vec.to(dtype=model_acts.dtype, device=model_acts.device)
            control_vec = control_vec.expand_as(model_acts)
        forget_mask = forget_inputs["labels"] != -100
        forget_loss = self.compute_activation_loss(model_acts, control_vec, forget_mask)

        retain_inputs = self._trim_inputs(inputs["retain"])
        retain_loss = self.compute_retain_loss(model, retain_inputs)
        weighted_forget_loss = self.gamma * forget_loss
        weighted_retain_loss = self.alpha * retain_loss
        loss = weighted_forget_loss + weighted_retain_loss
        if self.state.global_step % max(self.args.logging_steps, 1) == 0:
            self.log(
                {
                    "train/forget_loss": forget_loss.detach().float().item(),
                    "train/retain_loss": retain_loss.detach().float().item(),
                    "train/weighted_forget_loss": weighted_forget_loss.detach()
                    .float()
                    .item(),
                    "train/weighted_retain_loss": weighted_retain_loss.detach()
                    .float()
                    .item(),
                    "train/rmu_loss": loss.detach().float().item(),
                }
            )
        return (loss, forget_outputs) if return_outputs else loss
