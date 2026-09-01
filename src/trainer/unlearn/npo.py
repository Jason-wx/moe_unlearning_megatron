"""Negative Preference Optimization variants."""

import torch
import torch.nn.functional as F

from trainer.unlearn.grad_diff import GradDiff
from trainer.utils import compute_dpo_loss, compute_token_nll


class _SequenceNPO(GradDiff):
    """Shared reference handling for sequence-level NPO variants."""

    normalize_by_length = False

    def __init__(self, beta=1.0, gamma=1.0, *args, **kwargs):
        if beta <= 0:
            raise ValueError("beta must be positive")
        kwargs.pop("alpha", None)
        kwargs.pop("retain_loss_type", None)
        super().__init__(*args, gamma=gamma, alpha=0.0, retain_loss_type="NLL", **kwargs)
        self.beta = beta
        self._component_loss_sums = {"forget_loss": 0.0}
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        forget_inputs = inputs["forget"]
        nonempty = forget_inputs["labels"][..., 1:].ne(-100).any(dim=-1)
        if not torch.any(nonempty):
            raise ValueError(f"{type(self).__name__} batch contains no valid prediction sequences")
        if not torch.all(nonempty):
            batch_size = nonempty.shape[0]
            forget_inputs = {
                key: (value[nonempty] if torch.is_tensor(value) and value.shape[0] == batch_size else value)
                for key, value in forget_inputs.items()
            }
        forget_loss, (_, forget_outputs) = compute_dpo_loss(
            model=model,
            ref_model=self.ref_model,
            win_inputs=None,
            lose_inputs=forget_inputs,
            beta=self.beta,
            normalize_by_length=self.normalize_by_length,
        )
        loss = self.gamma * forget_loss
        self._component_loss_sums["forget_loss"] += forget_loss.detach().float().item()
        self._component_loss_count += 1
        return (loss, forget_outputs) if return_outputs else loss


class NPO(_SequenceNPO):
    """Original loser-only DPO/NPO over summed sequence log-probability."""


class PureNPO(_SequenceNPO):
    """NPO with per-sequence valid-token normalization."""

    normalize_by_length = True


class TokenNPO(GradDiff):
    """NPO with an independent reference-relative objective per valid token."""

    def __init__(self, beta=1.0, gamma=1.0, *args, **kwargs):
        if beta <= 0:
            raise ValueError("beta must be positive")
        kwargs.pop("alpha", None)
        kwargs.pop("retain_loss_type", None)
        super().__init__(*args, gamma=gamma, alpha=0.0, retain_loss_type="NLL", **kwargs)
        self.beta = beta
        self._component_loss_sums = {
            "forget_loss": 0.0,
            "current_token_nll": 0.0,
            "reference_token_nll": 0.0,
            "token_gradient_weight": 0.0,
        }
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        forget_inputs = inputs["forget"]
        current_token_nll, forget_outputs = compute_token_nll(model, forget_inputs)
        with torch.no_grad():
            if hasattr(self.ref_model, "token_nll"):
                reference_token_nll = self.ref_model.token_nll(forget_inputs).to(
                    current_token_nll.device
                )
            else:
                reference_token_nll, _ = compute_token_nll(self.ref_model, forget_inputs)

        valid = forget_inputs["labels"][..., 1:].ne(-100)
        if not torch.any(valid):
            raise ValueError("TokenNPO batch contains no valid prediction tokens")
        logit = self.beta * (current_token_nll[valid] - reference_token_nll[valid])
        forget_loss = (-2.0 / self.beta * F.logsigmoid(logit)).mean()
        loss = self.gamma * forget_loss
        values = {
            "forget_loss": forget_loss,
            "current_token_nll": current_token_nll[valid].mean(),
            "reference_token_nll": reference_token_nll[valid].mean(),
            "token_gradient_weight": (2.0 * torch.sigmoid(-logit)).mean(),
        }
        for name, value in values.items():
            self._component_loss_sums[name] += value.detach().float().item()
        self._component_loss_count += 1
        return (loss, forget_outputs) if return_outputs else loss
