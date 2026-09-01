"""Reference-free SimNPO variants."""

import torch
import torch.nn.functional as F

from trainer.unlearn.grad_diff import GradDiff
from trainer.utils import compute_batch_nll, compute_token_nll


class _SimNPOBase(GradDiff):
    token_level = False

    def __init__(self, delta=0.0, beta=1.0, *args, **kwargs):
        if beta <= 0:
            raise ValueError("beta must be positive")
        super().__init__(*args, **kwargs)
        self.delta = delta
        self.beta = beta
        self._component_loss_sums.update(
            {
                "forget_mean_nll": 0.0,
                "sigmoid_argument": 0.0,
                "forget_gradient_weight": 0.0,
            }
        )

    def _forget_objective(self, model, forget_inputs):
        valid_mask = forget_inputs["labels"][..., 1:].ne(-100)
        if self.token_level:
            token_nll, outputs = compute_token_nll(model, forget_inputs)
            reduced_nll = token_nll[valid_mask]
            if reduced_nll.numel() == 0:
                raise ValueError("TokenSimNPO batch contains no valid prediction tokens")
        else:
            sequence_nll, outputs = compute_batch_nll(model, forget_inputs)
            counts = valid_mask.sum(dim=-1)
            nonempty = counts.gt(0)
            if not torch.any(nonempty):
                raise ValueError("SimNPO batch contains no valid prediction sequences")
            reduced_nll = sequence_nll[nonempty] / counts[nonempty].to(sequence_nll.dtype)

        sigmoid_argument = self.beta * (reduced_nll - self.delta)
        forget_loss = (-2.0 / self.beta * F.logsigmoid(sigmoid_argument)).mean()
        return forget_loss, outputs, reduced_nll, sigmoid_argument

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        forget_loss, forget_outputs, reduced_nll, sigmoid_argument = self._forget_objective(
            model, inputs["forget"]
        )

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss

        component_values = {
            "forget_loss": forget_loss,
            "retain_loss": retain_loss,
            "forget_mean_nll": reduced_nll.mean(),
            "sigmoid_argument": sigmoid_argument.mean(),
            "forget_gradient_weight": (2.0 * torch.sigmoid(-sigmoid_argument)).mean(),
        }
        for name, value in component_values.items():
            self._component_loss_sums[name] += value.detach().float().item()
        self._component_loss_count += 1

        return (loss, forget_outputs) if return_outputs else loss


class SimNPO(_SimNPOBase):
    """Original SimNPO over length-normalized sequence NLL."""


class TokenSimNPO(_SimNPOBase):
    """SimNPO with its logistic transform applied independently per token."""

    token_level = True
