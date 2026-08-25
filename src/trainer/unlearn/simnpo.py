import torch
import torch.nn.functional as F

from trainer.utils import compute_token_nll
from trainer.unlearn.grad_diff import GradDiff


class SimNPO(GradDiff):
    def __init__(self, delta=0.0, beta=1.0, *args, **kwargs):
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

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_inputs = inputs["forget"]

        # Apply the bounded SimNPO objective independently to each valid
        # next-token NLL.  A few already-forgotten tokens therefore cannot
        # saturate the weight of every other token in a 1024-token document.
        token_nll, forget_outputs = compute_token_nll(model, forget_inputs)
        valid_mask = forget_inputs["labels"][..., 1:].ne(-100)
        valid_token_nll = token_nll[valid_mask]
        if valid_token_nll.numel() == 0:
            raise ValueError("SimNPO batch contains no valid prediction tokens")

        sigmoid_argument = self.beta * (valid_token_nll - self.delta)
        token_forget_loss = -2.0 / self.beta * F.logsigmoid(sigmoid_argument)
        forget_loss = token_forget_loss.mean()
        forget_mean_nll = valid_token_nll.mean()
        # Magnitude of dL/d(NLL). It starts near one and approaches zero only
        # for tokens that individually enter the saturated region.
        forget_gradient_weight = (2.0 * torch.sigmoid(-sigmoid_argument)).mean()

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
            "forget_mean_nll": forget_mean_nll,
            "sigmoid_argument": sigmoid_argument.mean(),
            "forget_gradient_weight": forget_gradient_weight,
        }
        for name, value in component_values.items():
            self._component_loss_sums[name] += value.detach().float().item()
        self._component_loss_count += 1

        return (loss, forget_outputs) if return_outputs else loss
