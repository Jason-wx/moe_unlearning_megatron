from trainer.utils import compute_dpo_loss
from trainer.unlearn.grad_diff import GradDiff


class NPO(GradDiff):
    """Pure Negative Preference Optimization over forget sequences only."""

    def __init__(self, beta=1.0, gamma=1.0, *args, **kwargs):
        # NPO reuses GradDiff's distributed reference-model preparation, but
        # pure NPO has no retain term. Discard legacy NPO+retain arguments so a
        # stale config cannot silently turn this back into NPO+GD.
        kwargs.pop("alpha", None)
        kwargs.pop("retain_loss_type", None)
        super().__init__(
            gamma=gamma,
            alpha=0.0,
            retain_loss_type="NLL",
            *args,
            **kwargs,
        )
        self.beta = beta
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_inputs = inputs["forget"]

        forget_loss, forget_outputs = compute_dpo_loss(
            model=model,
            ref_model=self.ref_model,
            win_inputs=None,
            lose_inputs=forget_inputs,
            beta=self.beta,
            normalize_by_length=True,
        )

        # Pure NPO intentionally has no retain forward pass or retain loss.
        loss = self.gamma * forget_loss
        return (loss, forget_outputs) if return_outputs else loss
