import torch
import torch.distributed as dist

from trainer.utils import compute_undial_loss
from trainer.unlearn.grad_diff import GradDiff


class UNDIAL(GradDiff):
    def __init__(self, beta=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_inputs = inputs["forget"]

        # The dedicated reference server handles requests from all ZeRO-3 ranks
        # sequentially.  Keep every rank at the same collective boundary while
        # the (large) teacher-logit responses are in flight; otherwise a fast
        # rank can enter the next model collective while another rank is still
        # waiting for the reference server, and DeepSpeed's trace-validation
        # broadcast can be matched with a model-parameter broadcast.
        distributed = dist.is_available() and dist.is_initialized()
        barrier_kwargs = (
            {"device_ids": [torch.cuda.current_device()]}
            if distributed and dist.get_backend() == "nccl"
            else {}
        )
        if distributed:
            dist.barrier(**barrier_kwargs)
        forget_loss, forget_outputs = compute_undial_loss(
            model, self.ref_model, forget_inputs, self.beta
        )
        if distributed:
            dist.barrier(**barrier_kwargs)

        # The paper-aligned Intern-S2 setup uses alpha=0. Computing a retain
        # forward in that case has no effect on the objective, doubles student
        # forward work, and introduces another unnecessary ZeRO-3 trace.
        if self.alpha == 0.0:
            retain_loss = forget_loss.detach().new_zeros(())
        else:
            retain_inputs = inputs["retain"]
            retain_inputs = {
                "input_ids": retain_inputs["input_ids"],
                "attention_mask": retain_inputs["attention_mask"],
                "labels": retain_inputs["labels"],
            }
            retain_loss = self.compute_retain_loss(
                model=model, retain_inputs=retain_inputs
            )

        loss = self.gamma * forget_loss + self.alpha * retain_loss

        self._component_loss_sums["forget_loss"] += (
            forget_loss.detach().float().item()
        )
        self._component_loss_sums["retain_loss"] += (
            retain_loss.detach().float().item()
        )
        self._component_loss_count += 1

        return (loss, forget_outputs) if return_outputs else loss
