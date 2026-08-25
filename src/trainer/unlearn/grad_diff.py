import copy
import os
import torch
import torch.distributed as dist
from trainer.utils import compute_kl_divergence
from trainer.unlearn.base import UnlearnTrainer
from trainer.reference_client import ReferenceClient


class GradDiff(UnlearnTrainer):
    def __init__(
        self,
        gamma=1.0,
        alpha=1.0,
        retain_loss_type="NLL",
        reference_host=None,
        reference_port=None,
        reference_authkey=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.retain_loss_type = retain_loss_type
        self.reference_host = reference_host or os.getenv("REFERENCE_HOST")
        self.reference_port = reference_port or os.getenv("REFERENCE_PORT")
        self.reference_authkey = reference_authkey or os.getenv(
            "REFERENCE_AUTHKEY", "open-unlearning"
        )
        # Component losses are accumulated over the same interval used by
        # Trainer's `logging_steps`, so console and W&B receive one averaged
        # forget/retain value alongside the averaged total training loss.
        self._component_loss_sums = {"forget_loss": 0.0, "retain_loss": 0.0}
        self._component_loss_count = 0
        self.ref_model = None
        if retain_loss_type == "KL":
            self.ref_model = self._prepare_ref_model(self.model)

    def _prepare_ref_model(self, model):
        if self.reference_host and self.reference_port:
            return ReferenceClient(
                self.reference_host,
                int(self.reference_port),
                self.reference_authkey,
            )

        # A 35B BF16 model is about 70 GiB. Cloning it concurrently in every
        # local rank creates an avoidable host-memory spike before ZeRO-3 gets
        # a chance to partition the reference model. Serialize only this
        # transient clone-and-H2D phase; once a rank has moved its clone to its
        # H200, the temporary CPU tensors can be released before the next rank
        # starts cloning.
        distributed = dist.is_available() and dist.is_initialized()
        if self.is_deepspeed_enabled and distributed:
            ref_model = None
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            barrier_kwargs = (
                {"device_ids": [torch.cuda.current_device()]}
                if dist.get_backend() == "nccl"
                else {}
            )
            for clone_rank in range(world_size):
                if rank == clone_rank:
                    ref_model = copy.deepcopy(model)
                    ref_model.eval()
                    for parameter in ref_model.parameters():
                        parameter.requires_grad_(False)
                    ref_model = ref_model.to(self.accelerator.device)
                dist.barrier(**barrier_kwargs)
            if ref_model is None:
                raise RuntimeError("Failed to construct the local reference model")
        else:
            ref_model = copy.deepcopy(model)
            ref_model.eval()
            for parameter in ref_model.parameters():
                parameter.requires_grad_(False)
            ref_model = ref_model.to(self.accelerator.device)

        if self.is_deepspeed_enabled:
            ref_model = self._prepare_deepspeed(ref_model)
        else:
            ref_model = self.accelerator.prepare_model(ref_model, evaluation_mode=True)
        return ref_model

    def compute_retain_loss(self, model, retain_inputs):
        retain_outputs = model(**retain_inputs)
        retain_loss = 0.0
        if self.retain_loss_type == "NLL":
            retain_loss += retain_outputs.loss
        elif self.retain_loss_type == "KL":
            kl_loss, retain_outputs = compute_kl_divergence(
                self.model, self.ref_model, retain_inputs
            )
            retain_loss += kl_loss
        else:
            raise NotImplementedError(
                f"{self.retain_loss_type} not implemented for retain set"
            )
        return retain_loss

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }

        forget_outputs = model(**forget_inputs)
        forget_loss = -forget_outputs.loss

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss

        self._component_loss_sums["forget_loss"] += forget_loss.detach().float().item()
        self._component_loss_sums["retain_loss"] += retain_loss.detach().float().item()
        self._component_loss_count += 1

        return (loss, forget_outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        """Add averaged GradDiff components to normal Trainer/W&B logs."""
        if "loss" in logs and self._component_loss_count:
            count = self._component_loss_count
            for name, value in self._component_loss_sums.items():
                logs[name] = value / count
            self._component_loss_sums = {
                name: 0.0 for name in self._component_loss_sums
            }
            self._component_loss_count = 0
        super().log(logs, start_time=start_time)
