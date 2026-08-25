"""Offline validation for candidate Megatron parallel layouts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParallelLayout:
    world_size: int = 8
    tensor_parallel: int = 4
    pipeline_parallel: int = 2
    context_parallel: int = 1
    expert_parallel: int = 4
    expert_tensor_parallel: int | None = 1
    sequence_parallel: bool = True
    num_experts: int | None = None

    @property
    def dense_model_parallel_size(self) -> int:
        return self.tensor_parallel * self.pipeline_parallel * self.context_parallel

    @property
    def data_parallel_size(self) -> int:
        return self.world_size // self.dense_model_parallel_size

    @property
    def resolved_expert_tensor_parallel(self) -> int:
        if self.expert_tensor_parallel is None:
            return self.tensor_parallel
        return self.expert_tensor_parallel

    @property
    def expert_model_parallel_size(self) -> int:
        return self.resolved_expert_tensor_parallel * self.expert_parallel * self.pipeline_parallel

    @property
    def expert_data_parallel_size(self) -> int:
        return self.world_size // self.expert_model_parallel_size

    def validate(self) -> ParallelLayout:
        integer_fields = {
            "world_size": self.world_size,
            "tensor_parallel": self.tensor_parallel,
            "pipeline_parallel": self.pipeline_parallel,
            "context_parallel": self.context_parallel,
            "expert_parallel": self.expert_parallel,
            "expert_tensor_parallel": self.resolved_expert_tensor_parallel,
        }
        invalid = {name: value for name, value in integer_fields.items() if value <= 0}
        if invalid:
            raise ValueError(f"parallel sizes must be positive: {invalid}")
        if self.world_size % self.dense_model_parallel_size:
            raise ValueError(
                "world_size must be divisible by TP * PP * CP: "
                f"{self.world_size} % {self.dense_model_parallel_size} != 0"
            )
        if self.world_size % self.expert_model_parallel_size:
            raise ValueError(
                "world_size must be divisible by ETP * EP * PP: "
                f"{self.world_size} % {self.expert_model_parallel_size} != 0"
            )
        if self.sequence_parallel and self.tensor_parallel <= 1:
            raise ValueError("sequence parallelism requires tensor_parallel > 1")
        if self.num_experts is not None:
            if self.num_experts <= 0:
                raise ValueError("num_experts must be positive")
            if self.num_experts % self.expert_parallel:
                raise ValueError(
                    f"num_experts ({self.num_experts}) must be divisible by "
                    f"expert_parallel ({self.expert_parallel})"
                )
        return self
