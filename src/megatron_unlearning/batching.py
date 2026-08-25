"""CPU-testable batch construction for GradAscent and GradDiff."""

from __future__ import annotations

from collections.abc import Sequence

import torch

IGNORE_INDEX = -100
FORGET_ROLE = 0
RETAIN_ROLE = 1


def _pad(values: Sequence[torch.Tensor], value: int, length: int) -> torch.Tensor:
    output = torch.full((len(values), length), value, dtype=torch.long)
    for row, tensor in enumerate(values):
        size = min(int(tensor.numel()), length)
        output[row, :size] = tensor[:size].to(dtype=torch.long)
    return output


def collate_role_batch(
    forget: Sequence[dict[str, torch.Tensor]],
    retain: Sequence[dict[str, torch.Tensor]] | None,
    *,
    pad_token_id: int,
    sequence_length: int,
) -> dict[str, torch.Tensor]:
    """Combine forget/retain examples along batch dimension for one forward."""
    examples = list(forget)
    roles = [FORGET_ROLE] * len(examples)
    if retain:
        examples.extend(retain)
        roles.extend([RETAIN_ROLE] * len(retain))
    if not examples:
        raise ValueError("at least one forget example is required")

    input_ids = _pad([item["input_ids"] for item in examples], pad_token_id, sequence_length)
    source_labels = _pad([item["labels"] for item in examples], IGNORE_INDEX, sequence_length)
    padding_mask = input_ids.ne(pad_token_id)
    # Transformers causal-LM heads shift labels internally. MCore consumes an
    # already aligned target for every input position, so label[t] must be the
    # source label at t + 1. Preserve any source-side -100 mask while shifting.
    labels = torch.full_like(source_labels, IGNORE_INDEX)
    labels[:, :-1] = source_labels[:, 1:]
    labels = labels.masked_fill(~padding_mask, IGNORE_INDEX)
    # Intermediate-activation objectives such as RMU follow the original HF
    # implementation's unshifted label mask, while MCore LM losses use the
    # next-token-aligned labels above.
    activation_mask = source_labels.ne(IGNORE_INDEX) & padding_mask
    position_ids = torch.arange(sequence_length, dtype=torch.long).expand(len(examples), -1)
    causal = torch.triu(
        torch.ones((sequence_length, sequence_length), dtype=torch.bool), diagonal=1
    )
    attention_mask = causal[None, None, :, :] | (~padding_mask[:, None, None, :])
    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": labels.ne(IGNORE_INDEX),
        "activation_mask": activation_mask,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "roles": torch.tensor(roles, dtype=torch.long),
    }
