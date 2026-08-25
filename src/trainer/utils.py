import torch
import random
import numpy as np
from torch import nn
import torch.nn.functional as F


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_kl_divergence(model, target_model, inputs):
    with torch.no_grad():
        if hasattr(target_model, "logits"):
            ref_logits = target_model.logits(inputs)
        else:
            ref_logits = target_model(**inputs).logits

    outputs = model(**inputs)
    ref_logits = ref_logits.to(outputs.logits.device, non_blocking=True)
    ref_probs = F.log_softmax(ref_logits, dim=-1)
    ref_probs = ref_probs.view(-1, ref_logits.shape[-1])
    current_probs = F.log_softmax(outputs.logits, dim=-1)
    current_probs = current_probs.view(-1, outputs.logits.shape[-1])

    # minimum KL divergence
    return nn.functional.kl_div(
        current_probs, ref_probs, reduction="batchmean", log_target=True
    ), outputs


def compute_token_nll(model, inputs):
    """Return unreduced next-token NLL with shape ``[batch, seq_len - 1]``."""
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs["labels"]
    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    loss = loss_function(logits.transpose(-1, -2), shifted_labels)
    return loss, outputs


def compute_batch_nll(model, inputs):
    # get the sum loss for each sequence in a batch
    # NOTE: not same as model(**inputs).loss but has sum loss for each seq in a batch
    token_nll, outputs = compute_token_nll(model, inputs)
    loss = token_nll.sum(dim=-1)
    return loss, outputs


def _mean_sequence_nll(sequence_nll, inputs):
    """Convert summed sequence NLL to mean NLL over valid predicted tokens."""
    labels = inputs["labels"]
    valid_token_count = labels[..., 1:].ne(-100).sum(dim=-1).clamp_min(1)
    return sequence_nll / valid_token_count.to(
        device=sequence_nll.device, dtype=sequence_nll.dtype
    )


def compute_dpo_loss(
    model,
    ref_model,
    win_inputs=None,
    lose_inputs=None,
    beta=1.0,
    normalize_by_length=False,
):
    if win_inputs is None and lose_inputs is None:
        raise ValueError("Both win_inputs and lose_inputs can't be None")

    win_log_ratio, lose_log_ratio = 0.0, 0.0
    win_outputs, lose_outputs = None, None

    if win_inputs is not None:
        win_loss, win_outputs = compute_batch_nll(model, win_inputs)
        with torch.no_grad():
            if hasattr(ref_model, "sequence_nll"):
                win_ref_loss = ref_model.sequence_nll(win_inputs).to(win_loss.device)
            else:
                win_ref_loss, _ = compute_batch_nll(ref_model, win_inputs)
        if normalize_by_length:
            win_loss = _mean_sequence_nll(win_loss, win_inputs)
            win_ref_loss = _mean_sequence_nll(win_ref_loss, win_inputs)
        win_log_ratio = -(win_loss - win_ref_loss)

    if lose_inputs is not None:
        lose_loss, lose_outputs = compute_batch_nll(model, lose_inputs)
        with torch.no_grad():
            if hasattr(ref_model, "sequence_nll"):
                lose_ref_loss = ref_model.sequence_nll(lose_inputs).to(lose_loss.device)
            else:
                lose_ref_loss, _ = compute_batch_nll(ref_model, lose_inputs)
        if normalize_by_length:
            lose_loss = _mean_sequence_nll(lose_loss, lose_inputs)
            lose_ref_loss = _mean_sequence_nll(lose_ref_loss, lose_inputs)
        lose_log_ratio = -(lose_loss - lose_ref_loss)

    loss = -2 / beta * F.logsigmoid(beta * (win_log_ratio - lose_log_ratio)).mean()
    return loss, (win_outputs, lose_outputs)


def compute_undial_loss(model, ref_model, inputs, beta):
    # Forward pass on the student (trainable) model
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs["labels"]

    shift_labels = labels[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()

    # Forward pass on the teacher model (no grad)
    with torch.no_grad():
        if hasattr(ref_model, "logits"):
            teacher_logits = ref_model.logits(inputs)
        else:
            teacher_logits = ref_model(**inputs).logits
        teacher_logits = teacher_logits.to(logits.device, non_blocking=True)
    shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()

    # Build the mask that identifies the tokens need to be unlearned
    valid_tokens = shift_labels.ne(-100)
    safe_labels = shift_labels.masked_fill(~valid_tokens, 0)
    mask = torch.zeros_like(shift_teacher_logits)
    mask.scatter_(
        -1,
        safe_labels.unsqueeze(-1),
        valid_tokens.unsqueeze(-1).to(mask.dtype),
    )

    # Adjust teacher logits: subtract di_strength on the correct token
    pre_softmax = shift_teacher_logits - mask * beta
    soft_label = F.softmax(pre_softmax, dim=-1)

    loss_fct = nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        soft_label.view(-1, soft_label.size(-1)),
    )
    valid_tokens = valid_tokens.view(-1)
    return loss[valid_tokens].mean(), outputs


def compute_wga_loss(model, inputs, beta):
    outputs = model(**inputs)
    labels = inputs["labels"]
    labels = labels.to(outputs.logits.device)

    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    lm_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    weight_ce = ((-lm_loss).exp().detach()) ** beta
    forget_loss = -(weight_ce * lm_loss)[shift_labels.view(-1) != -100].mean()
    return forget_loss, outputs


def compute_satimp_loss(model, inputs, beta1, beta2):
    outputs = model(**inputs)
    labels = inputs["labels"]
    labels = labels.to(outputs.logits.device)

    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    lm_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    weight_sat = ((-lm_loss).exp().detach()) ** beta1
    weight_imp = (1 - (-lm_loss).exp().detach()) ** beta2
    forget_loss = -((weight_sat * weight_imp) * lm_loss)[
        shift_labels.view(-1) != -100
    ].mean()
    return forget_loss, outputs
