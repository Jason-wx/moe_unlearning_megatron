"""In-training online evaluation on lm-eval MCQ benchmarks.

The training loop can score the *resident* GPU parameters on wmdp_bio /
wmdp_cyber / mmlu at chosen steps without exporting a checkpoint or holding a
second copy of the model. Prompts and scoring reuse the ``lm_eval`` task API so
results are identical-by-construction to the offline baseline; only the
log-likelihood backend is swapped for an MCore forward pass.

Design notes:

* ``lm_eval`` only shards / pads / gathers requests when ``lm.world_size > 1``.
  :class:`MCoreEvalLM` keeps the base-class default ``rank=0 / world_size=1`` so
  every rank receives the *full* ordered request list and computes the *same*
  metrics. The EP8 cross-rank coordination is done privately inside
  :meth:`MCoreEvalLM.loglikelihood` (question-strided DP shard + all-gather). Do
  **not** set ``lm._world_size`` to the DP size -- that triggers
  ``accelerator.gather``.
* The four answer choices of one MCQ question share a prompt, so they are scored
  together, and several *questions* are in turn packed into a single forward:
  with ``batch_size`` questions per forward an N-choice task runs
  ``ceil(questions_per_rank / batch_size)`` forwards per rank instead of one per
  question (e.g. mmlu: 14042 questions become ~28 forwards/rank at batch_size=64
  and dp_size=8, not 1756). ``batch_size``, the per-rank question count and the
  forward count are all derived identically on every rank from the full request
  list, and each forward is padded to a fixed ``[batch_size * max_choices,
  seq_length]`` shape -- so forward shapes and counts stay bit-identical across
  ranks as EP8 all-to-all requires. ``seq_length`` is the global longest real
  length (rounded up for kernel alignment, capped at the training
  sequence_length). The effective questions-per-forward is additionally clamped
  (:func:`clamp_questions_per_forward`) so one forward routes at most
  ``_MAX_TOKENS_PER_FORWARD`` tokens through the MoE -- the token dispatcher's
  transient scales with tokens/forward and eval runs with the training footprint
  resident, so an unclamped ``batch_size`` OOMs inside the MoE layer.
* Scoring never materializes the full ``[rows, seq, vocab]`` logits. The output
  layer (a ``vocab_size=251392`` projection whose fp32 cast under TP=1 was the
  53.7 GiB OOM) is skipped: a forward hook captures the decoder hidden states
  ``[s, b, h=2048]`` while ``post_process=False`` returns early, then the LM head
  is applied by :func:`torch.nn.functional.linear` at *only* the loss-mask
  (continuation) positions, log-softmaxed in fp32 there. This reproduces the
  CE-based continuation NLL exactly (same weights, same decoder states, same
  positions) but the transient fp32 logits shrink from ``[rows*seq, vocab]`` to
  ``[num_continuation_tokens, vocab]`` -- ~122x (h vs vocab) smaller per row and
  restricted to the handful of masked positions, so peak drops from ~80 GiB to a
  few GiB and ``batch_size`` can stay 16+. Any model-specific *output-side*
  transform the real ``_forward_output_layer`` would apply must be replayed here,
  since ``post_process=False`` bypasses it: Gemma's final-logit softcapping
  (``softcap * tanh(logits / softcap)``) is forwarded into
  :func:`reduce_selected_logprobs` for that reason. (The input-side Gemma
  ``embed_scale`` is *not* bypassed -- it runs inside the ordinary embedding
  forward that ``_forward_hidden`` still executes in full.)
* Megatron / CUDA imports are lazy (inside functions) so this module stays
  importable on a CPU login node; the pure-torch helpers below are unit-tested
  there.
"""

from __future__ import annotations

import math
import os
import random
from collections.abc import Sequence

import torch

from .batching import IGNORE_INDEX, collate_role_batch
from .forward_step import extra_model_kwargs

# Forward sequence length is padded up to a multiple of this for kernel
# alignment (flash-attention / MoE permute prefer aligned lengths). Kept small
# so 0-shot MCQ prompts stay far below the training sequence_length.
_SEQ_ALIGN = 128

# Upper bound on tokens routed through the MoE per eval forward
# (qpf * max_choices * seq_length). The MoE token dispatcher's transient
# (all-to-all permute + sort_chunks_by_idxs) scales with this; online eval runs
# with the full training footprint resident (~121 GiB), so only a few GiB are
# free and a forward routing many more tokens than a *training* forward
# (micro_batch_size * seq_length ~ a few thousand) OOMs inside the MoE layer even
# though the output-layer logits are no longer materialized. 4096 keeps that
# transient ~0.5 GiB (the 7.01 GiB failure was 57344 tokens/forward), well within
# the eval-time free memory, at the cost of a few more forwards.
_MAX_TOKENS_PER_FORWARD = 4096

__all__ = [
    "MCoreEvalLM",
    "build_mcq_batch",
    "clamp_questions_per_forward",
    "continuation_logprob",
    "gather_scoring_targets",
    "group_choices_by_context",
    "plan_dp_forwards",
    "reduce_selected_logprobs",
    "round_up_to_multiple",
    "run_online_evaluation",
    "split_context_continuation",
]


def plan_dp_forwards(
    num_groups: int, dp_rank: int, dp_size: int, questions_per_forward: int
) -> tuple[int, int, list[list[tuple[int, int]]]]:
    """Plan one DP rank's batched forwards over the question groups.

    Questions (groups) are owned strided: rank ``r`` owns groups ``r, r+dp_size,
    r+2*dp_size, ...`` (so group ``g`` lives at ``rank=g % dp_size, slot=g //
    dp_size`` -- the layout :meth:`MCoreEvalLM.loglikelihood` reassembles by).
    Every rank owns ``groups_per_rank = ceil(num_groups / dp_size)`` slots
    (tail-padded by repeating the last real question) and packs ``qpf =
    min(questions_per_forward, groups_per_rank)`` questions per forward, giving
    ``ceil(groups_per_rank / qpf)`` forwards. Because ``groups_per_rank``, ``qpf``
    and the forward count depend only on ``num_groups`` / ``dp_size`` (identical
    on every rank), all ranks run the same number of forwards at the same
    ``[qpf * max_choices, seq_length]`` shape -- the EP8 all-to-all lockstep
    invariant.

    Returns ``(groups_per_rank, qpf, forwards)`` where ``forwards`` is a list
    (one per forward) of ``qpf`` ``(slot, source_group_idx)`` pairs. ``slot`` is
    this rank's local slot index; ``source_group_idx`` is the global group whose
    choices fill it (clamped to the last real group for tail/batch padding). A
    padding pair has ``slot >= groups_per_rank`` and its result is discarded.
    """
    if questions_per_forward <= 0:
        raise ValueError("questions_per_forward must be positive")
    if num_groups <= 0 or dp_size <= 0:
        return 0, 0, []
    groups_per_rank = math.ceil(num_groups / dp_size)
    qpf = min(questions_per_forward, groups_per_rank)
    num_forwards = math.ceil(groups_per_rank / qpf)
    forwards: list[list[tuple[int, int]]] = []
    for f in range(num_forwards):
        slots: list[tuple[int, int]] = []
        for j in range(qpf):
            slot = f * qpf + j
            real_slot = slot if slot < groups_per_rank else groups_per_rank - 1
            group_idx = dp_rank + real_slot * dp_size
            source_idx = group_idx if group_idx < num_groups else num_groups - 1
            slots.append((slot, source_idx))
        forwards.append(slots)
    return groups_per_rank, qpf, forwards


def group_choices_by_context(contexts: Sequence[str]) -> list[list[int]]:
    """Group the answer-choice requests of each MCQ question together.

    lm_eval emits one ``(context, continuation)`` request per answer choice, and
    the choices of a single question share a byte-identical ``context`` and
    arrive consecutively (doc order). This returns, for each maximal run of
    identical adjacent contexts, the list of the original request indices in that
    run -- i.e. one list per question. Scoring a whole run in a single forward
    turns the 4-choice wmdp/mmlu tasks into one forward per question instead of
    one per choice.
    """
    groups: list[list[int]] = []
    for i, context in enumerate(contexts):
        if groups and contexts[groups[-1][0]] == context:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def round_up_to_multiple(value: int, multiple: int) -> int:
    """Round ``value`` up to the next positive multiple of ``multiple``."""
    if multiple <= 0:
        raise ValueError("multiple must be positive")
    if value <= 0:
        return multiple
    return ((value + multiple - 1) // multiple) * multiple


def clamp_questions_per_forward(
    requested: int,
    *,
    max_choices: int,
    seq_length: int,
    max_tokens_per_forward: int,
) -> int:
    """Lower questions-per-forward so one forward routes a bounded token count.

    One forward routes ``qpf * max_choices * seq_length`` tokens through the MoE;
    the dispatcher's transient (all-to-all permute + ``sort_chunks_by_idxs``)
    scales with that, and online eval has only a few GiB free above the resident
    training footprint. This clamps ``qpf`` so the routed tokens stay at/below
    ``max_tokens_per_forward`` (always >= 1). It depends only on the global
    ``max_choices`` / ``seq_length`` (identical on every rank), so all ranks pick
    the same ``qpf`` and run the same number of forwards at the same shape -- the
    EP8 all-to-all lockstep invariant.
    """
    if requested <= 0:
        raise ValueError("requested questions_per_forward must be positive")
    per_question_tokens = max(1, max_choices * seq_length)
    budget = max(1, max_tokens_per_forward // per_question_tokens)
    return max(1, min(requested, budget))


def split_context_continuation(
    tokenizer, context: str, continuation: str
) -> tuple[list[int], list[int]]:
    """Tokenize ``(context, continuation)`` the way lm_eval's HFLM does.

    Mirrors ``HFLM._encode_pair``: a trailing space on the context is merged into
    the continuation before encoding, and the continuation token span is the
    suffix of the whole-string encoding beyond the context prefix length. This is
    robust to sub-word merges at the boundary. Both encodes use the tokenizer's
    *default* ``add_special_tokens`` (``HFLM._encode_pair`` passes no override, so
    ``add_bos_token=None`` falls through to the tokenizer default) -- for the
    Gemma family that prepends ``<bos>`` to the whole string and to the context,
    so the continuation slice is unchanged but the context carries its leading
    ``<bos>``. Forcing ``add_special_tokens=False`` here dropped that BOS and,
    because Gemma is trained to always see a BOS, collapsed MCQ accuracy to chance
    (wmdp_bio 0.79 -> 0.41). Non-BOS tokenizers (Intern-S2 / Qwen3.5-MoE) are
    unaffected: their default adds no BOS, matching the offline HFLM baseline.
    """
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]
    whole_enc = tokenizer.encode(context + continuation)
    context_enc = tokenizer.encode(context)
    context_len = len(context_enc)
    continuation_enc = whole_enc[context_len:]
    return context_enc, continuation_enc


def build_mcq_batch(
    examples: Sequence[tuple[list[int], list[int]]],
    *,
    pad_token_id: int,
    sequence_length: int,
) -> dict[str, torch.Tensor]:
    """Build an MCore forward batch for MCQ log-likelihood scoring.

    ``examples`` is a sequence of ``(context_ids, continuation_ids)`` pairs. Each
    is laid out as ``input_ids = context + continuation`` with a source label of
    ``IGNORE_INDEX`` on the context positions and the token id on the
    continuation positions. Delegating to :func:`collate_role_batch` applies the
    exact MCore contract used in training (next-token label shift, causal +
    padding attention mask, ``loss_mask = labels.ne(IGNORE_INDEX)``), so after
    the shift the loss mask covers precisely the positions whose per-token NLL is
    the continuation's ``-logprob``.

    Over-long sequences are left-truncated (drop context tokens) so the full
    continuation is always preserved -- the lm_eval convention.
    """
    collate_examples: list[dict[str, torch.Tensor]] = []
    for context_ids, continuation_ids in examples:
        if not continuation_ids:
            raise ValueError("continuation must contain at least one token")
        full_ids = list(context_ids) + list(continuation_ids)
        source_labels = [IGNORE_INDEX] * len(context_ids) + list(continuation_ids)
        if len(full_ids) > sequence_length:
            # Left-truncate context, keep the continuation tail intact.
            overflow = len(full_ids) - sequence_length
            if overflow >= len(context_ids):
                raise ValueError(
                    "continuation does not fit in sequence_length after truncating context"
                )
            full_ids = full_ids[overflow:]
            source_labels = source_labels[overflow:]
        collate_examples.append(
            {
                "input_ids": torch.tensor(full_ids, dtype=torch.long),
                "labels": torch.tensor(source_labels, dtype=torch.long),
            }
        )
    return collate_role_batch(
        collate_examples,
        None,
        pad_token_id=pad_token_id,
        sequence_length=sequence_length,
    )


def continuation_logprob(per_token_nll: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """Sum masked per-token NLL into a per-example continuation log-probability.

    ``per_token_nll`` and ``loss_mask`` are ``[batch, seq]``; returns ``[batch]``
    log-probabilities (``logprob = -sum(nll * loss_mask)``).
    """
    masked = per_token_nll.float() * loss_mask.to(per_token_nll.dtype)
    return -masked.sum(dim=1)


def gather_scoring_targets(
    labels: torch.Tensor, loss_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten the loss-mask (continuation) positions to be scored.

    ``labels`` / ``loss_mask`` are ``[batch, seq]``. Returns ``(row_idx, pos_idx,
    target_ids)`` -- three parallel ``[num_selected]`` tensors naming every masked
    position: the batch row, the sequence position, and the gold token id there.

    The collate contract already applied the next-token shift, so at a masked
    position ``t`` the decoder hidden state ``hidden[t]`` predicts ``labels[t]``.
    Scoring the log-prob of ``labels[t]`` from ``hidden[t]`` at exactly these
    positions therefore reproduces the CE-based continuation NLL term for term,
    without ever forming logits at the ignored context / padding positions.
    """
    mask = loss_mask.to(torch.bool)
    row_idx, pos_idx = torch.nonzero(mask, as_tuple=True)
    target_ids = labels[row_idx, pos_idx]
    return row_idx, pos_idx, target_ids


def reduce_selected_logprobs(
    selected_logits: torch.Tensor,
    row_idx: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    num_rows: int,
    final_logit_softcapping: float | None = None,
) -> torch.Tensor:
    """Sum per-position gold log-probs back into a ``[num_rows]`` logprob vector.

    ``selected_logits`` is ``[num_selected, vocab]`` (the LM head applied at the
    masked positions), ``target_ids`` ``[num_selected]`` the gold token at each,
    and ``row_idx`` ``[num_selected]`` the batch row each position belongs to.
    Returns ``[num_rows]`` where ``out[r]`` is the sum over row ``r``'s masked
    positions of ``log_softmax(logits)[target]`` -- i.e. the continuation
    log-probability, identical to ``-sum(per_token_nll * loss_mask)`` but computed
    only where the loss mask is set. The fp32 log-softmax is confined to the small
    ``[num_selected, vocab]`` slice.

    ``final_logit_softcapping`` reproduces Gemma-family final-logit softcapping
    (``Gemma4TextGPTModel._forward_output_layer``): when set, the raw
    ``F.linear`` logits are passed through ``softcap * tanh(logits / softcap)``
    *before* the log-softmax, exactly as the model's real output layer does. The
    selective path bypasses that override (it applies the LM head by hand to save
    memory), so without this the online scores diverge from the offline baseline
    for capped models. ``None`` (the default) is the identity -- correct for
    models without softcapping (e.g. Intern-S2 / Qwen3.5-MoE).
    """
    if final_logit_softcapping is not None:
        selected_logits = selected_logits / final_logit_softcapping
        selected_logits = torch.tanh(selected_logits)
        selected_logits = selected_logits * final_logit_softcapping
    logp = torch.log_softmax(selected_logits.float(), dim=-1)
    tok = logp.gather(1, target_ids.view(-1, 1).long()).squeeze(1)
    out = torch.zeros(num_rows, dtype=torch.float32, device=selected_logits.device)
    out.index_add_(0, row_idx.long(), tok.to(out.dtype))
    return out


def _import_lm_eval_base():
    from lm_eval.api.model import LM

    return LM


def _make_mcore_eval_lm_class():
    """Build the MCoreEvalLM class lazily so lm_eval is only imported on GPU."""
    LM = _import_lm_eval_base()

    class _MCoreEvalLM(LM):
        """lm_eval backend that scores continuations with an MCore forward pass.

        Kept at the base-class default ``rank=0 / world_size=1`` so lm_eval does
        no internal sharding; DP fan-out happens privately in
        :meth:`loglikelihood`.
        """

        def __init__(
            self,
            model,
            tokenizer,
            *,
            sequence_length: int,
            pad_token_id: int,
            dp_group,
            dp_rank: int,
            dp_size: int,
            device,
            questions_per_forward: int = 1,
            max_tokens_per_forward: int = _MAX_TOKENS_PER_FORWARD,
        ) -> None:
            super().__init__()
            self._model = model
            self._tokenizer = tokenizer
            self._sequence_length = sequence_length
            self._pad_token_id = pad_token_id
            self._dp_group = dp_group
            self._dp_rank = dp_rank
            self._dp_size = dp_size
            self._device = device
            self._questions_per_forward = questions_per_forward
            self._max_tokens_per_forward = max_tokens_per_forward

        def _forward_hidden(
            self, batch: dict[str, torch.Tensor], *, seq_length: int
        ) -> torch.Tensor:
            """Run the decoder only and return its hidden states as ``[b, s, h]``.

            A forward hook on ``gpt.decoder`` captures its output (``[s, b, h]``,
            or ``output[0]`` when the decoder returns the mHC+MTP multistream
            tuple) -- exactly the tensor the output layer would consume. Setting
            ``post_process=False`` makes the GPTModel return *before* the output
            layer materializes the ``[rows, seq, vocab]`` fp32 logits that OOM'd,
            and before the MTP block; the early-return value itself is ignored
            (for this multimodal+MTP model it would concat an MTP stream), so the
            hook is the robust source. ``post_process`` and the hook are restored
            on exit. Decoder execution -- and therefore the EP8 all-to-all -- is
            byte-identical to the training/eval forward.
            """
            gpt = getattr(self._model, "language_model", self._model)
            captured: dict[str, torch.Tensor] = {}

            def _hook(_module, _inputs, output):
                captured["hidden"] = output[0] if isinstance(output, tuple) else output

            handle = gpt.decoder.register_forward_hook(_hook)
            prev_post_process = gpt.post_process
            gpt.post_process = False
            try:
                self._model(
                    input_ids=batch["input_ids"].to(self._device),
                    position_ids=batch["position_ids"].to(self._device),
                    attention_mask=batch["attention_mask"].to(self._device),
                    labels=None,
                    **extra_model_kwargs(self._model, batch),
                )
            finally:
                gpt.post_process = prev_post_process
                handle.remove()
            hidden = captured["hidden"]  # [s, b, h]
            return hidden.transpose(0, 1)  # [b, s, h]

        def _output_weight(self) -> torch.Tensor:
            """Return the LM-head weight (``[vocab, h]``) for the selective matmul.

            Mirrors ``_postprocess``: a tied model reuses the embedding weight,
            otherwise the output layer owns its own. TP==1 here (asserted in
            :func:`run_online_evaluation`), so this is the full unsharded vocab.
            """
            gpt = getattr(self._model, "language_model", self._model)
            if getattr(gpt, "share_embeddings_and_output_weights", False):
                return gpt.shared_embedding_or_output_weight()
            return gpt.output_layer.weight

        def _score_batch(
            self,
            examples: Sequence[tuple[list[int], list[int]]],
            *,
            seq_length: int,
        ) -> torch.Tensor:
            """Forward one packed batch of MCQ answer choices in a single pass.

            ``examples`` is the flat, row-major list of ``(context_ids,
            continuation_ids)`` pairs for a whole forward -- ``batch_size``
            questions concatenated, each already padded to ``max_choices`` rows
            (so ``len(examples) == batch_size * max_choices``, identical on every
            rank for EP8 all-to-all lockstep). Returns a ``[len(examples)]`` tensor
            of continuation log-probabilities in the same row order.

            The LM head is applied (via :func:`torch.nn.functional.linear`) only at
            the continuation (loss-mask) positions, so the fp32 log-softmax touches
            just ``[num_continuation_tokens, vocab]`` instead of the full
            ``[rows, seq, vocab]`` logits that OOM'd -- exact CE parity at a few
            GiB peak. When the model applies final-logit softcapping (Gemma
            family), its ``final_logit_softcapping`` is forwarded so the manual LM
            head reproduces the model's ``_forward_output_layer`` tanh-cap that the
            ``post_process=False`` early-return bypasses.
            """
            batch = build_mcq_batch(
                examples,
                pad_token_id=self._pad_token_id,
                sequence_length=seq_length,
            )
            hidden = self._forward_hidden(batch, seq_length=seq_length)  # [b, s, h]
            labels = batch["labels"].to(self._device)
            loss_mask = batch["loss_mask"].to(self._device)
            row_idx, pos_idx, target_ids = gather_scoring_targets(labels, loss_mask)
            selected_hidden = hidden[row_idx, pos_idx]  # [num_selected, h]
            weight = self._output_weight()
            selected_logits = torch.nn.functional.linear(
                selected_hidden.to(weight.dtype), weight
            )  # [num_selected, vocab]
            # Gemma-family models tanh-softcap the final logits inside their
            # overridden _forward_output_layer; the selective path skips that
            # override, so reproduce it here. getattr returns None for models
            # without softcapping (Intern-S2 / Qwen3.5-MoE) -> identity.
            gpt = getattr(self._model, "language_model", self._model)
            final_logit_softcapping = getattr(gpt, "final_logit_softcapping", None)
            return reduce_selected_logprobs(
                selected_logits,
                row_idx,
                target_ids,
                num_rows=labels.shape[0],
                final_logit_softcapping=final_logit_softcapping,
            )

        def loglikelihood(self, requests, disable_tqdm: bool = False):
            args = [request.args for request in requests]
            total = len(args)
            if total == 0:
                return []
            # Every rank holds the full ordered request list (world_size=1), so
            # tokenization, grouping and the global padding dims below are all
            # computed identically on every rank -- a prerequisite for the shared
            # forward shapes that EP8 all-to-all requires.
            encoded = [
                split_context_continuation(self._tokenizer, context, continuation)
                for context, continuation in args
            ]
            groups = group_choices_by_context([context for context, _ in args])
            num_groups = len(groups)
            max_choices = max(len(group) for group in groups)
            # Pad only to the longest real (context+continuation), rounded up for
            # kernel alignment and capped at the training sequence_length -- far
            # cheaper than always padding 0-shot MCQ prompts to 4096.
            max_len = max(len(ctx) + len(cont) for ctx, cont in encoded)
            seq_length = min(self._sequence_length, round_up_to_multiple(max_len, _SEQ_ALIGN))
            # Clamp questions/forward so the MoE routes a bounded token count per
            # forward (qpf * max_choices * seq_length); derived from globals so
            # every rank agrees on qpf and the forward count (EP8 lockstep).
            effective_qpf = clamp_questions_per_forward(
                self._questions_per_forward,
                max_choices=max_choices,
                seq_length=seq_length,
                max_tokens_per_forward=self._max_tokens_per_forward,
            )
            groups_per_rank, qpf, forwards = plan_dp_forwards(
                num_groups, self._dp_rank, self._dp_size, effective_qpf
            )
            local = torch.zeros(
                groups_per_rank, max_choices, dtype=torch.float64, device=self._device
            )
            for slots in forwards:
                # Pack this forward: `qpf` questions, each padded to `max_choices`
                # rows by repeating its last choice, laid out row-major.
                batch_examples: list[tuple[list[int], list[int]]] = []
                for _slot, source_idx in slots:
                    choices = [encoded[m] for m in groups[source_idx]]
                    while len(choices) < max_choices:
                        choices.append(choices[-1])
                    batch_examples.extend(choices)
                logprobs = self._score_batch(batch_examples, seq_length=seq_length)
                logprobs = logprobs.view(qpf, max_choices)
                for row, (slot, _source_idx) in enumerate(slots):
                    if slot < groups_per_rank:  # drop tail/batch padding
                        local[slot] = logprobs[row].to(local.dtype)
            gathered = [torch.zeros_like(local) for _ in range(self._dp_size)]
            import torch.distributed as dist

            dist.all_gather(gathered, local, group=self._dp_group)
            results: list[tuple[float, bool]] = [None] * total  # type: ignore[list-item]
            for group_idx in range(num_groups):
                rank = group_idx % self._dp_size
                slot = group_idx // self._dp_size
                for choice_pos, request_idx in enumerate(groups[group_idx]):
                    logprob = float(gathered[rank][slot][choice_pos].item())
                    results[request_idx] = (logprob, True)
            return results

        def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
            raise NotImplementedError("online evaluation only supports MCQ loglikelihood tasks")

        def generate_until(self, requests, disable_tqdm: bool = False):
            raise NotImplementedError("online evaluation only supports MCQ loglikelihood tasks")

    return _MCoreEvalLM


# Public name; the actual class object is created lazily to avoid importing
# lm_eval on CPU-only nodes.
def MCoreEvalLM(*args, **kwargs):
    return _make_mcore_eval_lm_class()(*args, **kwargs)


def _capture_rng_state():
    state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:  # numpy is optional
        pass
    return state


def _restore_rng_state(state) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "numpy" in state:
        try:
            import numpy as np

            np.random.set_state(state["numpy"])
        except ImportError:  # numpy is optional
            pass


def _append_online_eval_record(
    output_dir: str, run_name: str, step: int, accuracies: dict[str, float]
) -> None:
    """Append one online-eval step's results to a live JSONL log (rank 0 only).

    Writes ``{output_dir}/{run_name}_online_eval.jsonl`` with one JSON record per
    eval step (``step`` + each task's accuracy), created/appended as training
    runs so the file can be tailed. Best-effort: a write failure must never abort
    training, so any error is swallowed after a warning.
    """
    import json
    import time as _time

    try:
        os.makedirs(output_dir, exist_ok=True)
        safe_name = run_name or "online_eval"
        path = os.path.join(output_dir, f"{safe_name}_online_eval.jsonl")
        record = {
            "run": run_name,
            "step": step,
            "unix_time": _time.time(),
            **{task: float(acc) for task, acc in accuracies.items()},
        }
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - best-effort logging
        print(f"[online_eval] WARNING: could not write result log: {exc}", flush=True)


def _extract_task_accuracy(task_manager, results: dict, task: str) -> float:
    """Read ``acc`` for a task/group, mirroring LMEvalEvaluator.summarize."""
    if task in task_manager.all_groups:
        metrics = results.get("groups", {}).get(task, {})
    else:
        metrics = results.get("results", {}).get(task, {})
    for key, value in metrics.items():
        if key == "alias":
            continue
        if key.split(",", 1)[0].strip() == "acc":
            return float(value)
    raise KeyError(f"no acc metric found for task {task!r}")


def run_online_evaluation(
    models,
    tokenizer,
    eval_config,
    *,
    sequence_length: int,
    device,
) -> dict[str, float]:
    """Score the resident model on the configured tasks; return ``{task: acc}``.

    Every rank participates (the DP all-gather inside ``loglikelihood`` is
    collective). The model is switched to ``eval()`` for the duration and the
    original ``training`` flag plus RNG state are restored on exit so training
    resumes bit-identically.
    """
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from lm_eval import simple_evaluate
    from lm_eval.tasks import TaskManager
    from megatron.core import parallel_state

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    dp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
    dp_rank = parallel_state.get_data_parallel_rank(with_context_parallel=True)
    dp_size = parallel_state.get_data_parallel_world_size(with_context_parallel=True)

    # Selective-position scoring applies the LM head via a local F.linear against
    # the full output weight; that is only correct when the vocab is not sharded.
    if parallel_state.get_tensor_model_parallel_world_size() != 1:
        raise NotImplementedError(
            "online evaluation selective-position scoring requires "
            "tensor_parallel==1 (the output weight is vocab-sharded under TP>1); "
            "extend with a vocab-parallel gather like _forward_generative_reranker"
        )

    rng_state = _capture_rng_state()
    was_training = models[0].training
    module = getattr(models[0], "module", models[0])
    models[0].eval()
    task_manager = TaskManager()
    accuracies: dict[str, float] = {}
    try:
        lm = MCoreEvalLM(
            module,
            tokenizer,
            sequence_length=sequence_length,
            pad_token_id=pad_token_id,
            dp_group=dp_group,
            dp_rank=dp_rank,
            dp_size=dp_size,
            device=device,
            questions_per_forward=getattr(eval_config, "batch_size", 1),
        )
        with torch.no_grad():
            for task in eval_config.tasks:
                results = simple_evaluate(
                    model=lm,
                    tasks=[task],
                    num_fewshot=eval_config.num_fewshot,
                    limit=eval_config.limit_for(task),
                    task_manager=task_manager,
                    random_seed=None,
                    numpy_random_seed=None,
                    torch_random_seed=None,
                    fewshot_random_seed=None,
                    bootstrap_iters=0,
                    verbosity="ERROR",
                )
                accuracies[task] = _extract_task_accuracy(task_manager, results, task)
    finally:
        if was_training:
            models[0].train()
        _restore_rng_state(rng_state)
    return accuracies
