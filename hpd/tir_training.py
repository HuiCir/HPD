"""TIR three-view training core.

This module contains only the objective and schedule. Model loading, LoRA
construction, checkpointing, token caches, and distributed launch are left to
the caller so the algorithm can be moved between environments unchanged.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from .harness_replay import build_three_view_examples
from .types import ReplayBoundary, TextExample


VIEW_ORDER = ("native", "injected", "runtime")


def make_three_view_examples(boundary: ReplayBoundary) -> tuple[TextExample, ...]:
    """Expose the replay boundary as the three TIR supervision views."""

    return build_three_view_examples(boundary)


def validate_complete_triplets(
    examples: Iterable[TextExample],
) -> dict[tuple[str, int], dict[str, TextExample]]:
    """Validate that every boundary has exactly the three required views."""

    grouped: dict[tuple[str, int], dict[str, TextExample]] = defaultdict(dict)
    for example in examples:
        if example.view not in VIEW_ORDER:
            raise ValueError(f"unknown TIR view: {example.view}")
        if example.view in grouped[example.boundary_key]:
            raise ValueError(f"duplicate TIR view at {example.boundary_key}: {example.view}")
        grouped[example.boundary_key][example.view] = example

    expected = set(VIEW_ORDER)
    for key, by_view in grouped.items():
        missing = expected - set(by_view)
        if missing:
            raise ValueError(f"incomplete TIR triplet at {key}: missing {sorted(missing)}")
        if by_view["native"].target != by_view["injected"].target:
            raise ValueError(f"native/injected action targets differ at {key}")
    return dict(grouped)


class ThreeStageSchedule:
    """Deterministic one-view-per-update schedule.

    The default schedule reproduces the original balanced objective: Native,
    Injected, Runtime, repeating with equal frequency. ``lockstep=True`` keeps
    the three views of one boundary together for each cycle, which is useful
    for controlled ablations without changing the loss itself.
    """

    def __init__(
        self,
        examples: Iterable[TextExample],
        *,
        seed: int = 0,
        view_order: Sequence[str] = VIEW_ORDER,
        lockstep: bool = False,
    ) -> None:
        if tuple(view_order) != VIEW_ORDER:
            raise ValueError(f"TIR requires view order {VIEW_ORDER}")
        self.groups = validate_complete_triplets(examples)
        if not self.groups:
            raise ValueError("TIR schedule requires at least one complete triplet")
        self.seed = int(seed)
        self.lockstep = bool(lockstep)
        self._keys = sorted(self.groups)
        self._buckets = {
            view: [self.groups[key][view] for key in self._keys]
            for view in VIEW_ORDER
        }

    @property
    def triplet_count(self) -> int:
        return len(self._keys)

    def _permutation(self, view: str, cycle: int) -> list[int]:
        digest = hashlib.sha256(f"{self.seed}:{view}:{cycle}".encode()).digest()
        order = list(range(self.triplet_count))
        random.Random(int.from_bytes(digest[:8], "big")).shuffle(order)
        return order

    def at(self, index: int) -> TextExample:
        if index < 0:
            raise ValueError("schedule index must be non-negative")
        if self.lockstep:
            cycle, offset = divmod(index, self.triplet_count * len(VIEW_ORDER))
            selected = self._permutation("lockstep", cycle)[offset // len(VIEW_ORDER)]
            view = VIEW_ORDER[offset % len(VIEW_ORDER)]
            return self._buckets[view][selected]
        occurrence, view_index = divmod(index, len(VIEW_ORDER))
        view = VIEW_ORDER[view_index]
        cycle, position = divmod(occurrence, self.triplet_count)
        selected = self._permutation(view, cycle)[position]
        return self._buckets[view][selected]

    def cycle(self, index: int) -> tuple[TextExample, TextExample, TextExample]:
        """Return one boundary's three views in the configured view order."""

        if index < 0:
            raise ValueError("schedule index must be non-negative")
        cycle, position = divmod(index, self.triplet_count)
        selected = self._permutation("lockstep", cycle)[position]
        return tuple(self._buckets[view][selected] for view in VIEW_ORDER)  # type: ignore[return-value]


def tokenize_text_examples(
    examples: Iterable[TextExample],
    tokenizer: Any,
    *,
    eos_text: str = "",
    max_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """Encode complete examples without silently truncating context.

    A sample that exceeds ``max_tokens`` is rejected. The caller may choose a
    larger model context or filter the sample explicitly; this function never
    drops the beginning of a harness state.
    """

    encoded: list[dict[str, Any]] = []
    for example in examples:
        prompt_ids = tokenizer(
            example.prompt,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        target_ids = tokenizer(
            example.target + eos_text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        if not prompt_ids or not target_ids:
            raise ValueError(f"empty token segment: {example.example_id}")
        total = len(prompt_ids) + len(target_ids)
        if max_tokens is not None and total > max_tokens:
            raise ValueError(
                f"context budget exceeded for {example.example_id}: {total} > {max_tokens}"
            )
        encoded.append(
            {
                "id": example.example_id,
                "kind": example.view,
                "case_id": example.case_id,
                "turn_index": example.turn_index,
                "prompt_ids": list(prompt_ids),
                "completion_ids": list(target_ids),
                "metadata": dict(example.metadata),
            }
        )
    return encoded


def completion_only_nll(
    model: Any,
    prompt_ids: Sequence[int],
    completion_ids: Sequence[int],
    *,
    device: Any,
) -> Any:
    """Mean negative log-likelihood over completion tokens only.

    Prompt tokens provide context but receive no gradient target. The model is
    expected to return an object with ``.logits`` in the usual causal-LM form.
    """

    import torch
    import torch.nn.functional as F

    if not prompt_ids or not completion_ids:
        raise ValueError("prompt and completion must both contain tokens")
    full_ids = torch.tensor(
        [list(prompt_ids) + list(completion_ids)], dtype=torch.long, device=device
    )
    attention = torch.ones_like(full_ids)
    keep = len(completion_ids) + 1
    try:
        output = model(
            input_ids=full_ids,
            attention_mask=attention,
            use_cache=False,
            logits_to_keep=keep,
        )
    except TypeError:
        output = model(input_ids=full_ids, attention_mask=attention, use_cache=False)

    logits = output.logits
    if logits.shape[1] == full_ids.shape[1]:
        prediction = logits[:, len(prompt_ids) - 1 : -1, :]
    elif logits.shape[1] == keep:
        prediction = logits[:, :-1, :]
    else:
        raise ValueError(
            f"model returned {logits.shape[1]} positions; expected "
            f"{full_ids.shape[1]} or {keep}"
        )
    if prediction.shape[1] != len(completion_ids):
        raise ValueError(
            f"completion alignment mismatch: {prediction.shape[1]} vs {len(completion_ids)}"
        )
    targets = torch.tensor(
        [list(completion_ids)], dtype=torch.long, device=prediction.device
    )
    token_logprobs = F.log_softmax(prediction.float(), dim=-1).gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(0).squeeze(-1)
    return -token_logprobs.mean()


@dataclass(frozen=True)
class TIRStep:
    step: int
    view: str
    example_id: str
    loss: float


class TIRTrainer:
    """Small optimizer loop for the three-view TIR objective."""

    def __init__(
        self,
        model: Any,
        optimizer: Any,
        examples: Iterable[Mapping[str, Any]],
        *,
        device: Any,
        seed: int = 0,
        max_grad_norm: float | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.rows = [dict(row) for row in examples]
        self.schedule = ThreeStageSchedule(
            [
                TextExample(
                    example_id=str(row["id"]),
                    view=str(row["kind"]),
                    case_id=str(row["case_id"]),
                    turn_index=int(row["turn_index"]),
                    prompt="<encoded>",
                    target="<encoded>",
                    metadata=row.get("metadata", {}),
                )
                for row in self.rows
            ],
            seed=seed,
        )
        self.by_id = {str(row["id"]): row for row in self.rows}

    def step(self, step_index: int) -> TIRStep:
        row_ref = self.schedule.at(step_index)
        row = self.by_id[row_ref.example_id]
        self.model.train()
        loss = completion_only_nll(
            self.model,
            row["prompt_ids"],
            row["completion_ids"],
            device=self.device,
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.max_grad_norm is not None:
            import torch

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return TIRStep(
            step=step_index + 1,
            view=str(row["kind"]),
            example_id=str(row["id"]),
            loss=float(loss.detach().cpu()),
        )

    def train(self, steps: int) -> list[TIRStep]:
        if steps <= 0:
            raise ValueError("steps must be positive")
        return [self.step(index) for index in range(steps)]


__all__ = [
    "TIRStep",
    "TIRTrainer",
    "ThreeStageSchedule",
    "completion_only_nll",
    "make_three_view_examples",
    "tokenize_text_examples",
    "validate_complete_triplets",
]
