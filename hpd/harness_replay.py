"""Deterministic harness replay and three-view supervision construction.

The adapter is the only harness-specific component. It reads the state at the
same decision boundary that a live harness would expose, validates the
reference action, and applies the recorded observations. No source-code
inspection or external service is required by this module.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Generic, Mapping, Protocol, TypeVar

from .types import (
    ModelAction,
    ReplayBoundary,
    TextExample,
    ToolObservation,
    Trajectory,
    canonical_action,
    canonical_json,
)


StateT = TypeVar("StateT")


class HarnessAdapter(Protocol, Generic[StateT]):
    """Harness operations needed for deterministic reference replay."""

    def initialize(self, trajectory: Trajectory) -> StateT:
        """Create a fresh mutable runtime state for one episode."""

    def native_prompt(self, state: StateT, trajectory: Trajectory, turn_index: int) -> str:
        """Render the prompt that the live harness sends at this boundary."""

    def runtime_state(
        self,
        state: StateT,
        trajectory: Trajectory,
        turn_index: int,
    ) -> Mapping[str, Any]:
        """Read and normalize only the control state relevant to the policy."""

    def validate_action(self, state: StateT, action: ModelAction) -> None:
        """Reject an action that is illegal at the current harness state."""

    def apply_action(self, state: StateT, action: ModelAction) -> None:
        """Advance the harness through the reference action."""

    def apply_observation(self, state: StateT, observation: ToolObservation) -> None:
        """Apply one recorded tool result in the order seen by the harness."""

    def is_terminal(self, state: StateT) -> bool:
        """Report whether the harness has reached its terminal state."""


class ReplayError(RuntimeError):
    """Raised when a trajectory is inconsistent with the target harness."""


def replay_trajectory(
    trajectory: Trajectory,
    adapter: HarnessAdapter[StateT],
) -> tuple[ReplayBoundary, ...]:
    """Replay a complete reference episode and return all decision boundaries.

    The runtime snapshot is captured before each action. Actions and tool
    observations are then applied in their original order. A trajectory must
    end in the adapter's terminal state; extra turns, missing observations, or
    illegal actions are rejected instead of silently becoming supervision.
    """

    state = adapter.initialize(trajectory)
    boundaries: list[ReplayBoundary] = []

    for turn_index, turn in enumerate(trajectory.turns):
        if adapter.is_terminal(state):
            raise ReplayError(
                f"{trajectory.case_id}: extra turn {turn_index} after terminal state"
            )

        runtime_state = deepcopy(
            dict(adapter.runtime_state(state, trajectory, turn_index))
        )
        native_prompt = adapter.native_prompt(state, trajectory, turn_index)
        if not native_prompt:
            raise ReplayError(
                f"{trajectory.case_id}: empty native prompt at turn {turn_index}"
            )

        try:
            adapter.validate_action(state, turn.action)
            adapter.apply_action(state, turn.action)
            for observation in turn.observations:
                adapter.apply_observation(state, observation)
        except Exception as exc:  # adapters may use domain-specific exceptions
            raise ReplayError(
                f"{trajectory.case_id}: invalid transition at turn {turn_index}: {exc}"
            ) from exc

        boundaries.append(
            ReplayBoundary(
                case_id=trajectory.case_id,
                turn_index=turn_index,
                native_prompt=native_prompt,
                runtime_state=runtime_state,
                action=turn.action,
                metadata=dict(trajectory.metadata),
            )
        )

    if not adapter.is_terminal(state):
        raise ReplayError(f"{trajectory.case_id}: replay ended before terminal state")
    return tuple(boundaries)


def build_three_view_examples(
    boundary: ReplayBoundary,
    *,
    injection_header: str = "## Harness runtime state",
    probe_header: str = "Recover the harness control state as canonical JSON.",
) -> tuple[TextExample, TextExample, TextExample]:
    """Create Native, Injected, and Runtime examples for one boundary.

    Native and Injected predict the same canonical action. Runtime predicts a
    canonical, normalized control-state document from the native context. The
    latter is a state-reconstruction target, not an alternate task answer.
    """

    runtime_document = canonical_json(boundary.runtime_state)
    action_target = canonical_action(boundary.action)
    injected_prompt = (
        f"{boundary.native_prompt}\n\n{injection_header}\n{runtime_document}"
    )
    runtime_prompt = f"{probe_header}\n\n{boundary.native_prompt}"
    common = dict(
        case_id=boundary.case_id,
        turn_index=boundary.turn_index,
        metadata=dict(boundary.metadata),
    )
    return (
        TextExample(
            example_id=f"native:{boundary.case_id}:{boundary.turn_index}",
            view="native",
            prompt=boundary.native_prompt,
            target=action_target,
            **common,
        ),
        TextExample(
            example_id=f"injected:{boundary.case_id}:{boundary.turn_index}",
            view="injected",
            prompt=injected_prompt,
            target=action_target,
            **common,
        ),
        TextExample(
            example_id=f"runtime:{boundary.case_id}:{boundary.turn_index}",
            view="runtime",
            prompt=runtime_prompt,
            target=runtime_document,
            **common,
        ),
    )


__all__ = [
    "HarnessAdapter",
    "ReplayError",
    "build_three_view_examples",
    "replay_trajectory",
]
