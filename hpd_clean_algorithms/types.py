"""Small public data contracts shared by replay and training.

The contracts deliberately contain no model-loader, filesystem, cluster, or
service configuration. A harness supplies its own adapter and state objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ToolCall:
    """One canonical tool call in a reference trajectory."""

    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("tool call name must not be empty")


@dataclass(frozen=True)
class ModelAction:
    """A transport action: a call wave or a final response."""

    calls: tuple[ToolCall, ...] = ()
    final: str | None = None

    def __post_init__(self) -> None:
        if self.final is not None and self.calls:
            raise ValueError("an action cannot contain both calls and a final response")
        if self.final is not None and not isinstance(self.final, str):
            raise TypeError("final response must be text")

    @property
    def is_final(self) -> bool:
        return self.final is not None


@dataclass(frozen=True)
class ToolObservation:
    """One tool result observed by the harness after a call wave."""

    tool: str
    content: str
    is_error: bool = False

    def __post_init__(self) -> None:
        if not self.tool.strip():
            raise ValueError("tool observation name must not be empty")


@dataclass(frozen=True)
class ReferenceTurn:
    """One reference action plus the observations produced by that action."""

    action: ModelAction
    observations: tuple[ToolObservation, ...] = ()

    def __post_init__(self) -> None:
        if self.action.is_final and self.observations:
            raise ValueError("a final response cannot produce tool observations")


@dataclass(frozen=True)
class Trajectory:
    """A complete task episode in harness-neutral form."""

    case_id: str
    task: str
    turns: tuple[ReferenceTurn, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("trajectory case_id must not be empty")
        if not self.turns:
            raise ValueError("trajectory must contain at least one turn")


@dataclass(frozen=True)
class ReplayBoundary:
    """The deterministic state immediately before one model decision."""

    case_id: str
    turn_index: int
    native_prompt: str
    runtime_state: Mapping[str, Any]
    action: ModelAction
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, int]:
        return self.case_id, self.turn_index


@dataclass(frozen=True)
class TextExample:
    """One text-to-text supervision item before tokenizer encoding."""

    example_id: str
    view: str
    case_id: str
    turn_index: int
    prompt: str
    target: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def boundary_key(self) -> tuple[str, int]:
        return self.case_id, self.turn_index


def canonical_action(action: ModelAction) -> str:
    """Serialize an action without relying on dictionary insertion order."""

    import json

    if action.is_final:
        return json.dumps({"final": action.final}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return json.dumps(
        {
            "calls": [
                {"tool": call.name, "args": dict(call.arguments)}
                for call in action.calls
            ]
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_json(value: Any) -> str:
    """Stable JSON for runtime targets and injected state documents."""

    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


__all__ = [
    "ModelAction",
    "ReferenceTurn",
    "ReplayBoundary",
    "TextExample",
    "ToolCall",
    "ToolObservation",
    "Trajectory",
    "canonical_action",
    "canonical_json",
]
