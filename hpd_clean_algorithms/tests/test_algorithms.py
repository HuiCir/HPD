from __future__ import annotations

import unittest

from hpd_clean_algorithms.harness_replay import (
    ReplayError,
    build_three_view_examples,
    replay_trajectory,
)
from hpd_clean_algorithms.tir_training import ThreeStageSchedule, validate_complete_triplets
from hpd_clean_algorithms.types import (
    ModelAction,
    ReferenceTurn,
    ToolCall,
    ToolObservation,
    Trajectory,
)


class CounterHarness:
    def initialize(self, trajectory):
        return {"phase": "call", "observed": [], "done": False}

    def native_prompt(self, state, trajectory, turn_index):
        return f"task={trajectory.task}; turn={turn_index}; phase={state['phase']}"

    def runtime_state(self, state, trajectory, turn_index):
        return {
            "phase": state["phase"],
            "observed": list(state["observed"]),
            "turn": turn_index,
        }

    def validate_action(self, state, action):
        if state["phase"] == "call":
            if action.is_final or len(action.calls) != 1 or action.calls[0].name != "lookup":
                raise ValueError("lookup is required before final")
        elif not action.is_final:
            raise ValueError("final is required after lookup")

    def apply_action(self, state, action):
        if action.is_final:
            state["done"] = True
        else:
            state["phase"] = "result"

    def apply_observation(self, state, observation):
        state["observed"].append(observation.tool)
        state["phase"] = "final"

    def is_terminal(self, state):
        return state["done"]


def reference_trajectory() -> Trajectory:
    return Trajectory(
        case_id="case-1",
        task="look up a value",
        turns=(
            ReferenceTurn(
                action=ModelAction(calls=(ToolCall("lookup", {"key": "value"}),)),
                observations=(ToolObservation("lookup", "42"),),
            ),
            ReferenceTurn(action=ModelAction(final="The value is 42.")),
        ),
    )


class AlgorithmTests(unittest.TestCase):
    def test_replay_captures_pre_action_boundaries(self):
        boundaries = replay_trajectory(reference_trajectory(), CounterHarness())
        self.assertEqual(len(boundaries), 2)
        self.assertEqual(boundaries[0].runtime_state["phase"], "call")
        self.assertEqual(boundaries[1].runtime_state["observed"], ["lookup"])

    def test_replay_rejects_incomplete_episode(self):
        trajectory = Trajectory(
            case_id="unfinished",
            task="look up a value",
            turns=reference_trajectory().turns[:1],
        )
        with self.assertRaises(ReplayError):
            replay_trajectory(trajectory, CounterHarness())

    def test_schedule_is_balanced_and_triplets_are_complete(self):
        boundaries = replay_trajectory(reference_trajectory(), CounterHarness())
        examples = [
            view
            for boundary in boundaries
            for view in build_three_view_examples(boundary)
        ]
        groups = validate_complete_triplets(examples)
        self.assertEqual(len(groups), 2)
        schedule = ThreeStageSchedule(examples, seed=7)
        self.assertEqual(
            [schedule.at(index).view for index in range(6)],
            ["native", "injected", "runtime", "native", "injected", "runtime"],
        )
        lockstep = ThreeStageSchedule(examples, seed=7, lockstep=True)
        lockstep_keys = {lockstep.at(index).boundary_key for index in range(3)}
        self.assertEqual(len(lockstep_keys), 1)
        self.assertEqual(
            [lockstep.at(index).view for index in range(3)],
            ["native", "injected", "runtime"],
        )


if __name__ == "__main__":
    unittest.main()
