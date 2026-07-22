from datetime import datetime, timezone

from fusion.base import FusedResult
from actions.base import ActionTrigger, ActionResult
from actions.console_action import ConsoleAction


def test_console_action_returns_action_result() -> None:
    result = FusedResult(
        activity_label="activity_a",
        confidence=0.85,
        timestamp=datetime.now(timezone.utc),
        contributing_sensors=["mock-001"],
    )
    action: ActionTrigger = ConsoleAction()
    action_result = action.evaluate(result)
    assert action_result is not None
    assert action_result.fired is True
    assert action_result.action_name == "ConsoleAction"


def test_console_action_below_threshold_returns_none() -> None:
    result = FusedResult(
        activity_label="activity_a",
        confidence=0.3,
        timestamp=datetime.now(timezone.utc),
        contributing_sensors=["mock-001"],
    )
    action = ConsoleAction(min_confidence=0.5)
    assert action.evaluate(result) is None


def test_hysteresis_stub_is_noop() -> None:
    result = FusedResult(
        activity_label="activity_b",
        confidence=0.9,
        timestamp=datetime.now(timezone.utc),
        contributing_sensors=["s1"],
    )
    action = ConsoleAction()
    output = action.apply_hysteresis(result)
    assert output is result
