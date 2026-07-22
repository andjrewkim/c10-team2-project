import json
from datetime import timezone
from typing import Optional

from actions.base import ActionResult, ActionTrigger
from fusion.base import FusedResult


class ConsoleAction(ActionTrigger):
    """Trivial example action: log the fused result to stdout.

    Useful for verifying the pipeline works end-to-end.
    Replace with real actions (e.g. send Slack alert, toggle light).
    """

    def __init__(self, min_confidence: float = 0.0) -> None:
        self._min_confidence = min_confidence

    def evaluate(self, result: FusedResult) -> Optional[ActionResult]:
        if result.confidence < self._min_confidence:
            return None

        payload = {
            "activity_label": result.activity_label,
            "confidence": round(result.confidence, 4),
            "timestamp": result.timestamp.astimezone(timezone.utc).isoformat(),
            "contributing_sensors": result.contributing_sensors,
            "meta": result.meta,
        }
        print(f"[ACTION] ConsoleAction fired:\n{json.dumps(payload, indent=2)}")

        return ActionResult(
            fired=True,
            action_name="ConsoleAction",
            detail=payload,
        )
