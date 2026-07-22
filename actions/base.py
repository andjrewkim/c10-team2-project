from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from fusion.base import FusedResult


@dataclass
class ActionResult:
    """Record of what an action trigger decided to do."""

    fired: bool = False
    action_name: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


class ActionTrigger(ABC):
    """Decides whether a fused result warrants an action.

    The architecture expects a chain or registry of triggers so that
    different confidence levels / activity labels can map to different
    actions without hardcoding.
    """

    @abstractmethod
    def evaluate(self, result: FusedResult) -> Optional[ActionResult]:
        """Inspect a fused result and optionally fire an action.

        Parameters
        ----------
        result : FusedResult
            Output from a FusionStrategy.

        Returns
        -------
        Optional[ActionResult]
            None if no action should be taken.
        """
        ...

    # ------------------------------------------------------------------
    # Hysteresis / temporal smoothing hook
    # ------------------------------------------------------------------
    # Override in a subclass to prevent rapid on-off toggling of
    # actions (a known best-practice for activity-detection systems).
    # ------------------------------------------------------------------

    def apply_hysteresis(self, result: FusedResult) -> FusedResult:
        """Stub — override to smooth noisy fused results.

        The default implementation is a no-op pass-through.
        """
        return result
