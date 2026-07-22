from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sensors.base import SensorObservation


@dataclass
class FusedResult:
    """Output of a fusion strategy.

    Fields
    ------
    activity_label : str
        Placeholder for the detected activity (e.g. "activity_a").
        Concrete labels are injected by the application config, not
        hardcoded in this project.
    confidence : float
        Aggregated confidence in [0.0, 1.0].
    timestamp : datetime
        When the fused result was produced.
    contributing_sensors : list[str]
        sensor_ids that fed into this result.
    meta : dict
        Strategy-specific diagnostics (e.g. number of sources fused).
    """

    activity_label: str = ""
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    contributing_sensors: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


class FusionStrategy(ABC):
    """Pluggable fusion algorithm.

    Every fusion strategy receives a sequence of sensor observations
    and returns a single FusedResult.  Subclasses are free to maintain
    internal state (e.g. sliding windows, Bayesian priors) across calls.
    """

    @abstractmethod
    def fuse(self, observations: list[SensorObservation]) -> FusedResult:
        """Combine a batch of observations into one fused result.

        Parameters
        ----------
        observations : list[SensorObservation]
            Observations collected in the current fusion cycle (may be
            from different sensor types and locations).

        Returns
        -------
        FusedResult
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear any internal state (e.g. for online/streaming strategies)."""
        ...
