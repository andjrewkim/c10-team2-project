from collections import defaultdict

from fusion.base import FusedResult, FusionStrategy
from sensors.base import SensorObservation


class WeightedAverageFusion(FusionStrategy):
    """Reference fusion implementation: confidence-weighted average.

    Each sensor type can carry a configurable weight.  The fused
    confidence is computed as:

        fused_conf = Σ(w_i * confidence_i) / Σ(w_i)

    where w_i = sensor_type_weight * observation.confidence.

    This is intentionally trivial — a real system would replace this
    with Bayesian, Dempster-Shafer, or ML-based fusion.
    """

    def __init__(self, type_weights: dict[str, float] | None = None) -> None:
        self._weights: dict[str, float] = defaultdict(
            lambda: 1.0,
            (type_weights or {}),
        )

    def fuse(self, observations: list[SensorObservation]) -> FusedResult:
        if not observations:
            return FusedResult()

        weighted_sum = 0.0
        total_weight = 0.0
        sensor_ids: list[str] = []

        for obs in observations:
            w = self._weights[obs.sensor_type] * obs.confidence
            weighted_sum += w * obs.confidence
            total_weight += w
            sensor_ids.append(obs.sensor_id)

        fused_conf = weighted_sum / total_weight if total_weight > 0 else 0.0

        return FusedResult(
            activity_label="unknown",
            confidence=min(fused_conf, 1.0),
            contributing_sensors=sensor_ids,
            meta={"strategy": "weighted_average", "n_observations": len(observations)},
        )

    def reset(self) -> None:
        pass
