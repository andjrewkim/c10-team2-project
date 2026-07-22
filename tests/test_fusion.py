import pytest

from datetime import datetime, timezone

from sensors.base import SensorObservation
from fusion.base import FusionStrategy, FusedResult
from fusion.weighted_average import WeightedAverageFusion


def make_obs(
    sensor_id: str,
    sensor_type: str,
    confidence: float,
) -> SensorObservation:
    return SensorObservation(
        sensor_id=sensor_id,
        sensor_type=sensor_type,
        timestamp=datetime.now(timezone.utc),
        observation={},
        confidence=confidence,
    )


class TestWeightedAverageFusion:
    def test_empty_observations_returns_empty_result(self) -> None:
        fuser: FusionStrategy = WeightedAverageFusion()
        result = fuser.fuse([])
        assert result.confidence == 0.0

    def test_single_observation_passes_through(self) -> None:
        fuser = WeightedAverageFusion()
        obs = make_obs("s1", "mock", 0.8)
        result = fuser.fuse([obs])
        assert result.confidence == pytest.approx(0.8, abs=1e-6)
        assert result.contributing_sensors == ["s1"]

    def test_average_of_equal_observations(self) -> None:
        fuser = WeightedAverageFusion()
        obs_list = [
            make_obs("s1", "mock", 0.9),
            make_obs("s2", "mock", 0.7),
        ]
        result = fuser.fuse(obs_list)
        # (0.9*0.9*0.9 + 0.7*0.7*0.7) / (0.9*0.9 + 0.7*0.7)
        assert result.confidence == pytest.approx(0.8125, abs=1e-6)

    def test_type_weights_affect_fusion(self) -> None:
        fuser = WeightedAverageFusion(type_weights={"a": 2.0, "b": 0.5})
        obs_list = [
            make_obs("s1", "a", 1.0),
            make_obs("s2", "b", 0.0),
        ]
        result = fuser.fuse(obs_list)
        # weighted_sum = (2*1)*1 + (0.5*0)*0 = 2
        # total_weight = 2*1 + 0.5*0 = 2
        assert result.confidence == pytest.approx(1.0, abs=1e-6)

    def test_reset_does_not_raise(self) -> None:
        fuser = WeightedAverageFusion()
        fuser.reset()

    def test_fused_result_contract(self) -> None:
        fuser = WeightedAverageFusion()
        obs = make_obs("s1", "mock", 0.75)
        result = fuser.fuse([obs])
        assert isinstance(result, FusedResult)
        assert isinstance(result.activity_label, str)
        assert isinstance(result.confidence, float)
        assert isinstance(result.contributing_sensors, list)
        assert isinstance(result.meta, dict)
        assert "strategy" in result.meta
