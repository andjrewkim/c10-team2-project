import random
from datetime import datetime, timezone

from sensors.base import BaseSensor, SensorObservation


class MockSensor(BaseSensor):
    """Trivial simulated sensor that publishes random confidences.

    Useful for end-to-end testing of the pipeline without any
    real hardware.  Replace with a real sensor subclass when ready.
    """

    def __init__(
        self,
        sensor_id: str = "mock-001",
        sensor_type: str = "mock",
        min_confidence: float = 0.0,
        max_confidence: float = 1.0,
    ) -> None:
        super().__init__(sensor_id=sensor_id, sensor_type=sensor_type)
        self._min = min_confidence
        self._max = max_confidence

    def read(self) -> list[SensorObservation]:
        observation = SensorObservation(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            timestamp=datetime.now(timezone.utc),
            observation={"raw": random.random()},
            confidence=random.uniform(self._min, self._max),
            metadata={"mock": True},
        )
        return [observation]
