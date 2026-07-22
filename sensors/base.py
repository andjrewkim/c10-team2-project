from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class SensorObservation:
    """Universal observation contract every sensor node must publish.

    Fields
    ------
    sensor_id : str
        Unique identifier for the physical/logical sensor instance.
    sensor_type : str
        Semantic type label (e.g. "mock", "pir", "mmwave"). Never hardcoded
        in fusion or action code — treated as an opaque discriminant.
    timestamp : datetime
        UTC timestamp when the observation was captured.
    observation : Any
        The raw or preprocessed measurement value. Structure is
        sensor-specific; fusion strategies must document what they expect.
    confidence : float
        Probability or certainty in [0.0, 1.0] that the observation is
        accurate. 1.0 = certain, 0.0 = no confidence.
    metadata : dict[str, Any]
        Extensible bag of auxiliary info (e.g. firmware version,
        battery level, signal strength). Never required for core flow.
    """

    sensor_id: str
    sensor_type: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    observation: Any = None
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseSensor(ABC):
    """Abstract sensor node.

    Subclasses implement `read()` which returns zero or more
    SensorObservation instances.  The transport layer is responsible
    for publishing whatever `read()` produces — the sensor itself
    never touches MQTT or any other transport.
    """

    sensor_id: str
    sensor_type: str

    def __init__(self, sensor_id: str, sensor_type: str) -> None:
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type

    @abstractmethod
    def read(self) -> list[SensorObservation]:
        """Blocking read from the sensor hardware or simulation.

        Returns
        -------
        list[SensorObservation]
            One or more observations captured in a single read cycle.
            May be empty if no data is available.
        """
        ...
