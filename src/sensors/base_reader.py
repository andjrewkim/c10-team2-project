from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Reading:
    sensor_id: str
    sensor_type: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    data: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0


class BaseReader(ABC):
    sensor_id: str
    sensor_type: str

    def __init__(self, sensor_id: str, sensor_type: str) -> None:
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def read(self) -> Reading: ...

    @abstractmethod
    def stop(self) -> None: ...
