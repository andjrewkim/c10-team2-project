"""mmWave radar driver (TI IWR6843 / IWR1843, UART/SDK point cloud).

Expected hardware: TI IWR6843 or similar mmWave sensor over UART.
Data port carries TLV packets; CLI port accepts configuration commands.

TODO: Replace stub body with real UART TLV packet parsing via TI mmWave SDK.
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timezone

from sensors.base import BaseSensor, SensorObservation


class MmWaveRadarSensor(BaseSensor):
    """Driver for a mmWave radar sensor (TI IWR series or equivalent).

    Parameters
    ----------
    sensor_id : str
        Unique identifier for this radar unit.
    serial_port : str
        UART device path (data output from the radar).
    cli_port : str
        UART device path for CLI configuration.
    baudrate_data : int
        Baud rate for the data output port.
    baudrate_cli : int
        Baud rate for the CLI port.
    """

    def __init__(
        self,
        sensor_id: str,
        serial_port: str = "/dev/ttyUSB0",
        cli_port: str = "/dev/ttyUSB1",
        baudrate_data: int = 921600,
        baudrate_cli: int = 115200,
    ) -> None:
        super().__init__(sensor_id=sensor_id, sensor_type="mmwave")
        self.serial_port = serial_port
        self.cli_port = cli_port
        self.baudrate_data = baudrate_data
        self.baudrate_cli = baudrate_cli

    def read(self) -> list[SensorObservation]:
        """Blocking read: parse one frame of radar data.

        Expected observation dict (TI TLV packet types):
            num_detected_obj  : int
            objects           : list[dict] — point cloud (x, y, z, doppler, snr)
            range_profile     : list[float] | None
        """
        n_objects = random.randint(0, 6)
        objects = []
        for _ in range(n_objects):
            objects.append({
                "x": random.uniform(-3.0, 3.0),
                "y": random.uniform(0.5, 6.0),
                "z": random.uniform(-0.5, 1.5),
                "doppler": random.uniform(-2.0, 2.0),
                "snr": random.uniform(5.0, 40.0),
            })
        obs = {"num_detected_obj": n_objects, "objects": objects, "range_profile": None}
        return [
            SensorObservation(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                timestamp=datetime.now(timezone.utc),
                observation=obs,
                confidence=0.8 if n_objects > 0 else 0.3,
                metadata={"serial_port": self.serial_port, "cli_port": self.cli_port, "num_objects": n_objects},
            )
        ]


__all__ = ["MmWaveRadarSensor"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mmWave radar self-test")
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--cli-port", default="/dev/ttyUSB1")
    args = parser.parse_args()
    sensor = MmWaveRadarSensor(sensor_id="mmwave-self-test", serial_port=args.port, cli_port=args.cli_port)
    for o in sensor.read():
        print(f"[{o.timestamp.isoformat()}] {o.sensor_type}/{o.sensor_id}  objects={o.observation.get('num_detected_obj')}  conf={o.confidence}")
