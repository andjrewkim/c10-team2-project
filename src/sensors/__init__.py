from src.sensors.base import SensorObservation, BaseSensor
from src.sensors.mock_sensor import MockSensor

# Lab integration subpackage — real signal-processing algorithms ported
# from the COSMOS lab repository (mmwave/ folder).  These are imported
# lazily by the driver modules so they never break when numpy/pyserial
# are absent.
__all__ = ["SensorObservation", "BaseSensor", "MockSensor"]
