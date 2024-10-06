from datetime import datetime
from src.TelemetryLoader import TelemetryLoader
import threading
import time


class Event:
    def __init__(self, name, start_epoch, end_epoch) -> None:
        self.name = name
        self.start_epoch: datetime = start_epoch
        self.end_epoch: datetime = end_epoch
        self.record = None

    def __str__(self) -> str:
        return f"{self.name}: {self.data}"


class Instrument:
    """
    Instrument class for the rover subsystems. It is responsible for processing telemetry data and recording events
    """

    def __init__(self, name, config, logger) -> None:
        self.name = name
        self.config = config
        self.loader = TelemetryLoader(config, logger)
        self.event_list: list = [Event]

    def start_data_processing(self) -> None:
        """Processing data"""
        # Thread start
        thread = threading.Thread(target=self.handle)
        thread.start()

    def handle(self):
        # Start model calculations
        pass

    def record_event(self, event: Event) -> None:
        self.event_list.append(event)

    def logic(self):
        self.loader.tock()
