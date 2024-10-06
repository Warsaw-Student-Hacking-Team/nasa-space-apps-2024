import time

from src.ground_station.GroundStation import GroundStation
from src.rover.Rover import Rover
from src.LoggingHandler import LoggingHandler
from src.Node import NodeStatus

from datetime import datetime, timedelta


class Simulation:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.logger = LoggingHandler().apply_log_settings()
        self.ground_station = GroundStation(config, self.logger)
        self.rover = Rover(config, self.logger)

        # counters
        self.counter = 0
        self.sim_time = self.get_nanoseconds_dt(self.config["sim_start_time"])
        self.sim_end = self.get_nanoseconds_dt(self.config["sim_end_time"])
        self.sim_step = 1 / self.config["rover"]["sample_rate"]

    def run(self) -> None:
        print("Simulation started")

        while self.sim_time < self.sim_end:

            start_interation = time.time()

            if self.if_connection_window(self.sim_time):
                # Starting connection
                self.rover.connect_to(self.ground_station)
                self.ground_station.connect_to(self.rover)
            elif (
                not self.if_connection_window(self.sim_time)
                and self.rover.status == NodeStatus.OPEN
            ):
                self.rover.stop_connection()
                self.ground_station.stop_connection()

            # tasks
            if self.ground_station.status == NodeStatus.OPEN:
                self.ground_station.send_message(
                    f"Hello rover! I have counter: {self.counter}!"
                )
                self.rover.receive_message()
                self.rover.send_message(f"Hello GS! I have counter: {self.counter}!")
                self.ground_station.receive_message()

            # sleep
            end_iteration = time.time()
            iteration_time = end_iteration - start_interation

            if not self.config["real_time"]:
                self.counter += 1
                self.sim_time += timedelta(seconds=self.sim_step)
                self.logger.info(f"Current time: {self.sim_time}")
                time.sleep(0.1)
            else:
                time.sleep(max(0, self.sim_step - iteration_time))
                self.counter += 1
                self.sim_time += timedelta(seconds=self.sim_step)
                self.logger.info(f"Current time: {self.sim_time}")

            if iteration_time > self.sim_step:
                self.logger.critical(
                    f"Iteration time {iteration_time} [s] exceeded time step {self.sim_step} [s]"
                )

    def if_connection_window(self, now):
        for window in self.config["connection_windows"]:
            start = self.get_nanoseconds_dt(window["start"])
            end = self.get_nanoseconds_dt(window["end"])

            if start <= now <= end:
                self.logger.info(f"In window: {window}")
                return True

    def get_nanoseconds_dt(self, time_str):
        import decimal

        # Split into datetime and nanosecond parts
        dt_part, ns_part = time_str[:-1].split(".")  # Remove 'Z' at the end

        # Parse the datetime part (up to microseconds)
        dt = datetime.strptime(dt_part, "%Y-%m-%dT%H:%M:%S")

        # Calculate the nanoseconds (convert to decimal for precision)
        nanoseconds = decimal.Decimal(f"0.{ns_part}")

        # Add the nanoseconds to the datetime
        total_seconds = nanoseconds * decimal.Decimal(1e-9)
        return dt + timedelta(seconds=float(total_seconds))
