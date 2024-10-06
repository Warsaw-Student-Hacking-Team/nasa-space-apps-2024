import pandas as pd
from datetime import datetime
import os
from os import path
import numpy as np

from obspy.core import read
from obspy.core.stream import Stream
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.inventory import read_inventory


class TelemetryLoader:
    """
    Class for loading telemetry data from mseed files
    """

    def __init__(self, config, logger) -> None:
        self.logger = logger
        self.config = config
        self.telemetry = self.read_mseed_data(config["rover"]["seismic_file_name"])
        self.stream: Stream = None

        self.current_tick_epoch: datetime = None
        self.logger.info("TelemetryLoader initialized")

    def fetch_telemetry(self, tick_time: datetime):
        return {"measurement": self.telemetry["measurement"].iloc[tick_time]}

    def tock(self):
        self.logger.info(f"TelemetryLoader tocked at {self.current_tick_epoch}")

    def read_mseed_data(self, file_name: str):

        path_to_telemetry = path.join(os.getcwd(), "resources", file_name)
        self.logger.debug(f"Reading mseed data from {path_to_telemetry}")

        # read mseed files with obspy
        self.stream: Stream = read(path_to_telemetry)

        tr = self.stream.traces[0].copy()

        tr_times = tr.times()
        tr_data = tr.data

        self.logger.debug(f"File {file_name} with stats: {self.stream[0].stats}")

        # create pandas dataframe
        return pd.DataFrame(
            {
                "time": tr_times,
                "measurement": tr_data,
            }
        )
