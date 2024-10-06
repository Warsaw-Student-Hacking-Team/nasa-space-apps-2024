from src.Node import Node


class GroundStation(Node):
    def __init__(self, config, logger) -> None:
        self.logger = logger
        self.config = config
        super().__init__(config["ground_station"]["port"], "GroundStation", logger)

    def logic(self):
        pass
