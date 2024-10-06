# from node.Node import Node
from src.Node import Node
from src.rover.subsystems.Instrument import Instrument


class Rover(Node):
    def __init__(self, config, logger) -> None:
        self.logger = logger
        self.config = config
        self.instrument = Instrument("Seisic detector", config, logger)
        super().__init__(config["rover"]["port"], "Rover", logger)

    def logic(self):
        self.instrument.logic()
