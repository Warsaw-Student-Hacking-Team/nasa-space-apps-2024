import zmq
from enum import Enum


class NodeStatus(Enum):
    OPEN = 1
    CLOSE = 0
    DEAD = -1


class Node:
    def __init__(self, port, name, logger) -> None:
        """_summary_

        Args:
            port (int): _description_
            name (str): _description_
            status (NodeStatus): _description_
            type (_type_): zmq publisher or subscriber
        """

        # This socket is for sending messages
        self.port = port
        self.name = name
        self.logger = logger

        self.context_send = zmq.Context()
        self.socket_sending = self.context_send.socket(zmq.PUB)

        self.logger.debug(f"{self.name} node created")

        self.socket_sending.bind(f"tcp://127.0.0.1:{self.port}")
        self.socket_sending.send_string(f"init message from {self.name}")
        self.logger.debug(f"Binded to tcp://127.0.0.1:{self.port}")
        self.status = None

        self.context_receive = zmq.Context()
        self.socket_receive = self.context_receive.socket(zmq.SUB)

    # src/Node.py
    def connect_to(self, target: "Node") -> None:
        """_summary_

        Args:
            target (Node): _description_
        """

        if target.status != NodeStatus.OPEN:
            # self.logger.info(f"{target.name} is listening to port {self.get_port}")
            self.socket_receive.connect(f"tcp://127.0.0.1:{target.port}")
            self.logger.info(f"{self.name} Connected to tcp://127.0.0.1:{target.port}")

            self.socket_receive.setsockopt_string(zmq.SUBSCRIBE, "")

            target.status = NodeStatus.OPEN

        else:
            self.logger.error(f"Connection to {target.name} failed")

    def stop_connection(self) -> None:
        """_summary_"""

        if self.status == NodeStatus.OPEN:
            self.socket_sending.close()
            self.status = NodeStatus.CLOSE
            self.logger.info(f"Connection to {self.name} closed")
        else:
            self.logger.error(f"Connection to {self.name} already closed")

    def send_message(self, message: str) -> None:
        """_summary_

        Args:
            message (str): _description_
        """

        if self.status == NodeStatus.OPEN:
            self.socket_sending.send_string(message)
            self.logger.info(f"Message sent from {self.socket_sending}")
            self.logger.info(f"Message sent from {self.name} : \n\t {message}")
        else:
            self.logger.error(f"Failed to send message from {self.name}")

    # src/Node.py
    # src/Node.py

    def receive_message(self) -> str:
        import traceback

        """_summary_
        Returns:
            str: _description_
        """
        if self.status == NodeStatus.OPEN:
            self.logger.info(f"Waiting for message by {self.socket_receive}")
            message = self.socket_receive.recv_string()
            self.logger.info(f"Message received by {self.name} : \n\t{message}")
            return message
        else:
            self.logger.error(f"Failed to receive message by {self.name}")
            return ""

    def ping(self) -> str:
        self.logger.info(f"Ping from {self.name} - Hello")

    def get_port(self):
        return self.port
