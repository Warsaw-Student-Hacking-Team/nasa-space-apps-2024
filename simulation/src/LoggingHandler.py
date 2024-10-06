# src/LoggingHandler.py
import logging
from rich.logging import RichHandler


class LoggingHandler:

    def apply_log_settings(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)  # Set the logging level to DEBUG

        # Create a RichHandler
        console_handler = RichHandler()
        console_handler.setLevel(
            logging.DEBUG
        )  # Ensure the handler also logs DEBUG level messages

        # Clear existing handlers to avoid duplicate logs
        if logger.hasHandlers():
            logger.handlers.clear()

        # Add the RichHandler to the logger
        logger.addHandler(console_handler)

        # Set the logging format
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[console_handler],
        )

        logging.warning("Hello, Warning!")
        return logger
