import logging
import os

# Create logs directory if not exists
if not os.path.exists("logs"):
    os.makedirs("logs")


def get_logger(name, log_file):
    """Creates and returns a custom logger for different modules."""

    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture all logs

    # File Handler
    file_handler = logging.FileHandler(f"logs/{log_file}")
    file_handler.setLevel(logging.DEBUG)  # Capture all logs in file

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Show only INFO+ logs in console

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add Handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
