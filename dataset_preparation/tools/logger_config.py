import logging
import os
from datetime import datetime

def setup_logger(name, log_dir='logs'):
    """
    Sets up a logger with both file and console handlers.

    Args:
        name (str): Name of the logger.
        log_dir (str): Directory to store log files.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create the log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Create a log file name with the current date and time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"{current_time}.log")

    # Configure the logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create a file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger