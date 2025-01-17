import logging
import os
from datetime import datetime


def setup_logger():
    """
    Configure and return the logger instance with timestamped files in the logs directory.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create logs directory if it doesn't exist
    logs_dir = 'logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # Generate timestamp for the log file name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(logs_dir, f'upscaler_{timestamp}.log')

    # Configure formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Configure file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Configure stream handler (console output)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    # Configure logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Remove any existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # Log initial message with file location
    logger.info(f"Log file created at: {os.path.abspath(log_file)}")

    return logger