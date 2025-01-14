import logging

def setup_logger():
    """Configure and return the logger instance"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('upscaler.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)