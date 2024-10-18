import logging
import os
from datetime import datetime

def setup_logger(name, log_dir='logs'):
    # Создаем директорию для логов, если она не существует
    os.makedirs(log_dir, exist_ok=True)

    # Создаем имя файла лога с текущей датой и временем
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"{current_time}.log")

    # Настраиваем логгер
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Создаем обработчик файла
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Создаем обработчик консоли
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Создаем форматтер и добавляем его к обработчикам
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Добавляем обработчики к логгеру
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger