import cv2
import hashlib
from collections import OrderedDict

class FrameCache:
    def __init__(self, max_size=100):
        """
        Инициализация кэша кадров

        Args:
            max_size (int): Максимальный размер кэша
        """
        self.max_size = max_size
        self.cache = OrderedDict()

    def _calculate_frame_hash(self, frame):
        """
        Вычисление хэша кадра

        Args:
            frame (numpy.ndarray): Входной кадр

        Returns:
            str: Хэш кадра
        """
        # Уменьшаем размер кадра для более быстрого хэширования
        downscaled = cv2.resize(frame, (32, 32))
        # Используем только каждый второй пиксель для еще большего ускорения
        return hashlib.md5(downscaled[::2, ::2].tobytes()).hexdigest()

    def get(self, frame):
        """
        Получение кадра из кэша

        Args:
            frame (numpy.ndarray): Входной кадр

        Returns:
            numpy.ndarray or None: Улучшенный кадр если найден в кэше, иначе None
        """
        frame_hash = self._calculate_frame_hash(frame)
        if frame_hash in self.cache:
            # Перемещаем элемент в конец (помечаем как недавно использованный)
            self.cache.move_to_end(frame_hash)
            return self.cache[frame_hash]
        return None

    def put(self, frame, enhanced_frame):
        """
        Добавление кадра в кэш

        Args:
            frame (numpy.ndarray): Исходный кадр
            enhanced_frame (numpy.ndarray): Улучшенный кадр
        """
        frame_hash = self._calculate_frame_hash(frame)

        # Если кэш переполнен, удаляем самый старый элемент
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)

        self.cache[frame_hash] = enhanced_frame