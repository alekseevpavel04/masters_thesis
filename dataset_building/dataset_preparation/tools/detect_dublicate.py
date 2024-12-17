import os
from PIL import Image
import imagehash
from tools.logger_config import setup_logger 

class ImageDeduplicator:
    def __init__(self, output_dir="output_images", threshold=5):
        self.output_dir = output_dir
        self.threshold = threshold
        self.hashes = {}
        self.duplicates = set()  # Используем set для уникальных дубликатов
        self.logger = setup_logger("ImageDeduplicator")  # Настраиваем логгер

    def find_similar_images(self):
        """Находит похожие изображения в указанной папке."""
        self.logger.info("Начинаем поиск похожих изображений.")
        for filename in os.listdir(self.output_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_path = os.path.join(self.output_dir, filename)
                try:
                    image = Image.open(image_path)
                    hash_value = imagehash.phash(image)

                    # Проверка на наличие похожих изображений
                    for existing_hash in self.hashes:
                        if hash_value - existing_hash < self.threshold:
                            # Добавляем в множество уникальных дубликатов
                            self.duplicates.add((filename, self.hashes[existing_hash]))

                    # Сохраняем хэш изображения
                    self.hashes[hash_value] = filename
                except Exception as e:
                    self.logger.error(f"Ошибка при обработке изображения {filename}: {e}")

    def remove_duplicates(self):
        """Удаляет дубликаты изображений."""
        if not self.duplicates:
            self.logger.info("Нет дубликатов для удаления.")
            return

        total_images = len(self.hashes)  # Общее количество изображений
        removed_count = 0  # Количество удаленных дубликатов

        for img1, img2 in self.duplicates:
            img_path = os.path.join(self.output_dir, img2)
            if os.path.exists(img_path):  # Проверка на существование файла
                # self.logger.info(f"Удаляем дубликат: {img_path}")
                os.remove(img_path)
                removed_count += 1  # Увеличиваем счетчик удаленных изображений
            else:
                pass
                # self.logger.warning(f"Файл не найден, пропускаем: {img_path}")

        return total_images, removed_count  # Возвращаем общее количество и количество удаленных изображений

    def deduplicate_images(self):
        """Основной метод для поиска и удаления дубликатов."""
        self.find_similar_images()
        total_images, removed_count = self.remove_duplicates()
        self.logger.info("Процесс удаления дубликатов завершен.")

        # Расчет доли удаленных изображений
        if total_images > 0:
            removal_fraction = removed_count / total_images
            self.logger.info(f"Удалено {removed_count} дубликатов из {total_images} изображений. Доля удаленных: {removal_fraction:.2%}")
        else:
            self.logger.info("Нет изображений для обработки.")
