import os
import cv2
import torch
import numpy as np
from time import time
import logging
from datetime import datetime
from moviepy.editor import VideoFileClip, AudioFileClip
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from tqdm import tqdm
import sys
from contextlib import contextmanager
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

sys.path.append('../..')
from src.model import RRDBNet

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('upscaler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VideoUpscaler:
    def __init__(self, model_path='model/RealESRGAN_final.pth', batch_size=1, num_workers=4):
        """
        Инициализация апскейлера

        Args:
            model_path (str): Путь к модели
            batch_size (int): Размер батча для обработки кадров
            num_workers (int): Количество рабочих потоков
        """
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Инициализация устройства
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Загрузка модели
        try:
            self.model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=2)
            loadnet = torch.load(model_path, map_location=self.device, weights_only=True)

            if 'params_ema' in loadnet:
                self.model.load_state_dict(loadnet['params_ema'])
            else:
                self.model.load_state_dict(loadnet)

            self.model.eval()
            self.model = self.model.to(self.device)
            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    @contextmanager
    def video_context(self, path):
        """Контекстный менеджер для безопасной работы с видео"""
        video = None
        try:
            video = VideoFileClip(path)
            yield video
        finally:
            if video is not None:
                video.close()

    def enhance_frame(self, img):
        """
        Улучшение одного кадра

        Args:
            img (numpy.ndarray): Входное изображение

        Returns:
            numpy.ndarray: Улучшенное изображение
        """
        try:
            # Предобработка
            img = img.astype(np.float32) / 255.
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            img = img.to(self.device)

            # Инференс
            with torch.no_grad():
                output = self.model(img)

            # Постобработка
            output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
            output = (output * 255.0).clip(0, 255).astype(np.uint8)

            # Очистка CUDA кэша
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

            return output

        except Exception as e:
            logger.error(f"Error in enhance_frame: {str(e)}")
            raise

    def enhance_batch(self, frames):
        """
        Улучшение батча кадров

        Args:
            frames (list): Список кадров

        Returns:
            list: Список улучшенных кадров
        """
        try:
            # Подготовка батча
            batch = np.array(frames).astype(np.float32) / 255.
            batch = torch.from_numpy(batch).permute(0, 3, 1, 2)
            batch = batch.to(self.device)

            # Инференс
            with torch.no_grad():
                output = self.model(batch)

            # Постобработка
            output = output.permute(0, 2, 3, 1).cpu().numpy()
            output = (output * 255.0).clip(0, 255).astype(np.uint8)

            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

            return list(output)

        except Exception as e:
            logger.error(f"Error in enhance_batch: {str(e)}")
            raise

    def extract_audio(self, video, temp_audio_path):
        """
        Извлечение аудио из видео

        Args:
            video (VideoFileClip): Видео файл
            temp_audio_path (str): Путь для сохранения аудио

        Returns:
            bool: Успешность извлечения
        """
        try:
            if video.audio is not None:
                video.audio.write_audiofile(
                    temp_audio_path,
                    codec='aac',
                    ffmpeg_params=['-strict', '-2'],
                    verbose=False,
                    logger=None
                )
                logger.info("Audio extracted successfully")
                return True
        except Exception as e:
            logger.warning(f"Could not extract audio: {str(e)}")
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
        return False

    def merge_audio(self, video_path, audio_path, output_path):
        """
        Объединение видео и аудио

        Args:
            video_path (str): Путь к видео
            audio_path (str): Путь к аудио
            output_path (str): Путь для сохранения результата
        """
        logger.info("Merging video with audio...")
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-i', audio_path,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-strict', '-2',
                output_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info("Audio merged successfully")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to merge audio: {e.stderr.decode()}")
            os.rename(video_path, output_path)

    def save_progress(self, output_path, processed_frames, total_frames):
        """Сохранение прогресса обработки"""
        progress = {
            'processed_frames': processed_frames,
            'total_frames': total_frames,
            'timestamp': datetime.now().isoformat()
        }
        with open(f"{output_path}.progress", 'w') as f:
            json.dump(progress, f)

    def load_progress(self, output_path):
        """Загрузка прогресса обработки"""
        try:
            with open(f"{output_path}.progress", 'r') as f:
                progress = json.load(f)
            return progress['processed_frames']
        except:
            return 0

    def process_video(self, input_path, output_path, resume=False):
        """
        Обработка видео

        Args:
            input_path (str): Путь к входному видео
            output_path (str): Путь для сохранения результата
            resume (bool): Продолжить обработку с последнего сохраненного состояния
        """
        temp_audio_path = f"{output_path}.temp.aac"
        temp_output_path = f"{output_path}.temp.mp4"
        has_audio = False

        try:
            with self.video_context(input_path) as video:
                # Извлечение аудио
                logger.info("Extracting audio...")
                has_audio = self.extract_audio(video, temp_audio_path)

                # Подготовка к обработке видео
                logger.info("Processing video...")
                fps = video.fps
                total_frames = int(video.duration * fps)
                width, height = video.size

                # Вывод информации о видео
                logger.info(f"\nVideo Info:")
                logger.info(f"Resolution: {width}x{height}")
                logger.info(f"FPS: {fps}")
                logger.info(f"Duration: {video.duration:.2f} seconds")
                logger.info(f"Total frames: {total_frames}")
                logger.info(f"Output resolution: {width * 2}x{height * 2}\n")

                # Загрузка прогресса если нужно
                start_frame = self.load_progress(output_path) if resume else 0
                if start_frame > 0:
                    logger.info(f"Resuming from frame {start_frame}")

                with FFMPEG_VideoWriter(
                        temp_output_path,
                        (width * 2, height * 2),
                        fps,
                        ffmpeg_params=['-vcodec', 'libx264', '-crf', '17']
                ) as writer:

                    pbar = tqdm(total=total_frames,
                                initial=start_frame,
                                desc="Processing frames",
                                unit="frames")

                    try:
                        processed_frames = start_frame
                        start_time = time()
                        fps_buffer = []
                        current_batch = []

                        # Обработка кадров
                        for i, frame in enumerate(video.iter_frames()):
                            if i < start_frame:
                                continue

                            frame_start_time = time()

                            # Батчинг
                            current_batch.append(frame)
                            if len(current_batch) == self.batch_size:
                                enhanced_frames = self.enhance_batch(current_batch)
                                for enhanced_frame in enhanced_frames:
                                    writer.write_frame(enhanced_frame)
                                    processed_frames += 1
                                current_batch = []

                                # Обновление прогресса
                                frame_time = (time() - frame_start_time) / self.batch_size
                                fps_buffer.append(1 / frame_time if frame_time > 0 else 0)
                                if len(fps_buffer) > 30:
                                    fps_buffer.pop(0)

                                current_fps = sum(fps_buffer) / len(fps_buffer)
                                pbar.set_postfix({
                                    'FPS': f"{current_fps:.1f}",
                                    'Elapsed': f"{(time() - start_time):.1f}s"
                                })
                                pbar.update(self.batch_size)

                                # Сохранение прогресса каждые 100 кадров
                                if processed_frames % 100 == 0:
                                    self.save_progress(output_path, processed_frames, total_frames)

                        # Обработка оставшихся кадров
                        if current_batch:
                            enhanced_frames = self.enhance_batch(current_batch)
                            for enhanced_frame in enhanced_frames:
                                writer.write_frame(enhanced_frame)
                                processed_frames += 1
                            pbar.update(len(current_batch))

                    finally:
                        pbar.close()

            # Объединение с аудио если есть
            if has_audio:
                self.merge_audio(temp_output_path, temp_audio_path, output_path)
            else:
                os.rename(temp_output_path, output_path)

            # Очистка файла прогресса
            if os.path.exists(f"{output_path}.progress"):
                os.remove(f"{output_path}.progress")

            end_time = time()
            logger.info(f"\nVideo processing completed in {end_time - start_time:.2f} seconds")
            logger.info(f"Average FPS: {processed_frames / (end_time - start_time):.2f}")

        finally:
            # Очистка временных файлов
            for temp_file in [temp_output_path, temp_audio_path]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)


def main():
    """Основная функция"""
    try:
        model_path = 'model/RealESRGAN_final.pth'
        logger.info(f"Model path: {os.path.abspath(model_path)}")

        # Проверка наличия необходимых файлов и директорий
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model file not found at /model/RealESRGAN_final.pth")

        if not os.path.exists('input'):
            raise FileNotFoundError("Input directory not found at /input")

        if not os.path.exists('output'):
            os.makedirs('output')

        # Поиск видео файлов
        input_files = [f for f in os.listdir('input') if f.endswith(('.mkv', '.mp4', '.avi'))]

        if not input_files:
            raise FileNotFoundError("No video files found in /input directory")

        # Инициализация апскейлера
        upscaler = VideoUpscaler(
            model_path=model_path,
            batch_size=1,
            num_workers=12
        )

        # Обработка каждого видео
        for i, input_file in enumerate(input_files, 1):
            input_path = os.path.join('input', input_file)
            output_path = os.path.join('output', f'upscaled_{input_file}')

            logger.info(f"\nProcessing video {i}/{len(input_files)}: {input_file}")
            upscaler.process_video(input_path, output_path)

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise


if __name__ == "__main__":
    main()