# utils/video_utils.py
import os
import subprocess
from contextlib import contextmanager
from moviepy.editor import VideoFileClip
import numpy as np


@contextmanager
def video_context(path):
    """Context manager for safe video handling with proper frame counting"""
    video = None
    try:
        video = VideoFileClip(path)
        # Создаем обертку для более точного подсчета кадров
        original_iter_frames = video.iter_frames

        # Точно подсчитываем количество кадров
        frame_count = sum(1 for _ in original_iter_frames())
        # Обновляем duration для соответствия реальному количеству кадров
        video.duration = frame_count / video.fps

        def safe_iter_frames():
            # Заново открываем видео для итерации
            with VideoFileClip(path) as temp_video:
                last_valid_frame = None
                for i, frame in enumerate(temp_video.iter_frames()):
                    if i >= frame_count:
                        break
                    if frame is not None and frame.size > 0:
                        last_valid_frame = frame.copy()
                        yield frame
                    elif last_valid_frame is not None:
                        yield last_valid_frame.copy()
                    else:
                        yield np.zeros((video.size[1], video.size[0], 3), dtype=np.uint8)

        # Заменяем оригинальный метод на наш безопасный
        video.iter_frames = safe_iter_frames
        yield video
    finally:
        if video is not None:
            video.close()


def extract_audio(video, temp_audio_path, logger):
    """Extract audio from video file"""
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


def merge_audio(video_path, audio_path, output_path, logger):
    """Merge video and audio files"""
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