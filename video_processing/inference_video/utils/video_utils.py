import os
import subprocess
from contextlib import contextmanager
import numpy as np
import cv2
from moviepy import *
import sys
import io


@contextmanager
def video_context(path):
    """Context manager for safe video handling with accurate frame counting"""
    video = None
    # Сохраняем оригинальный stdout
    original_stdout = sys.stdout
    try:
        # Перенаправляем stdout в null для подавления вывода VideoFileClip
        sys.stdout = io.StringIO()

        # First, use cv2 to get accurate frame count
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # Now open with moviepy for processing
        video = VideoFileClip(path)

        # Update duration based on accurate frame count
        video.duration = total_frames / fps if fps > 0 else video.duration

        def safe_iter_frames():
            """Iterator that ensures safe frame reading with proper validation"""
            with VideoFileClip(path) as temp_video:
                last_valid_frame = None
                frame_count = 0

                for frame in temp_video.iter_frames():
                    if frame_count >= total_frames:
                        break

                    if frame is not None and frame.size > 0 and not np.all(frame == 0):
                        # Validate frame content
                        if frame.shape[2] == 3 and np.any(frame != 0):
                            last_valid_frame = frame.copy()
                            yield frame
                            frame_count += 1
                            continue

                    # If current frame is invalid, use last valid frame
                    if last_valid_frame is not None:
                        yield last_valid_frame.copy()
                    else:
                        # If no valid frame yet, yield black frame
                        yield np.zeros((video.size[1], video.size[0], 3), dtype=np.uint8)
                    frame_count += 1

        # Replace original iter_frames with our safe version
        video.iter_frames = safe_iter_frames
        yield video

    except Exception as e:
        raise RuntimeError(f"Error processing video {path}: {str(e)}")
    finally:
        # Восстанавливаем оригинальный stdout
        sys.stdout = original_stdout
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