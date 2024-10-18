import os
import glob
import subprocess
from logger_config import setup_logger

logger = setup_logger(__name__)

class VideoCooker:
    def __init__(self, video_dir="raw_video", output_dir="output_images", frame_interval=1000, crop_width=224, crop_height=224):
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.frame_interval = frame_interval
        self.crop_width = crop_width
        self.crop_height = crop_height
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_frames(self):
        video_extensions = ['*.mp4', '*.mkv']
        video_files = []

        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(self.video_dir, ext)))

        if not video_files:
            logger.warning("No video files found in the raw_video directory.")
            return

        for video_file in video_files:
            logger.info(f"Extracting frames from {video_file}...")
            output_pattern = os.path.join(self.output_dir, f"{os.path.basename(video_file)}_frame_%04d.jpg")
            
            # Команда для извлечения кадров с центральным кропом
            command = [
                'ffmpeg', '-i', video_file,
                '-vf', f'select=not(mod(n\\,{self.frame_interval})),'
                       f'crop={self.crop_width}:{self.crop_height}:'
                       f'(in_w-{self.crop_width})/2:'
                       f'(in_h-{self.crop_height})/2',
                '-fps_mode', 'vfr', output_pattern
            ]
            logger.debug(f"Running command: {' '.join(command)}")

            # Запуск процесса и обработка вывода в реальном времени
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

            for line in process.stdout:
                if "frame=" in line:
                    # Логирование информации о каждом созданном кадре
                    logger.info(f"Created frame: {line.strip()}")

            process.wait()  # Дожидаемся завершения процесса
            
            if process.returncode != 0:
                logger.error(f"Error extracting frames: {process.returncode}")
            else:
                logger.info(f"Successfully extracted frames from {video_file}")
