import os
import glob
import subprocess
from tools.logger_config import setup_logger

logger = setup_logger(__name__)

class VideoCooker:
    def __init__(self, video_dir="raw_video", output_dir="output_images", similarity_threshold=0.3):
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.similarity_threshold = similarity_threshold
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
            output_pattern = os.path.join(self.output_dir, f"{os.path.basename(video_file)}_frame_%04d.png")
            
            command = [
                'ffmpeg', 
                '-i', video_file,
                '-vf', f'select=\'gt(scene,{self.similarity_threshold})\'',
                '-fps_mode', 'vfr',
                output_pattern
            ]
            
            logger.debug(f"Running command: {' '.join(command)}")
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            for line in process.stdout:
                if "frame=" in line:
                    pass
                    # logger.info(f"Created frame: {line.strip()}")
            
            process.wait()
            
            if process.returncode != 0:
                logger.error(f"Error extracting frames: {process.returncode}")
            else:
                logger.info(f"Successfully extracted frames from {video_file}")
