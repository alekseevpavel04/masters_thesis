import os
import glob
import subprocess
from image_filter_ResNet import ImageFilter

class DatasetPreparer:
    def __init__(self, video_dir='raw_video', output_dir='output_images'):
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.image_filter = ImageFilter()

        os.makedirs(self.output_dir, exist_ok=True)

    def extract_frames(self):
        video_extensions = ['*.mp4', '*.mkv']
        video_files = []

        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(self.video_dir, ext)))

        if not video_files:
            print("No video files found in the raw_video directory.")
            return

        for video_file in video_files:
            print(f"Extracting frames from {video_file}...")
            output_pattern = os.path.join(self.output_dir, f"{os.path.basename(video_file)}_frame_%04d.jpg")
            command = [
                'ffmpeg', '-i', video_file, '-vf', 'select=not(mod(n\\,1000))', 
                '-vsync', 'vfr', output_pattern
            ]
            print(f"Running command: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error extracting frames: {result.stderr}")

    def process_images(self):
        image_files = glob.glob(os.path.join(self.output_dir, '*.jpg'))
        if not image_files:
            print("No images found in the output directory. Please check the video extraction process.")
            return

        print(f"Processing {len(image_files)} images...")
        files_to_remove = self.image_filter.get_files_to_remove(image_files)

        # Удаление наименее информативных изображений
        for file in files_to_remove:
            print(f"Removing {file}")
            os.remove(file)

        print(f"Removed {len(files_to_remove)} least informative images.")

if __name__ == '__main__':
    dataset_preparer = DatasetPreparer()
    dataset_preparer.extract_frames()
    dataset_preparer.process_images()