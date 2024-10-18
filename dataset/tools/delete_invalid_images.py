import os
import glob
from tools.image_filter_ResNet import ImageFilterNet
from tools.logger_config import setup_logger

logger = setup_logger(__name__)

class DatasetFilter:
    def __init__(self, output_dir='output_images', image_fitter=ImageFilterNet()):
        self.output_dir = output_dir
        self.image_filter = image_fitter
        os.makedirs(self.output_dir, exist_ok=True)

    def process_images(self):
        image_files = glob.glob(os.path.join(self.output_dir, '*.jpg'))
        if not image_files:
            logger.warning("No images found in the output directory. Please check the video extraction process.")
            return

        logger.info(f"Processing {len(image_files)} images...")
        files_to_remove = self.image_filter.get_files_to_remove(image_files)

        for file in files_to_remove:
            logger.info(f"Removing {file}")
            os.remove(file)

        logger.info(f"Removed {len(files_to_remove)} least informative images.")