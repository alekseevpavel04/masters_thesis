from delete_invalid_images import DatasetFilter
from prepair_images import VideoCooker
from image_filter_ResNet import ImageFilterNet
from logger_config import setup_logger

logger = setup_logger(__name__)

class PreprocessData:
    def __init__(self):
        self.coocker = VideoCooker(
            video_dir="raw_video",
            output_dir="output_images",
            frame_interval=1000,
            crop_width=224,
            crop_height=224
        )

        self.image_filter = ImageFilterNet()

        self.deleter = DatasetFilter(
            image_fitter=self.image_filter
        )

    def create_dataset(self):
        logger.info("Starting dataset creation process")
        logger.info("Extracting frames from videos")
        self.coocker.extract_frames()
        logger.info("Processing images and removing low quality ones")
        self.deleter.process_images()
        logger.info("Dataset creation process completed")

if __name__ == '__main__':
    logger.info("Starting preprocessing script")
    dataset_preparer = PreprocessData()
    dataset_preparer.create_dataset()
    logger.info("Preprocessing script completed")