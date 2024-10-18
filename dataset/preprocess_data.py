from tools.delete_invalid_images import DatasetFilter
from tools.prepair_images import VideoCooker
from tools.image_filter_ResNet import ImageFilterNet
from tools.detect_dublicate import ImageDeduplicator
from tools.logger_config import setup_logger

logger = setup_logger(__name__)

class PreprocessData:
    def __init__(self):
        self.coocker = VideoCooker(
            video_dir="raw_video",
            output_dir="output_images",
            crop_width=224,
            crop_height=224,
            similarity_threshold = 0.1
        )

        self.image_filter = ImageFilterNet()

        self.deleter = DatasetFilter(
            image_fitter=self.image_filter
        )

        self.duplicate_filter = ImageDeduplicator(output_dir="output_images", threshold = 10)

    def create_dataset(self):
        logger.info("Starting dataset creation process")
        logger.info("Extracting frames from videos")
        self.coocker.extract_frames()
        logger.info("Processing images and removing low quality ones")
        self.deleter.process_images()
        logger.info("Removing duplicate images based on PSNR")
        self.duplicate_filter.deduplicate_images()
        logger.info("Dataset creation process completed")
        

if __name__ == '__main__':
    logger.info("Starting preprocessing script")
    dataset_preparer = PreprocessData()
    dataset_preparer.create_dataset()
    logger.info("Preprocessing script completed")