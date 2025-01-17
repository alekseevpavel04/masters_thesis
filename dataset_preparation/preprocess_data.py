from tools.delete_invalid_images import DatasetFilter
from tools.prepair_images import VideoCooker
from tools.image_filter_ResNet import ImageFilterNet
from tools.detect_dublicate import ImageDeduplicator
from tools.logger_config import setup_logger

logger = setup_logger(__name__)

class PreprocessData:
    """
    A class to handle the preprocessing of video data into a dataset of images.
    This includes extracting frames from videos, filtering out low-quality images,
    and removing duplicate images.
    """

    def __init__(self):
        """
        Initializes the PreprocessData class with necessary components.
        """
        self.coocker = VideoCooker(
            video_dir="raw_video",
            output_dir="output_images",
            similarity_threshold=0.1
        )

        self.image_filter = ImageFilterNet(info_threshold=2)

        self.deleter = DatasetFilter(
            image_fitter=self.image_filter
        )

        self.duplicate_filter = ImageDeduplicator(output_dir="output_images", threshold=10)

    def create_dataset(self):
        """
        Executes the dataset creation process by extracting frames from videos,
        filtering out low-quality images, and removing duplicates.
        """
        logger.info("Starting dataset creation process")
        logger.info("Extracting frames from videos")
        self.coocker.extract_frames()
        logger.info("Processing images and removing low quality ones")
        self.deleter.process_images()
        logger.info("Removing duplicate images based on imagehash")
        self.duplicate_filter.deduplicate_images()
        logger.info("Dataset creation process completed")

if __name__ == '__main__':
    logger.info("Starting preprocessing script")
    dataset_preparer = PreprocessData()
    dataset_preparer.create_dataset()
    logger.info("Preprocessing script completed")