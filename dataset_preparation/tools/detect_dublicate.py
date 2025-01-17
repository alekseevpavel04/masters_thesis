import os
from PIL import Image
import imagehash
from tools.logger_config import setup_logger

logger = setup_logger("ImageDeduplicator")

class ImageDeduplicator:
    """
    A class to detect and remove duplicate images from a dataset based on image hashing.
    """

    def __init__(self, output_dir="output_images", threshold=5):
        """
        Initializes the ImageDeduplicator class.

        Args:
            output_dir (str): Directory where the images are stored.
            threshold (int): Threshold for determining image similarity.
        """
        self.output_dir = output_dir
        self.threshold = threshold
        self.hashes = {}
        self.duplicates = set()  # Using a set to store unique duplicates
        self.logger = logger

    def find_similar_images(self):
        """
        Finds similar images in the specified directory based on image hashing.
        """
        self.logger.info("Starting to find similar images.")
        for filename in os.listdir(self.output_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_path = os.path.join(self.output_dir, filename)
                try:
                    image = Image.open(image_path)
                    hash_value = imagehash.phash(image)

                    # Check for similar images
                    for existing_hash in self.hashes:
                        if hash_value - existing_hash < self.threshold:
                            # Add to the set of unique duplicates
                            self.duplicates.add((filename, self.hashes[existing_hash]))

                    # Save the image hash
                    self.hashes[hash_value] = filename
                except Exception as e:
                    self.logger.error(f"Error processing image {filename}: {e}")

    def remove_duplicates(self):
        """
        Removes duplicate images from the directory.
        """
        if not self.duplicates:
            self.logger.info("No duplicates to remove.")
            return

        total_images = len(self.hashes)  # Total number of images
        removed_count = 0  # Number of removed duplicates

        for img1, img2 in self.duplicates:
            img_path = os.path.join(self.output_dir, img2)
            if os.path.exists(img_path):  # Check if the file exists
                os.remove(img_path)
                removed_count += 1  # Increment the count of removed images
            else:
                self.logger.warning(f"File not found, skipping: {img_path}")

        return total_images, removed_count  # Return total and removed counts

    def deduplicate_images(self):
        """
        Main method to find and remove duplicate images.
        """
        self.find_similar_images()
        total_images, removed_count = self.remove_duplicates()
        self.logger.info("Duplicate removal process completed.")

        # Calculate the fraction of removed images
        if total_images > 0:
            removal_fraction = removed_count / total_images
            self.logger.info(f"Removed {removed_count} duplicates out of {total_images} images. Removal fraction: {removal_fraction:.2%}")
        else:
            self.logger.info("No images to process.")