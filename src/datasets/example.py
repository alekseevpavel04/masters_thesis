import os
import zipfile
import subprocess
from pathlib import Path
import safetensors
import safetensors.torch
import torchvision
from tqdm.auto import tqdm
from PIL import Image

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class CustomDataset(BaseDataset):
    """
    Custom dataset for images downloaded from Kaggle dataset (with no labels).
    The images will be treated as part of a generic class.
    """

    def __init__(self, *args, **kwargs):
        """
        Args:
            name (str): partition name (default 'train', but we won't use it)
        """
        index_path = ROOT_PATH / "data" / "custom" / "index.json"

        # If the index file exists, load it; otherwise, create it.
        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index()

        super().__init__(index, *args, **kwargs)

    def _create_index(self):
        """
        Create index for the dataset. The function processes dataset metadata
        and utilizes it to get information dict for each element of
        the dataset.

        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as object path.
        """
        index = []
        data_path = ROOT_PATH / "data" / "custom"
        data_path.mkdir(exist_ok=True, parents=True)

        # Define the path where the zip file is located
        zip_file = ROOT_PATH / "data" / "dataset-nano.zip"

        # Check if the dataset zip file exists, and if not, download it
        if not zip_file.exists():
            print("Dataset not found. Downloading from Kaggle...")
            self._download_dataset()

        # Unzip the dataset if it has not been unzipped
        if not (data_path / "dataset_nano").exists():
            self._unzip_dataset(zip_file, data_path)

        image_folder = data_path / "dataset_nano"
        image_files = [f for f in image_folder.glob("*") if f.is_file()]

        print(f"Parsing custom dataset metadata...")
        # Create dataset and index
        for i, img_path in enumerate(tqdm(image_files)):
            # Save the original image in the same format as the input
            save_path = image_folder / f"{i:06}{img_path.suffix}"  # Keep the original extension (e.g., .jpg, .png)
            img = Image.open(img_path)

            # Save the image in the same format as it was originally
            img.save(save_path)

            # Remove the original image file
            os.remove(img_path)

            # Add metadata for this image to the index (without 'label')
            index.append({"path": str(save_path)})

        # Write the index file
        write_json(index, str(data_path / "index.json"))

        # Remove the zip file after extraction
        self._cleanup(zip_file)

        return index

    def _download_dataset(self):
        """
        Download the dataset from Kaggle using the Kaggle CLI command.
        """
        dataset_name = "alekseevpavel/dataset-nano"  # Update this to the correct Kaggle dataset identifier
        try:
            # Execute the kaggle command to download the dataset
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", dataset_name, "-p", str(ROOT_PATH / "data")],
                check=True
            )
            print("Dataset downloaded successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error downloading dataset: {e}")

    def _unzip_dataset(self, zip_file, data_path):
        """
        Unzips the dataset into the specified directory.

        Args:
            zip_file (Path): path to the zip file.
            data_path (Path): directory where to unzip the data.
        """
        print(f"Unzipping dataset from {zip_file} to {data_path}...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_path)
        print("Dataset unzipped successfully.")

    def _cleanup(self, zip_file):
        """
        Remove the zip file after the dataset has been unzipped.

        Args:
            zip_file (Path): path to the zip file.
        """
        if zip_file.exists():
            print(f"Cleaning up by removing zip file {zip_file}...")
            os.remove(zip_file)

    def load_img(self, path):
        """
        Load image from disk and convert it to tensor.

        Args:
            path (str): path to the image.
        Returns:
            img (Tensor): image as a PyTorch tensor.
        """
        img = torchvision.transforms.ToTensor()(Image.open(path))
        return img
