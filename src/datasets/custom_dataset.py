import os
import zipfile
import subprocess
from pathlib import Path
import safetensors.torch
import torchvision
from tqdm.auto import tqdm
from PIL import Image

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class CustomDataset(BaseDataset):
    """
    Custom dataset for images downloaded from Kaggle dataset.
    """
    ROOT_PATH = ROOT_PATH  # Make ROOT_PATH accessible to parent class

    def __init__(self, partition: str, *args, **kwargs):
        """
        Args:
            partition (str): Dataset partition ('train', 'val', or 'test')
        """
        super().__init__(partition=partition, *args, **kwargs)

    def _create_index(self):
        """
        Create index for the dataset partition.

        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as object path.
        """
        index = []
        data_path = ROOT_PATH / "data" / "custom"
        data_path.mkdir(exist_ok=True, parents=True)

        # Dataset-specific configurations
        # dataset_configs = {
        #     'train': {'name': 'sr-dataset', 'folder': 'SR_dataset'},
        #     'val': {'name': 'sr-dataset-val', 'folder': 'SR_dataset_val'},
        #     'test': {'name': 'sr-dataset-test', 'folder': 'SR_dataset_test'}
        # }

        # To test we may use nano versions
        dataset_configs = {
            'train': {'name': 'sr-dataset-nano', 'folder': 'SR_dataset_nano'},
            'val': {'name': 'sr-dataset-val-nano', 'folder': 'SR_dataset_val_nano'},
            'test': {'name': 'sr-dataset-test-nano', 'folder': 'SR_dataset_test_nano'}
        }

        config = dataset_configs[self.partition]
        zip_file = ROOT_PATH / "data" / f"{config['name']}.zip"
        folder_path = data_path / config['folder']

        # Download and extract if needed
        if not zip_file.exists():
            print(f"Dataset {self.partition} not found. Downloading from Kaggle...")
            self._download_dataset(config['name'])

        if not folder_path.exists():
            self._unzip_dataset(zip_file, data_path)

        image_files = [f for f in folder_path.glob("*") if f.is_file()]

        print(f"Parsing {self.partition} dataset metadata...")
        for i, img_path in enumerate(tqdm(image_files)):
            save_path = folder_path / f"{i:06}{img_path.suffix}"
            img = Image.open(img_path)
            img.save(save_path)
            os.remove(img_path)
            index.append({"path": str(save_path)})

        # Write partition-specific index file
        index_path = self.get_index_path()
        write_json(index, str(index_path))

        self._cleanup(zip_file)
        return index

    def _download_dataset(self, dataset_name):
        """
        Download dataset from Kaggle.

        Args:
            dataset_name (str): Name of the dataset on Kaggle
        """
        kaggle_name = f"alekseevpavel/{dataset_name}"
        try:
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", kaggle_name, "-p", str(ROOT_PATH / "data")],
                check=True
            )
            print(f"Dataset {self.partition} downloaded successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error downloading dataset: {e}")

    def _unzip_dataset(self, zip_file, data_path):
        """
        Unzip the dataset.

        Args:
            zip_file (Path): Path to the zip file
            data_path (Path): Directory where to unzip the data
        """
        print(f"Unzipping dataset from {zip_file} to {data_path}...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_path)
        print("Dataset unzipped successfully.")

    def _cleanup(self, zip_file):
        """
        Clean up zip file after extraction.

        Args:
            zip_file (Path): Path to the zip file to remove
        """
        if zip_file.exists():
            print(f"Cleaning up by removing zip file {zip_file}...")
            os.remove(zip_file)

    def load_img(self, path):
        """
        Load image and convert to tensor.

        Args:
            path (str): Path to the image file
        Returns:
            Tensor: Image as a PyTorch tensor
        """
        return torchvision.transforms.ToTensor()(Image.open(path))


class CustomDataset_train(CustomDataset):
    """Custom dataset class for training data."""

    def __init__(self, *args, **kwargs):
        super().__init__(partition='train', *args, **kwargs)


class CustomDataset_val(CustomDataset):
    """Custom dataset class for validation data."""

    def __init__(self, *args, **kwargs):
        super().__init__(partition='val', *args, **kwargs)


class CustomDataset_test(CustomDataset):
    """Custom dataset class for test data."""

    def __init__(self, *args, **kwargs):
        super().__init__(partition='test', *args, **kwargs)