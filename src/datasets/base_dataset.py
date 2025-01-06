import logging
import random
from typing import List
import safetensors.torch
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """
    Base class for the datasets.

    Given a proper index (list[dict]), allows to process different datasets
    for the same task in the identical manner. Therefore, to work with
    several datasets, the user only has to define index in a nested class.
    """

    def __init__(
            self,
            partition: str,
            index=None,
            limit=None,
            shuffle_index=False,
            instance_transforms=None
    ):
        """
        Args:
            partition (str): Dataset partition ('train', 'val', or 'test')
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
            instance_transforms (dict[Callable] | None): transforms that
                should be applied on the instance. Depend on the
                tensor name.
        """
        self.partition = partition

        # If no index provided, create it
        if index is None:
            index = self._create_index()

        self._assert_index_is_valid(index)
        index = self._shuffle_and_limit_index(index, limit, shuffle_index)
        self._index: List[dict] = index
        self.instance_transforms = instance_transforms

    def __getitem__(self, ind):
        """
        Get element from the index, preprocess it, and combine it
        into a dict.

        Notice that the choice of key names is defined by the template user.
        However, they should be consistent across dataset getitem, collate_fn,
        loss_function forward method, and model forward method.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        """
        data_dict = self._index[ind]
        data_path = data_dict["path"]
        data_object = self.load_object(data_path)

        instance_data = {"data_object": data_object}
        instance_data = self.preprocess_data(instance_data)

        return instance_data

    def __len__(self):
        """
        Get length of the dataset (length of the index).
        """
        return len(self._index)

    def get_index_path(self):
        """Get the path for the index file based on partition"""
        return Path(self.ROOT_PATH) / "data" / "custom" / f"index_{self.partition}.json"

    def get_data_folder(self):
        """Get the path for the data folder based on partition"""
        return Path(self.ROOT_PATH) / "data" / "custom" / f"SR_dataset_{self.partition}"

    def load_object(self, path):
        """
        Load object from disk.

        Args:
            path (str): path to the object.
        Returns:
            data_object (Tensor): loaded tensor object.
        """
        # For image files
        if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # Load image using PIL
            image = Image.open(path)

            # Convert image to tensor
            transform = transforms.Compose([
                transforms.ToTensor(),  # Converts to tensor and scales to [0, 1]
            ])

            return transform(image)

        # For PyTorch tensor files
        elif path.lower().endswith('.pt') or path.lower().endswith('.pth'):
            # Use weights_only=True for security
            return torch.load(path, weights_only=True)

        else:
            raise ValueError(f"Unsupported file format: {path}")

    def preprocess_data(self, instance_data):
        """
        Preprocess data with instance transforms.

        Each tensor in a dict undergoes its own transform defined by the key.

        Args:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element) (possibly transformed via
                instance transform).
        """
        if self.instance_transforms is not None:
            for transform_name in self.instance_transforms.keys():
                instance_data[transform_name] = self.instance_transforms[
                    transform_name
                ](instance_data[transform_name])
        return instance_data

    @staticmethod
    def _filter_records_from_dataset(
            index: list,
    ) -> list:
        """
        Filter some of the elements from the dataset depending on
        some condition.

        This is not used in the example. The method should be called in
        the __init__ before shuffling and limiting.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset that satisfied the condition. The dict has
                required metadata information, such as label and object path.
        """
        # Filter logic
        pass

    @staticmethod
    def _assert_index_is_valid(index):
        """
        Check the structure of the index and ensure it satisfies the desired
        conditions.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        for entry in index:
            assert "path" in entry, (
                "Each dataset item should include field 'path'" " - path to audio file."
            )

    @staticmethod
    def _sort_index(index):
        """
        Sort index via some rules.

        This is not used in the example. The method should be called in
        the __init__ before shuffling and limiting and after filtering.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        Returns:
            index (list[dict]): sorted list, containing dict for each element
                of the dataset. The dict has required metadata information,
                such as label and object path.
        """
        return sorted(index, key=lambda x: x["KEY_FOR_SORTING"])

    @staticmethod
    def _shuffle_and_limit_index(index, limit, shuffle_index):
        """
        Shuffle elements in index and limit the total number of elements.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
        """
        if shuffle_index:
            random.seed(42)
            random.shuffle(index)

        if limit is not None:
            index = index[:limit]
        return index