import torch
import torch.nn as nn
from src.transforms.img_augs.centralcrop import CentralCrop
from src.transforms.img_augs.randomcrop import RandomCrop


class CombinedCrop(nn.Module):
    """
    A module that combines central cropping and random cropping.

    This module first applies a central crop to the input tensor, reducing it to a specified size (e.g., 512x512).
    Then, it applies a random crop to the result of the central crop, further reducing it to another specified size
    (e.g., 256x256).

    Attributes:
        central_crop (CentralCrop): A module for central cropping to a specified size (e.g., 512x512).
        random_crop (RandomCrop): A module for random cropping to a specified size (e.g., 256x256).
    """

    def __init__(self, target_width_cc: int, target_height_cc: int, target_width_rc: int, target_height_rc: int) -> None:
        """
        Initializes the CombinedCrop module.

        Args:
            target_width_cc (int): The target width for the central crop.
            target_height_cc (int): The target height for the central crop.
            target_width_rc (int): The target width for the random crop.
            target_height_rc (int): The target height for the random crop.

        Note:
            The central crop is applied first, followed by the random crop.
        """
        super().__init__()
        self.central_crop = CentralCrop(target_width=target_width_cc, target_height=target_height_cc)
        self.random_crop = RandomCrop(target_width=target_width_rc, target_height=target_height_rc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies combined cropping to the input tensor.

        Args:
            x (torch.Tensor): The input tensor with at least two dimensions (height and width).
                              The last two dimensions are treated as height and width.

        Returns:
            torch.Tensor: A tensor that is first centrally cropped to the specified size (e.g., 512x512)
                          and then randomly cropped to the specified size (e.g., 256x256).

        Raises:
            ValueError: If the input tensor is smaller than the required size for cropping.
        """
        # Check if the input tensor is large enough for the central crop
        if x.size(-2) < self.central_crop.target_height or x.size(-1) < self.central_crop.target_width:
            raise ValueError(
                f"Input tensor must be at least {self.central_crop.target_height}x{self.central_crop.target_width} "
                f"for central cropping. Got {x.size(-2)}x{x.size(-1)}."
            )

        # Apply central cropping to the input tensor
        x = self.central_crop(x)

        # Check if the centrally cropped tensor is large enough for the random crop
        if x.size(-2) < self.random_crop.target_height or x.size(-1) < self.random_crop.target_width:
            raise ValueError(
                f"Centrally cropped tensor must be at least {self.random_crop.target_height}x{self.random_crop.target_width} "
                f"for random cropping. Got {x.size(-2)}x{x.size(-1)}."
            )

        # Apply random cropping to the centrally cropped tensor
        x = self.random_crop(x)

        return x