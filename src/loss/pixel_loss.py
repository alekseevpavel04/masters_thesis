import torch
import torch.nn as nn


class PixelLoss(nn.Module):
    """
    Pixel-wise loss for image reconstruction tasks. This loss measures the difference
    between the super-resolved (SR) and high-resolution (HR) images using L1 loss.

    Attributes:
        l1 (nn.L1Loss): L1 loss function.
    """

    def __init__(self):
        """
        Initializes the PixelLoss module with L1 loss.
        """
        super(PixelLoss, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, sr, hr):
        """
        Computes the pixel-wise loss between the super-resolved and high-resolution images.

        Args:
            sr (torch.Tensor): The super-resolved image (output of the model).
            hr (torch.Tensor): The high-resolution ground truth image.

        Returns:
            torch.Tensor: The computed L1 loss.
        """
        return self.l1(sr, hr)