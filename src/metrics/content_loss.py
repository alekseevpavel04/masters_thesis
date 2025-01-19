import torch
import torch.nn.functional as F
from typing import Optional, Any


class ContentLossMetric:
    """
    A class to calculate L1 content loss between generated and ground truth images.

    Content loss measures the absolute differences between generated and target images,
    providing a pixel-wise comparison metric that is less sensitive to outliers than L2 loss.

    Attributes:
        name (str): Name of the metric, defaults to class name if not specified
        device (str): Device where computations will be performed
    """

    def __init__(self, device: str, name: Optional[str] = None) -> None:
        """
        Initialize ContentLossMetric with specified device and optional name.

        Args:
            device (str): Device to run calculations on. Use 'cuda' for GPU, 'cpu' for CPU,
                         or 'auto' for automatic selection based on availability
            name (Optional[str]): Custom name for the metric. If None, uses class name
        """
        self.name = name if name is not None else self.__class__.__name__
        self.device = device if device != "auto" else "cuda" if torch.cuda.is_available() else "cpu"

    def __call__(
            self,
            gen_output: torch.Tensor,
            data_object: torch.Tensor,
            **kwargs: Any
    ) -> torch.Tensor:
        """
        Calculate L1 loss between generated and ground truth images.

        Args:
            gen_output (torch.Tensor): Generated high-resolution images, shape (B, C, H, W)
            data_object (torch.Tensor): Ground truth high-resolution images, shape (B, C, H, W)
            **kwargs: Additional arguments (not used but included for compatibility)

        Returns:
            torch.Tensor: L1 loss value, lower values indicate better similarity
        """
        return F.l1_loss(gen_output, data_object)