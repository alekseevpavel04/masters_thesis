import torch
from typing import Optional, Any
from torchmetrics.image import StructuralSimilarityIndexMeasure


class SSIMMetric:
    """
    A class to calculate Structural Similarity Index Measure (SSIM) between generated and ground truth images.

    SSIM is designed to improve upon traditional metrics like PSNR and MSE, which have been proven
    to be inconsistent with human visual perception. SSIM considers image degradation as perceived
    change in structural information.

    Attributes:
        name (str): Name of the metric, defaults to class name if not specified
        ssim (StructuralSimilarityIndexMeasure): TorchMetrics implementation of SSIM calculation
    """

    def __init__(self, device: str, name: Optional[str] = None) -> None:
        """
        Initialize SSIMMetric with specified device and optional name.

        Args:
            device (str): Device to run calculations on. Use 'cuda' for GPU, 'cpu' for CPU,
                         or 'auto' for automatic selection based on availability
            name (Optional[str]): Custom name for the metric. If None, uses class name
        """
        self.name = name if name is not None else self.__class__.__name__
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    def __call__(
            self,
            gen_output: torch.Tensor,
            data_object: torch.Tensor,
            **kwargs: Any
    ) -> torch.Tensor:
        """
        Calculate SSIM between generated and ground truth images.

        Args:
            gen_output (torch.Tensor): Generated high-resolution images, shape (B, C, H, W)
            data_object (torch.Tensor): Ground truth high-resolution images, shape (B, C, H, W)
            **kwargs: Additional arguments (not used but included for compatibility)

        Returns:
            torch.Tensor: SSIM value, ranges from -1 to 1, where 1 indicates perfect structural similarity
        """
        return self.ssim(gen_output, data_object)