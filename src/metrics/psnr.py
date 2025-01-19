import torch
from typing import Optional, Any
from torchmetrics.image import PeakSignalNoiseRatio


class PSNRMetric:
    """
    A class to calculate Peak Signal-to-Noise Ratio (PSNR) between generated and ground truth images.

    PSNR is a metric that measures the ratio between the maximum possible signal power and
    the noise that affects the quality of the representation. Higher PSNR values indicate
    better image quality.

    Attributes:
        name (str): Name of the metric, defaults to class name if not specified
        psnr (PeakSignalNoiseRatio): TorchMetrics implementation of PSNR calculation
    """

    def __init__(self, device: str, name: Optional[str] = None) -> None:
        """
        Initialize PSNRMetric with specified device and optional name.

        Args:
            device (str): Device to run calculations on. Use 'cuda' for GPU, 'cpu' for CPU,
                         or 'auto' for automatic selection based on availability
            name (Optional[str]): Custom name for the metric. If None, uses class name
        """
        self.name = name if name is not None else self.__class__.__name__
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)

    def __call__(
            self,
            gen_output: torch.Tensor,
            data_object: torch.Tensor,
            **kwargs: Any
    ) -> torch.Tensor:
        """
        Calculate PSNR between generated and ground truth images.

        Args:
            gen_output (torch.Tensor): Generated high-resolution images, shape (B, C, H, W)
            data_object (torch.Tensor): Ground truth high-resolution images, shape (B, C, H, W)
            **kwargs: Additional arguments (not used but included for compatibility)

        Returns:
            torch.Tensor: PSNR value, higher values indicate better quality
        """
        return self.psnr(gen_output, data_object)