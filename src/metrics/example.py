import torch
import torch.nn.functional as F
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from src.metrics.base_metric import BaseMetric


class PSNRMetric(BaseMetric):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)

    def __call__(self, gen_output: torch.Tensor, data_object: torch.Tensor, **kwargs):
        """
        Calculate PSNR between generated and ground truth images

        Args:
            gen_output (Tensor): generated high-resolution images
            data_object (Tensor): ground truth high-resolution images
        Returns:
            float: PSNR value
        """
        return self.psnr(gen_output, data_object)


class SSIMMetric(BaseMetric):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    def __call__(self, gen_output: torch.Tensor, data_object: torch.Tensor, **kwargs):
        """
        Calculate SSIM between generated and ground truth images

        Args:
            gen_output (Tensor): generated high-resolution images
            data_object (Tensor): ground truth high-resolution images
        Returns:
            float: SSIM value
        """
        return self.ssim(gen_output, data_object)


class ContentLossMetric(BaseMetric):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device if device != "auto" else "cuda" if torch.cuda.is_available() else "cpu"

    def __call__(self, gen_output: torch.Tensor, data_object: torch.Tensor, **kwargs):
        """
        Calculate L1 loss between generated and ground truth images

        Args:
            gen_output (Tensor): generated high-resolution images
            data_object (Tensor): ground truth high-resolution images
        Returns:
            float: L1 loss value
        """
        return F.l1_loss(gen_output, data_object)
