import torch
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
import torch.nn as nn
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
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


class LPIPSMetric(BaseMetric, nn.Module):  # Наследуем nn.Module
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.Module.__init__(self)  # Инициализируем nn.Module
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Initialize VGG model with pretrained weights
        self.vgg = vgg16(weights=VGG16_Weights.DEFAULT).features[:30].to(device)
        self.vgg.eval()

        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False

        # Normalization for input images
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x):
        """Normalize input images."""
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def get_features(self, x):
        """Extract VGG features."""
        x = self.normalize(x)
        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                features.append(x)
        return features

    def __call__(self, gen_output: torch.Tensor, data_object: torch.Tensor, **kwargs):
        """
        Calculate LPIPS distance between generated and ground truth images.
        Lower values indicate more similar images.

        Args:
            gen_output (Tensor): generated high-resolution images
            data_object (Tensor): ground truth high-resolution images
        Returns:
            float: LPIPS distance value
        """
        # Extract features
        gen_features = self.get_features(gen_output)
        target_features = self.get_features(data_object)

        # Calculate normalized distances
        dist = 0
        for gf, tf in zip(gen_features, target_features):
            # Normalize features along channel dimension
            gf = gf / (torch.norm(gf, dim=1, keepdim=True) + 1e-10)
            tf = tf / (torch.norm(tf, dim=1, keepdim=True) + 1e-10)

            # Calculate mean squared error
            dist += torch.mean((gf - tf) ** 2)

        return dist / len(gen_features)
