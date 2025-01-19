import torch
import torch.nn as nn
from typing import Optional, Any, List
from torchvision.models import vgg16, VGG16_Weights


class LPIPSMetric(nn.Module):
    """
    A class to calculate Learned Perceptual Image Patch Similarity (LPIPS) metric.

    LPIPS uses deep features from pretrained networks (VGG16 in this case) to compute
    perceptual similarity between images. This metric has been shown to correlate better
    with human perception compared to traditional metrics like PSNR or SSIM.

    Attributes:
        name (str): Name of the metric, defaults to class name if not specified
        device (str): Device where computations will be performed
        vgg (nn.Module): Pretrained VGG16 model for feature extraction
        mean (torch.Tensor): Mean values for input normalization
        std (torch.Tensor): Standard deviation values for input normalization
    """

    def __init__(self, device: str, name: Optional[str] = None) -> None:
        """
        Initialize LPIPSMetric with specified device and optional name.

        Args:
            device (str): Device to run calculations on. Use 'cuda' for GPU, 'cpu' for CPU,
                         or 'auto' for automatic selection based on availability
            name (Optional[str]): Custom name for the metric. If None, uses class name
        """
        super().__init__()
        self.name = name if name is not None else self.__class__.__name__
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Initialize VGG model with pretrained weights
        self.vgg = vgg16(weights=VGG16_Weights.DEFAULT).features[:30].to(device)
        self.vgg.eval()

        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False

        # Register normalization parameters as buffers
        self.register_buffer(
            'mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input images using ImageNet statistics.

        Args:
            x (torch.Tensor): Input images, shape (B, C, H, W)

        Returns:
            torch.Tensor: Normalized images
        """
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def get_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract VGG features from input images.

        Args:
            x (torch.Tensor): Input images, shape (B, C, H, W)

        Returns:
            List[torch.Tensor]: List of feature maps at different VGG layers
        """
        x = self.normalize(x)
        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                features.append(x)
        return features

    def __call__(
            self,
            gen_output: torch.Tensor,
            data_object: torch.Tensor,
            **kwargs: Any
    ) -> torch.Tensor:
        """
        Calculate LPIPS distance between generated and ground truth images.

        Args:
            gen_output (torch.Tensor): Generated high-resolution images, shape (B, C, H, W)
            data_object (torch.Tensor): Ground truth high-resolution images, shape (B, C, H, W)
            **kwargs: Additional arguments (not used but included for compatibility)

        Returns:
            torch.Tensor: LPIPS distance value, lower values indicate more perceptually
                         similar images
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