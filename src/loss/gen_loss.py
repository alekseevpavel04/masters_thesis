import torch
import torch.nn as nn
from src.loss.adversarial_loss import AdversarialLoss
from src.loss.pixel_loss import PixelLoss


class GeneratorCriterion(nn.Module):
    """
    Combined loss criterion for generator training that includes both
    adversarial loss and pixel-wise reconstruction loss.

    Attributes:
        adv_loss (AdversarialLoss): Adversarial loss component
        pixel_loss (PixelLoss): Pixel-wise loss component
        content_weight (float): Weight for the pixel loss component
    """

    def __init__(self, content_weight=1.0):
        """
        Initialize the combined generator criterion.

        Args:
            content_weight (float): Weight for the pixel loss component
        """
        super(GeneratorCriterion, self).__init__()
        self.adv_loss = AdversarialLoss()
        self.pixel_loss = PixelLoss()
        self.content_weight = content_weight

    def forward(self, disc_fake, batch):
        """
        Compute the combined generator loss.

        Args:
            disc_fake (torch.Tensor): Discriminator output for fake images
            gen_output (torch.Tensor): Generator output images
            target (torch.Tensor): Target/ground truth images

        Returns:
            torch.Tensor: Combined generator loss
        """
        # Calculate adversarial loss
        adv_loss = self.adv_loss(disc_fake, True)

        # Calculate pixel-wise content loss
        content_loss = self.pixel_loss(batch["gen_output"], batch["data_object"])

        # Combine losses
        total_loss = adv_loss + self.content_weight * content_loss

        return total_loss