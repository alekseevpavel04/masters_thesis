"""
This module provides various loss functions for training super-resolution models.

Available loss functions:
    - AdversarialLoss: Adversarial loss for GAN-based models.
    - PixelLoss: Pixel-wise L1 loss for image reconstruction tasks.
"""

from src.loss.adversarial_loss import AdversarialLoss
from src.loss.pixel_loss import PixelLoss