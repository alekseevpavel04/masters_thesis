import torch
import torch.nn as nn


class AdversarialLoss(nn.Module):
    """
    Adversarial loss for GAN-based models. This loss is used to train the generator
    to produce realistic images by comparing its predictions with real/fake labels.

    Attributes:
        criterion (nn.BCEWithLogitsLoss): Binary Cross-Entropy loss with logits.
    """

    def __init__(self):
        """
        Initializes the AdversarialLoss module with BCEWithLogitsLoss.
        """
        super(AdversarialLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, pred, target_is_real):
        """
        Computes the adversarial loss.

        Args:
            pred (torch.Tensor): The predicted output from the discriminator.
            target_is_real (bool): Whether the target is real (True) or fake (False).

        Returns:
            torch.Tensor: The computed adversarial loss.
        """
        # Create a target tensor filled with 1s if target is real, else 0s
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return self.criterion(pred, target)