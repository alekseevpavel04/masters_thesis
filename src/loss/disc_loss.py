import torch
import torch.nn as nn
from src.loss.adversarial_loss import AdversarialLoss


class DiscriminatorCriterion(nn.Module):
    """
    Combines the computation of adversarial losses for both fake and real predictions
    into a single forward pass.

    Attributes:
        criterion (AdversarialLoss): The base adversarial loss module for individual computations.
    """

    def __init__(self):
        """
        Initializes the CombinedAdversarialLoss module using the base AdversarialLoss.
        """
        super(DiscriminatorCriterion, self).__init__()
        self.criterion = AdversarialLoss()

    def forward(self, fake_pred, real_pred, batch):
        """
        Computes the combined adversarial loss for both fake and real predictions.

        Args:
            fake_pred (torch.Tensor): The discriminator's predictions for fake data
            real_pred (torch.Tensor): The discriminator's predictions for real data

        Returns:
            torch.Tensor: The average of fake and real losses
        """
        # Calculate discriminator losses for fake and real predictions
        disc_loss_fake = self.criterion(fake_pred, False)
        disc_loss_real = self.criterion(real_pred, True)

        # Return the average loss
        return (disc_loss_fake + disc_loss_real) * 0.5