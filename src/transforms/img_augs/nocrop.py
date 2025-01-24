import torch
import torch.nn as nn

class NoCrop(nn.Module):
    """
    A stub module that does not modify the input tensor.

    This class is designed to mimic the interface of a cropping module but simply returns the input tensor
    unchanged. It can be used as a placeholder or for testing purposes when no cropping is required.

    Attributes:
        None
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes the NoCrop module.

        Args:
            *args: Variable length argument list (ignored).
            **kwargs: Arbitrary keyword arguments (ignored).

        Note:
            This module does not perform any operations on the input tensor.
        """
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the input tensor unchanged.

        Args:
            x (torch.Tensor): The input tensor with any shape.

        Returns:
            torch.Tensor: The same input tensor, unmodified.
        """
        # Simply return the input tensor without any modifications
        return x