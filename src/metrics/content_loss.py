import torch
import torch.nn.functional as F
from typing import Optional, Any

class ContentLossMetric:
    def __init__(self, device: str, name: Optional[str] = None) -> None:
        self.name = name if name is not None else self.__class__.__name__
        self.device = device if device != "auto" else "cuda" if torch.cuda.is_available() else "cpu"

    def __call__(
            self,
            gen_output: Optional[torch.Tensor] = None,
            data_object: Optional[torch.Tensor] = None,
            diff_output: Optional[torch.Tensor] = None,
            **kwargs: Any
    ) -> torch.Tensor:
        """
        Calculate L1 loss between generated and ground truth images, or using diff_output.

        Args:
            gen_output (Optional[torch.Tensor]): Generated high-resolution images
            data_object (Optional[torch.Tensor]): Ground truth high-resolution images
            diff_output (Optional[torch.Tensor]): Alternative input when gen_output is not available
            **kwargs: Additional arguments

        Returns:
            torch.Tensor: L1 loss value
        """
        if diff_output is not None:
            gen_output = diff_output

        if gen_output is not None and data_object is not None:
            return F.l1_loss(gen_output, data_object)

        else:
            raise ValueError("Either (gen_output, data_object) or diff_output must be provided")