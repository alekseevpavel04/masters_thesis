import torch
from typing import Optional, Any
from torchmetrics.image import StructuralSimilarityIndexMeasure


class SSIMMetric:
    def __init__(self, device: str, name: Optional[str] = None) -> None:
        self.name = name if name is not None else self.__class__.__name__
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    def __call__(
            self,
            gen_output: Optional[torch.Tensor] = None,
            data_object: Optional[torch.Tensor] = None,
            diff_output: Optional[torch.Tensor] = None,
            **kwargs: Any
    ) -> torch.Tensor:
        """
        Calculate SSIM between images or use provided diff_output.

        Args:
            gen_output (Optional[torch.Tensor]): Generated high-resolution images
            data_object (Optional[torch.Tensor]): Ground truth high-resolution images
            diff_output (Optional[torch.Tensor]): Alternative input when gen_output is not available
            **kwargs: Additional arguments

        Returns:
            torch.Tensor: SSIM value
        """
        if diff_output is not None:
            gen_output = diff_output

        if gen_output is None or data_object is None:
            raise ValueError("Either (gen_output, data_object) or diff_output must be provided")

        return self.ssim(gen_output, data_object)