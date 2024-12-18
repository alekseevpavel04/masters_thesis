import torch
from torch import nn

class CentralCrop(nn.Module):
    """
    Central Crop for 2D Tensors based on given width and height.
    """

    def __init__(self, target_width, target_height):
        """
        Args:
            target_width (int): Целевая ширина для обрезки.
            target_height (int): Целевая высота для обрезки.
        """
        super().__init__()
        self.target_width = target_width
        self.target_height = target_height

    def forward(self, x):
        """
        Args:
            x (Tensor): Входной 2D тензор. Последние два измерения - height x width.
        Returns:
            x (Tensor): Центрально обрезанный тензор.
        """
        if x.ndim < 2:
            raise ValueError("Input tensor must have at least 2 dimensions (height and width).")

        h, w = x.shape[-2], x.shape[-1]  # Извлекаем высоту и ширину тензора

        # Проверяем, что размеры тензора больше или равны целевым
        if h < self.target_height or w < self.target_width:
            raise ValueError(
                f"Input tensor is smaller than target crop size: "
                f"input size ({h}, {w}), target size ({self.target_height}, {self.target_width})"
            )

        # Вычисляем центральные координаты
        start_h = (h - self.target_height) // 2
        start_w = (w - self.target_width) // 2

        # Выполняем центральное обрезание
        x = x[..., start_h:start_h + self.target_height, start_w:start_w + self.target_width]
        return x
