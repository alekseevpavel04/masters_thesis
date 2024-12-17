import torch
import torch.nn as nn


class PixelLoss(nn.Module):
    def __init__(self):
        super(PixelLoss, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, sr, hr):
        return self.l1(sr, hr)
