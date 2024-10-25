import torch
import numpy as np
from torch.nn import functional as F

def np2tensor(np_frame):
    """Convert numpy image to torch tensor"""
    return torch.from_numpy(np.transpose(np_frame, (2, 0, 1))).unsqueeze(0).float() / 255

def tensor2np(tensor):
    """Convert torch tensor to numpy image"""
    return (np.transpose(tensor.detach().squeeze(0).cpu().numpy(), (1, 2, 0))) * 255

def filter2D(img, kernel):
    """Apply 2D filter to image"""
    k = kernel.size(-1)
    b, c, h, w = img.size()
    if k % 2 == 1:
        img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode='reflect')
    else:
        raise ValueError('Wrong kernel size')
    
    ph, pw = img.size()[-2:]
    if kernel.size(0) == 1:
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)
