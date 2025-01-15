import torch
import torch.backends.cudnn as cudnn
import numpy as np


class FrameProcessor:
    def __init__(self, model, device):
        self.model = model
        self.device = device

        # Enable cuDNN optimizations if CUDA is available
        if self.device.type == 'cuda':
            # Enable cuDNN auto-tuner
            cudnn.benchmark = True
            # Enable cuDNN deterministic mode for reproducibility
            cudnn.deterministic = False
            # Enable TensorCore operations if available
            cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True

        # Set optimal memory formats for CUDA tensors
        self.memory_format = torch.channels_last if device.type == 'cuda' else torch.contiguous_format
        self.model = self.model.to(memory_format=self.memory_format)

    def process_batch(self, frames, frame_cache):
        """Process batch of frames with cuDNN optimizations"""
        batch = np.array(frames).astype(np.float32) / 255.
        batch = torch.from_numpy(batch).permute(0, 3, 1, 2)

        # Optimize memory layout for CUDA
        if self.device.type == 'cuda':
            batch = batch.to(memory_format=self.memory_format)

        batch = batch.to(self.device)

        with torch.amp.autocast(device_type=self.device.type, enabled=self.device.type == 'cuda'):
            with torch.no_grad():
                output = self.model(batch)

        output = output.permute(0, 2, 3, 1).cpu().numpy()
        output = (output * 255.0).clip(0, 255).astype(np.uint8)

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return output