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

    def process_frame(self, img, frame_cache):
        """Process single frame with cuDNN optimizations"""
        h, w, c = img.shape
        block_size = 64
        overlap = 8
        stride = block_size - overlap

        enhanced_img = np.zeros((h * 2, w * 2, c), dtype=np.uint8)

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y_start = max(y - overlap, 0)
                x_start = max(x - overlap, 0)
                y_end = min(y_start + block_size, h)
                x_end = min(x_start + block_size, w)

                block = img[y_start:y_end, x_start:x_end]
                block = block.astype(np.float32) / 255.
                block = torch.from_numpy(block).permute(2, 0, 1).unsqueeze(0)

                # Optimize memory layout for CUDA
                if self.device.type == 'cuda':
                    block = block.to(memory_format=self.memory_format)

                block = block.to(self.device)

                with torch.amp.autocast(device_type=self.device.type, enabled=self.device.type == 'cuda'):
                    with torch.no_grad():
                        output = self.model(block)

                output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
                output = (output * 255.0).clip(0, 255).astype(np.uint8)

                out_y_start = y_start * 2
                out_x_start = x_start * 2
                out_y_end = out_y_start + output.shape[0]
                out_x_end = out_x_start + output.shape[1]

                enhanced_img[out_y_start:out_y_end, out_x_start:out_x_end] = output

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return enhanced_img

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