import torch
import numpy as np

class FrameProcessor:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def process_frame(self, img, frame_cache):
        """Process single frame"""
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
                block = block.to(self.device)

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
        """Process batch of frames"""
        batch = np.array(frames).astype(np.float32) / 255.
        batch = torch.from_numpy(batch).permute(0, 3, 1, 2)
        batch = batch.to(self.device)

        with torch.no_grad():
            output = self.model(batch)

        output = output.permute(0, 2, 3, 1).cpu().numpy()
        output = (output * 255.0).clip(0, 255).astype(np.uint8)

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return output