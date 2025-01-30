import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image
import io
import random

class ImageDegradationPipeline_base:
    def __init__(
            self,
            jpeg_quality=[10, 50],
            mode='single_image',
            device="cpu"
    ):
        self.device = device
        self.jpeg_quality = jpeg_quality
        self.mode = mode
        self.to_pil = ToPILImage()
        self.to_tensor = ToTensor()

    def apply_resize(self, tensor):
        """Apply 2x downscale with bilinear interpolation"""
        out = F.interpolate(tensor, scale_factor=0.5, mode='bilinear', align_corners=False)
        return torch.clamp(out, 0, 1)

    def apply_jpeg_compression(self, tensor):
        """Apply JPEG compression to tensor"""
        batch_size = tensor.size(0)
        channels = tensor.size(1)
        compressed_tensors = []

        for i in range(batch_size):
            # Convert single tensor from batch to PIL Image
            img = self.to_pil(tensor[i].cpu())

            # Randomly select JPEG quality within the specified range
            quality = random.randint(self.jpeg_quality[0], self.jpeg_quality[1])

            # Apply JPEG compression
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            compressed_img = Image.open(buffer)

            # Convert back to tensor
            compressed_tensor = self.to_tensor(compressed_img)
            compressed_tensors.append(compressed_tensor)

        # Stack all compressed tensors back into a batch
        out = torch.stack(compressed_tensors).to(self.device)
        return torch.clamp(out, 0, 1)

    def _process_tensor(self, tensor):
        """Process either single image or batch of images"""
        # Ensure tensor is on correct device
        tensor = tensor.to(self.device)

        # Apply transformations
        out = self.apply_resize(tensor)
        out = self.apply_jpeg_compression(out)

        return out

    def degrade_image(self, input_path, output_path):
        """Process single image from file"""
        if self.mode != 'single_image':
            raise ValueError("This method is only for single image processing. Use process_batch for batches.")

        # Read and convert image
        img = Image.open(input_path).convert('RGB')
        img_t = self.to_tensor(img).unsqueeze(0)

        # Process the image
        out = self._process_tensor(img_t)

        # Save result
        out_img = self.to_pil(out.squeeze(0).cpu())
        out_img.save(output_path, 'JPEG')

    def process_batch(self, hr_batch):
        """Process batch of images during training"""
        if self.mode != 'batch':
            raise ValueError("Degrader must be initialized with mode='batch' for batch processing")

        # Ensure all operations are performed on the correct device
        if not isinstance(hr_batch, torch.Tensor):
            hr_batch = torch.tensor(hr_batch, device=self.device)
        elif hr_batch.device != self.device:
            hr_batch = hr_batch.to(self.device)

        return self._process_tensor(hr_batch)