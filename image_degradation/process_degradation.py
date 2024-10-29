import cv2
import torch
import random
import math
import numpy as np
from torch.nn import functional as F
from tools.kernels import (
    generate_isotropic_gaussian_kernel,
    generate_anisotropic_gaussian_kernel,
    generate_generalized_gaussian_kernel,
    generate_plateau_gaussian_kernel
)
from image_degradation.tools.utils import np2tensor, tensor2np, filter2D

class ImageDegrader:
    def __init__(
        self,
        gaussian_noise_prob=0.5,
        noise_range=[0.01, 0.1],
        kernel_list=['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 
                    'plateau_iso', 'plateau_aniso'],
        kernel_prob=[0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
        kernel_range=[2, 4],
        blur_sigma=[0.1, 3],
        betag_range=[0.5, 4],
        betap_range=[1, 2],
        second_blur_prob=0.3,
        gaussian_noise_prob2=0.3,
        noise_range2=[0.01, 0.05],
        mode='single_image'
    ):
        self.gaussian_noise_prob = gaussian_noise_prob
        self.noise_range = noise_range
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.kernel_range = kernel_range
        self.blur_sigma = blur_sigma
        self.betag_range = betag_range
        self.betap_range = betap_range
        self.second_blur_prob = second_blur_prob
        self.gaussian_noise_prob2 = gaussian_noise_prob2
        self.noise_range2 = noise_range2
        self.mode = mode

    def _generate_kernel_parameters(self):
        """Generate random parameters for kernel generation"""
        kernel_size = random.choice(range(
            self.kernel_range[0] * 2 + 1,
            self.kernel_range[1] * 2 + 1, 2
        ))
        sigma_x = random.uniform(self.blur_sigma[0], self.blur_sigma[1])
        sigma_y = random.uniform(self.blur_sigma[0], self.blur_sigma[1])
        rotation = random.uniform(-math.pi, math.pi)
        beta_g = random.uniform(self.betag_range[0], self.betag_range[1])
        beta_p = random.uniform(self.betap_range[0], self.betap_range[1])
        
        return kernel_size, sigma_x, sigma_y, rotation, beta_g, beta_p

    def _generate_single_kernel(self, kernel_size=None):
        """Generate a single kernel with optional fixed size"""
        if kernel_size is None:
            kernel_size, sigma_x, sigma_y, rotation, beta_g, beta_p = self._generate_kernel_parameters()
        else:
            _, sigma_x, sigma_y, rotation, beta_g, beta_p = self._generate_kernel_parameters()
            
        kernel_type = random.choices(self.kernel_list, self.kernel_prob)[0]
        
        if kernel_type == 'iso':
            kernel = generate_isotropic_gaussian_kernel(kernel_size, sigma_x)
        elif kernel_type == 'aniso':
            kernel = generate_anisotropic_gaussian_kernel(kernel_size, sigma_x, sigma_y, rotation)
        elif kernel_type == 'generalized_iso':
            kernel = generate_generalized_gaussian_kernel(kernel_size, sigma_x, beta_g)
        elif kernel_type == 'generalized_aniso':
            kernel = generate_anisotropic_gaussian_kernel(kernel_size, sigma_x, sigma_y, rotation)
        elif kernel_type == 'plateau_iso':
            kernel = generate_plateau_gaussian_kernel(kernel_size, sigma_x, beta_p)
        elif kernel_type == 'plateau_aniso':
            kernel = generate_anisotropic_gaussian_kernel(kernel_size, sigma_x, sigma_y, rotation)
        
        return kernel / np.sum(kernel)

    def _create_kernel(self, batch_size=1):
        """Create kernel(s) based on mode"""
        if self.mode == 'batch':
            # Generate kernels with same size for the batch
            kernel_size = random.choice(range(
                self.kernel_range[0] * 2 + 1,
                self.kernel_range[1] * 2 + 1, 2
            ))
            kernels = []
            for _ in range(batch_size):
                kernel = self._generate_single_kernel(kernel_size)
                # Reshape kernel to (H, W, 1) for proper stacking
                kernel = kernel.reshape(kernel_size, kernel_size, 1)
                kernels.append(kernel)
            # Stack kernels along the last dimension
            stacked_kernels = np.stack(kernels, axis=0)
            # Reshape to (B, H, W)
            return torch.FloatTensor(stacked_kernels.squeeze(-1))
        else:
            # Generate single kernel
            kernel = self._generate_single_kernel()
            return torch.FloatTensor(kernel)

    def apply_resize(self, tensor):
        """Apply 2x downscale with random interpolation method"""
        mode = random.choice(['bilinear', 'bicubic', 'area'])
        return F.interpolate(tensor, scale_factor=0.5, mode=mode)

    def apply_noise(self, tensor, noise_prob, noise_range):
        """Apply random noise to tensor"""
        if random.random() < noise_prob:
            noise = torch.randn_like(tensor) * random.uniform(*noise_range)
            tensor = tensor + noise
            return torch.clamp(tensor, 0, 1)
        return tensor

    def _process_tensor(self, tensor):
        """Common processing pipeline for both single images and batches"""
        batch_size = tensor.size(0)
        
        # Generate kernels
        kernel1 = self._create_kernel(batch_size)
        kernel2 = self._create_kernel(batch_size) if self.second_blur_prob > 0 else None
        
        # First degradation
        out = tensor
        
        # Apply transformations
        out = self.apply_resize(out)
        out = filter2D(out, kernel1)
        out = self.apply_noise(out, self.gaussian_noise_prob, self.noise_range)
        
        # Second degradation (optional)
        if kernel2 is not None and random.random() < self.second_blur_prob:
            out = filter2D(out, kernel2)
            out = self.apply_noise(out, self.gaussian_noise_prob2, self.noise_range2)
        
        return out

    def degrade_image(self, input_path, output_path):
        """Process single image from file"""
        if self.mode != 'single_image':
            raise ValueError("This method is only for single image processing. Use process_batch for batches.")
            
        # Read and convert image
        img = cv2.imread(input_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_t = np2tensor(img)
        
        # Process the image
        out = self._process_tensor(img_t)
        
        # Save result
        out_np = tensor2np(out)
        out_np = cv2.cvtColor(out_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, out_np)

    def process_batch(self, hr_batch):
        """Process batch of images during training"""
        if self.mode != 'batch':
            raise ValueError("Degrader must be initialized with mode='batch' for batch processing")
        return self._process_tensor(hr_batch)


def main():
    # Example usage for single image
    degrader_single = ImageDegrader(mode='single_image')
    degrader_single.degrade_image('input_images/input.png', 'output_images/degraded.png')
    
    # Example usage for batch processing
    degrader_batch = ImageDegrader(mode='batch')
    # Создаем случайный батч для демонстрации
    dummy_batch = torch.rand(4, 3, 256, 256)  # batch_size=4, channels=3, height=256, width=256
    lr_batch = degrader_batch.process_batch(dummy_batch)
    print(f"Processed batch shape: {lr_batch.shape}")

if __name__ == '__main__':
    main()
    
# For TRAIN use    
# degrader = ImageDegrader(mode='batch')
# lr_batch = degrader.process_batch(hr_batch)