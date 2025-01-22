'''
Degradation method from APISR
https://github.com/Kiteretsu77/APISR/
'''


import cv2
import torch
import random
import math
import numpy as np
from torch.nn import functional as F

class ImageDegradationPipeline_APISR:
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
        mode='single_image',
        device = "cpu"
    ):
        self.device = device
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
            # Reshape to (B, H, W) and move to correct device
            return torch.FloatTensor(stacked_kernels.squeeze(-1)).to(self.device)
        else:
            # Generate single kernel and move to correct device
            kernel = self._generate_single_kernel()
            return torch.FloatTensor(kernel).to(self.device)

    def apply_resize(self, tensor):
        """Apply 2x downscale with random interpolation method"""
        mode = random.choice(['bilinear', 'bicubic', 'area'])
        out = F.interpolate(tensor, scale_factor=0.5, mode=mode)
        return torch.clamp(out, 0, 1)  # Add clipping after resize

    def apply_noise(self, tensor, noise_prob, noise_range):
        """Apply random noise to tensor"""
        if random.random() < noise_prob:
            noise = torch.randn_like(tensor).to(self.device) * random.uniform(*noise_range)
            tensor = tensor + noise
            return torch.clamp(tensor, 0, 1)
        return tensor

    def _process_tensor(self, tensor):
        """Common processing pipeline for both single images and batches"""
        # Ensure tensor is on correct device
        tensor = tensor.to(self.device)
        batch_size = tensor.size(0)

        # Generate kernels (they will be on the correct device from _create_kernel)
        kernel1 = self._create_kernel(batch_size)
        kernel2 = self._create_kernel(batch_size) if self.second_blur_prob > 0 else None

        # First degradation
        out = tensor

        # Apply transformations
        out = self.apply_resize(out)
        out = filter2D(out, kernel1)
        out = torch.clamp(out, 0, 1)  # Add clipping in case if random.random() >= noise_prob
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
        # Ensure all operations are performed on the correct device
        if not isinstance(hr_batch, torch.Tensor):
            hr_batch = torch.tensor(hr_batch, device=self.device)
        elif hr_batch.device != self.device:
            hr_batch = hr_batch.to(self.device)
        """Process batch of images during training"""
        if self.mode != 'batch':
            raise ValueError("Degrader must be initialized with mode='batch' for batch processing")
        return self._process_tensor(hr_batch)


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


def generate_isotropic_gaussian_kernel(kernel_size, sigma):
    """Generate isotropic Gaussian kernel"""
    center = kernel_size // 2
    x = np.arange(kernel_size) - center
    x_2d, y_2d = np.meshgrid(x, x)
    kernel = np.exp(-(x_2d ** 2 + y_2d ** 2) / (2 * sigma ** 2))
    return kernel


def generate_anisotropic_gaussian_kernel(kernel_size, sigma_x, sigma_y, rotation):
    """Generate anisotropic Gaussian kernel"""
    center = kernel_size // 2
    x = np.arange(kernel_size) - center
    x_2d, y_2d = np.meshgrid(x, x)

    x_rot = x_2d * np.cos(rotation) - y_2d * np.sin(rotation)
    y_rot = x_2d * np.sin(rotation) + y_2d * np.cos(rotation)

    kernel = np.exp(-(x_rot ** 2 / (2 * sigma_x ** 2) + y_rot ** 2 / (2 * sigma_y ** 2)))
    return kernel


def generate_generalized_gaussian_kernel(kernel_size, sigma_x, beta):
    """Generate generalized Gaussian kernel"""
    center = kernel_size // 2
    x = np.arange(kernel_size) - center
    x_2d, y_2d = np.meshgrid(x, x)
    kernel = np.exp(-((x_2d ** 2 + y_2d ** 2) / (2 * sigma_x ** 2)) ** beta)
    return kernel


def generate_plateau_gaussian_kernel(kernel_size, sigma_x, beta):
    """Generate plateau-shaped Gaussian kernel"""
    center = kernel_size // 2
    x = np.arange(kernel_size) - center
    x_2d, y_2d = np.meshgrid(x, x)
    r = np.sqrt(x_2d ** 2 + y_2d ** 2)
    kernel = 1 / (1 + (r / sigma_x) ** beta)
    return kernel


def main():
    # Example usage for single image
    degrader_single = ImageDegradationPipeline_APISR(mode='single_image')
    degrader_single.degrade_image('input_images/input.png', 'output_images/degraded.png')
    
    # Example usage for batch processing
    degrader_batch = ImageDegradationPipeline_APISR(mode='batch')
    # Создаем случайный батч для демонстрации
    dummy_batch = torch.rand(4, 3, 256, 256)  # batch_size=4, channels=3, height=256, width=256
    lr_batch = degrader_batch.process_batch(dummy_batch)
    print(f"Processed batch shape: {lr_batch.shape}")

if __name__ == '__main__':
    main()
    
