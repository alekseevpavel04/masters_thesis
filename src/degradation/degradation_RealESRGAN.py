import torch
import numpy as np
import cv2
import math
import random
from torch.nn import functional as F
from scipy import special

class ImageDegradationPipeline_RealESRGAN:
    def __init__(
        self,
        # First degradation settings
        kernel_list=['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
        kernel_prob=[0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
        blur_sigma=[0.1, 10],
        betag_range=[0.5, 4],
        betap_range=[1, 2],
        sinc_prob=0.1,
        # Second degradation settings
        second_blur_prob=0.3,
        kernel_list2=['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
        kernel_prob2=[0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
        blur_sigma2=[0.1, 3],
        betag_range2=[0.5, 4],
        betap_range2=[1, 2],
        sinc_prob2=0.1,
        # Final sinc settings
        final_sinc_prob=0.2,
        mode='single_image',
        device="cpu"
    ):
        self.device = device
        self.mode = mode

        # First degradation params
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.betag_range = betag_range
        self.betap_range = betap_range
        self.sinc_prob = sinc_prob

        # Second degradation params
        self.second_blur_prob = second_blur_prob
        self.kernel_list2 = kernel_list2
        self.kernel_prob2 = kernel_prob2
        self.blur_sigma2 = blur_sigma2
        self.betag_range2 = betag_range2
        self.betap_range2 = betap_range2
        self.sinc_prob2 = sinc_prob2

        # Final sinc params
        self.final_sinc_prob = final_sinc_prob

        # Kernel size range (7 to 21)
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]
        self.pulse_tensor = torch.zeros(21, 21).float()
        self.pulse_tensor[10, 10] = 1

    def _generate_kernel(self, kernel_size, kernel_list, kernel_prob, blur_sigma,
                         betag_range, betap_range, sinc_prob, batch_size=1):
        """Generate a random kernel based on settings"""
        if batch_size > 1:
            # Для батча генерируем несколько ядер
            kernels = []
            for _ in range(batch_size):
                kernel = self._generate_single_kernel(
                    kernel_size, kernel_list, kernel_prob,
                    blur_sigma, betag_range, betap_range, sinc_prob
                )
                kernels.append(kernel)
            # Объединяем ядра в один тензор размера (batch_size, 21, 21)
            return torch.stack(kernels, dim=0)
        else:
            # Для одного изображения генерируем одно ядро
            return self._generate_single_kernel(
                kernel_size, kernel_list, kernel_prob,
                blur_sigma, betag_range, betap_range, sinc_prob
            ).unsqueeze(0)

    def _generate_single_kernel(self, kernel_size, kernel_list, kernel_prob, blur_sigma,
                                betag_range, betap_range, sinc_prob):
        """Generate a single kernel"""
        if random.random() < sinc_prob:
            if kernel_size < 13:
                omega_c = random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = random.uniform(np.pi / 5, np.pi)
            kernel = self._circular_lowpass_kernel(omega_c, kernel_size)
        else:
            sigma_x = random.uniform(blur_sigma[0], blur_sigma[1])
            sigma_y = random.uniform(blur_sigma[0], blur_sigma[1])
            rotation = random.uniform(-math.pi, math.pi)
            beta_g = random.uniform(betag_range[0], betag_range[1])
            beta_p = random.uniform(betap_range[0], betap_range[1])

            kernel_type = random.choices(kernel_list, kernel_prob)[0]

            if kernel_type == 'iso':
                kernel = self._generate_isotropic_gaussian_kernel(kernel_size, sigma_x)
            elif kernel_type == 'aniso':
                kernel = self._generate_anisotropic_gaussian_kernel(kernel_size, sigma_x, sigma_y, rotation)
            elif kernel_type == 'generalized_iso':
                kernel = self._generate_generalized_gaussian_kernel(kernel_size, sigma_x, beta_g)
            elif kernel_type == 'generalized_aniso':
                kernel = self._generate_anisotropic_gaussian_kernel(kernel_size, sigma_x, sigma_y, rotation)
            elif kernel_type == 'plateau_iso':
                kernel = self._generate_plateau_gaussian_kernel(kernel_size, sigma_x, beta_p)
            elif kernel_type == 'plateau_aniso':
                kernel = self._generate_anisotropic_gaussian_kernel(kernel_size, sigma_x, sigma_y, rotation)

        # Pad kernel to 21x21
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
        return torch.FloatTensor(kernel / np.sum(kernel)).to(self.device)

    def _process_tensor(self, tensor):
        """Process input tensor with RealESRGAN degradation pipeline"""
        tensor = tensor.to(self.device)
        batch_size = tensor.size(0)

        # First degradation
        kernel_size = random.choice(self.kernel_range)
        kernel1 = self._generate_kernel(
            kernel_size, self.kernel_list, self.kernel_prob,
            self.blur_sigma, self.betag_range, self.betap_range,
            self.sinc_prob, batch_size=batch_size
        )

        out = tensor
        # Apply first degradation
        out = F.interpolate(out, scale_factor=0.5, mode='bicubic', align_corners=False)
        out = self.filter2D(out, kernel1)

        # Second degradation (optional)
        if random.random() < self.second_blur_prob:
            kernel_size2 = random.choice(self.kernel_range)
            kernel2 = self._generate_kernel(
                kernel_size2, self.kernel_list2, self.kernel_prob2,
                self.blur_sigma2, self.betag_range2, self.betap_range2,
                self.sinc_prob2, batch_size=batch_size
            )
            out = self.filter2D(out, kernel2)

        # Final sinc degradation (optional)
        if random.random() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = random.uniform(np.pi / 3, np.pi)
            final_kernel = self._circular_lowpass_kernel(omega_c, kernel_size)
            pad_size = (21 - kernel_size) // 2
            final_kernel = np.pad(final_kernel, ((pad_size, pad_size), (pad_size, pad_size)))
            final_kernel = torch.FloatTensor(final_kernel).to(self.device)
            final_kernel = final_kernel.expand(batch_size, -1, -1)  # Расширяем до размера батча
            out = self.filter2D(out, final_kernel)

        return torch.clamp(out, 0, 1)

    def filter2D(self, img, kernel):
        """Apply 2D filter to image with proper batch handling"""
        b, c, h, w = img.size()
        k = kernel.size(-1)

        # Проверяем, что ядро нечетного размера
        if k % 2 == 1:
            img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode='reflect')
        else:
            raise ValueError('Kernel size must be odd')

        ph, pw = img.size()[-2:]

        # Преобразуем изображение и ядро для batch conv2d
        img = img.reshape(1, b * c, ph, pw)

        # Расширяем ядро для каждого канала в батче
        if len(kernel.size()) == 3:  # если ядро размера (b, k, k)
            kernel = kernel.unsqueeze(1).repeat(1, c, 1, 1)  # -> (b, c, k, k)
            kernel = kernel.reshape(b * c, 1, k, k)
        else:  # если ядро размера (k, k)
            kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(b * c, 1, 1, 1)

        # Применяем свертку
        out = F.conv2d(img, kernel, groups=b * c)

        # Возвращаем к исходной форме
        out = out.reshape(b, c, h, w)

        return out

    def _circular_lowpass_kernel(self, omega_c, kernel_size):
        """Generate sinc kernel"""
        v = np.arange(-(kernel_size - 1) // 2, (kernel_size + 1) // 2)
        x, y = np.meshgrid(v, v)
        r = np.sqrt(x ** 2 + y ** 2)
        kernel = omega_c * special.j1(omega_c * r) / (2 * np.pi * r)
        kernel[r == 0] = omega_c ** 2 / (4 * np.pi)
        kernel = kernel / np.sum(kernel)
        return kernel

    def _generate_isotropic_gaussian_kernel(self, kernel_size, sigma):
        """Generate isotropic Gaussian kernel"""
        center = kernel_size // 2
        x = np.arange(kernel_size) - center
        x_2d, y_2d = np.meshgrid(x, x)
        kernel = np.exp(-(x_2d ** 2 + y_2d ** 2) / (2 * sigma ** 2))
        return kernel

    def _generate_anisotropic_gaussian_kernel(self, kernel_size, sigma_x, sigma_y, rotation):
        """Generate anisotropic Gaussian kernel"""
        center = kernel_size // 2
        x = np.arange(kernel_size) - center
        x_2d, y_2d = np.meshgrid(x, x)

        x_rot = x_2d * np.cos(rotation) - y_2d * np.sin(rotation)
        y_rot = x_2d * np.sin(rotation) + y_2d * np.cos(rotation)

        kernel = np.exp(-(x_rot ** 2 / (2 * sigma_x ** 2) + y_rot ** 2 / (2 * sigma_y ** 2)))
        return kernel

    def _generate_generalized_gaussian_kernel(self, kernel_size, sigma, beta):
        """Generate generalized Gaussian kernel"""
        center = kernel_size // 2
        x = np.arange(kernel_size) - center
        x_2d, y_2d = np.meshgrid(x, x)
        kernel = np.exp(-((x_2d ** 2 + y_2d ** 2) / (2 * sigma ** 2)) ** beta)
        return kernel

    def _generate_plateau_gaussian_kernel(self, kernel_size, sigma, beta):
        """Generate plateau-shaped Gaussian kernel"""
        center = kernel_size // 2
        x = np.arange(kernel_size) - center
        x_2d, y_2d = np.meshgrid(x, x)
        r = np.sqrt(x_2d ** 2 + y_2d ** 2)
        kernel = 1 / (1 + (r / sigma) ** beta)
        return kernel


    def process_batch(self, hr_batch):
        """Process batch of images"""
        if not isinstance(hr_batch, torch.Tensor):
            hr_batch = torch.tensor(hr_batch, device=self.device)
        elif hr_batch.device != self.device:
            hr_batch = hr_batch.to(self.device)

        if self.mode != 'batch':
            raise ValueError("Degrader must be initialized with mode='batch' for batch processing")
        return self._process_tensor(hr_batch)

    def degrade_image(self, input_path, output_path):
        """Process single image from file"""
        if self.mode != 'single_image':
            raise ValueError("This method is only for single image processing. Use process_batch for batches.")

        img = cv2.imread(input_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_t = torch.from_numpy(np.transpose(img, (2, 0, 1))).unsqueeze(0).float() / 255

        out = self._process_tensor(img_t)

        out_np = (np.transpose(out.detach().squeeze(0).cpu().numpy(), (1, 2, 0)) * 255).astype(np.uint8)
        out_np = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, out_np)
