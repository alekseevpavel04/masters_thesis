import torch
import random
import numpy as np
import torch.nn.functional as F
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
import math


class ImageDegradationPipeline_RealESRGAN:
    def __init__(
            self,
            scale=2,
            device='cuda',
            kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
            kernel_prob = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
            blur_sigma = [0.2, 3],
            betag_range = [0.5, 4],
            betap_range = [1, 2],
            sinc_prob = 0.1,
            kernel_list2=['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
            kernel_prob2=[0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
            blur_sigma2=[0.2, 1.5],
            betag_range2=[0.5, 4],
            betap_range2=[1, 2],
            sinc_prob2=0.1,
            resize_prob=[0.2, 0.7, 0.1],
            resize_range=[0.15, 1.5],
            gaussian_noise_prob=0.5,
            noise_range=[1, 30],
            poisson_scale_range=[0.05, 3],
            gray_noise_prob=0.4,
            jpeg_range=[30, 95],
            resize_prob2=[0.3, 0.4, 0.3],
            resize_range2=[0.3, 1.2],
            gaussian_noise_prob2=0.5,
            noise_range2=[1, 25],
            poisson_scale_range2=[0.05, 2.5],
            gray_noise_prob2=0.4,
            jpeg_range2=[30, 95],
            final_sinc_prob=0.8,
            second_blur_prob=0.8

    ):
        self.scale = scale
        self.device = device

        # Initialize components
        self.jpeger = DiffJPEG(differentiable=True).to(device)  # Changed to differentiable
        self.usm_sharpener = USMSharp().to(device)

        # Default parameters from RealESRGAN
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]
        self.pulse_tensor = torch.zeros(21, 21).float().to(device)
        self.pulse_tensor[10, 10] = 1

        # First degradation parameters
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.betag_range = betag_range
        self.betap_range = betap_range
        self.sinc_prob = sinc_prob

        # Second degradation parameters
        self.kernel_list2 = kernel_list2
        self.kernel_prob2 = kernel_prob2
        self.blur_sigma2 = blur_sigma2
        self.betag_range2 = betag_range2
        self.betap_range2 = betap_range2
        self.sinc_prob2 = sinc_prob2

        # Other parameters
        self.resize_prob = resize_prob
        self.resize_range = resize_range
        self.gaussian_noise_prob = gaussian_noise_prob
        self.noise_range = noise_range
        self.poisson_scale_range = poisson_scale_range
        self.gray_noise_prob = gray_noise_prob
        self.jpeg_range = jpeg_range

        self.resize_prob2 = resize_prob2
        self.resize_range2 = resize_range2
        self.gaussian_noise_prob2 = gaussian_noise_prob2
        self.noise_range2 = noise_range2
        self.poisson_scale_range2 = poisson_scale_range2
        self.gray_noise_prob2 = gray_noise_prob2
        self.jpeg_range2 = jpeg_range2
        self.final_sinc_prob = final_sinc_prob
        self.second_blur_prob = second_blur_prob

    def generate_kernel1(self, batch_size):
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma,
                [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)

        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
        kernel = torch.FloatTensor(kernel).to(self.device)
        return kernel.unsqueeze(0).repeat(batch_size, 1, 1)

    def generate_kernel2(self, batch_size):
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob2:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2,
                [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))
        kernel2 = torch.FloatTensor(kernel2).to(self.device)
        return kernel2.unsqueeze(0).repeat(batch_size, 1, 1)

    def generate_sinc_kernel(self, batch_size):
        if np.random.uniform() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel).to(self.device)
        else:
            sinc_kernel = self.pulse_tensor.clone()  # Use clone instead of direct assignment
        return sinc_kernel.unsqueeze(0).repeat(batch_size, 1, 1)

    def process_batch(self, gt_batch):
        batch_size = gt_batch.size(0)
        gt = gt_batch
        gt_usm = self.usm_sharpener(gt)

        # Generate kernels with proper batch size
        kernel1 = self.generate_kernel1(batch_size)
        kernel2 = self.generate_kernel2(batch_size)
        sinc_kernel = self.generate_sinc_kernel(batch_size)

        ori_h, ori_w = gt.size()[2:4]

        # ----------------------- The first degradation process ----------------------- #
        # blur
        out = torch.zeros_like(gt_usm)
        for i in range(batch_size):
            out[i] = filter2D(gt_usm[i:i + 1], kernel1[i:i + 1])

        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], self.resize_prob)[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.resize_range[1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.resize_range[0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)

        # add noise
        if np.random.uniform() < self.gaussian_noise_prob:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=self.noise_range, clip=True, rounds=False, gray_prob=self.gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.poisson_scale_range,
                gray_prob=self.gray_noise_prob,
                clip=True,
                rounds=False)

        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range)
        out = torch.clamp(out, 0, 1)  # Use new tensor
        out = self.jpeger(out, quality=jpeg_p)

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if np.random.uniform() < self.second_blur_prob:
            temp_out = torch.zeros_like(out)
            for i in range(batch_size):
                temp_out[i] = filter2D(out[i:i + 1], kernel2[i:i + 1])
            out = temp_out

        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], self.resize_prob2)[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.resize_range2[1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.resize_range2[0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
            out, size=(int(ori_h / self.scale * scale), int(ori_w / self.scale * scale)), mode=mode)

        # add noise
        if np.random.uniform() < self.gaussian_noise_prob2:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=self.noise_range2, clip=True, rounds=False, gray_prob=self.gray_noise_prob2)
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.poisson_scale_range2,
                gray_prob=self.gray_noise_prob2,
                clip=True,
                rounds=False)

        # JPEG compression + the final sinc filter
        if np.random.uniform() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // self.scale, ori_w // self.scale), mode=mode)
            temp_out = torch.zeros_like(out)
            for i in range(batch_size):
                temp_out[i] = filter2D(out[i:i + 1], sinc_kernel[i:i + 1])
            out = temp_out
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range2)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range2)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // self.scale, ori_w // self.scale), mode=mode)
            temp_out = torch.zeros_like(out)
            for i in range(batch_size):
                temp_out[i] = filter2D(out[i:i + 1], sinc_kernel[i:i + 1])
            out = temp_out

        # clamp and round
        out = torch.clamp(out, 0, 1)
        lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

        return lq


def main():
    # Example usage for batch processing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    degrader_batch = ImageDegradationPipeline_RealESRGAN(scale=2, device=device)

    # Create a random batch for demonstration
    dummy_batch = torch.rand(4, 3, 256, 256).to(device)  # batch_size=4, channels=3, height=256, width=256
    lr_batch = degrader_batch.process_batch(dummy_batch)
    print(f"Processed batch shape: {lr_batch.shape}")


if __name__ == '__main__':
    main()