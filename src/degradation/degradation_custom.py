'''
This is the original degradation method based on:
"Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data"
https://github.com/xinntao/Real-ESRGAN

@InProceedings{wang2021realesrgan,
    author    = {Xintao Wang and Liangbin Xie and Chao Dong and Ying Shan},
    title     = {Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data},
    booktitle = {International Conference on Computer Vision Workshops (ICCVW)},
    date      = {2021}
}
'''


import torch
import random
import numpy as np
import torch.nn.functional as F
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
import math
import cv2


class ImageDegradationPipeline_custom:
    def __init__(
            self,
            scale=2,
            device='cuda',
            kernel_list=['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
            kernel_prob=[0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
            blur_sigma=[0.2, 3],
            betag_range=[0.5, 4],
            betap_range=[1, 2],
            sinc_prob=0.1,
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
            final_sinc_prob=1,
            second_blur_prob=0.8,
            intermittent_edges_prob=0.5,
            rainbow_effects_prob=0.5,
            intermittent_edges_prob2=0.5,
            rainbow_effects_prob2=0.5,
            rainbow_effects_edge_threshold =[10, 40],
            rainbow_effects_channel_shift =[20, 50],
            rainbow_effects_edge_threshold2=[10, 40],
            rainbow_effects_channel_shift2=[20, 50],
            intermittent_edges_canny_thresholds=[50, 150],
            intermittent_edges_length_range=[1, 3],
            intermittent_edges_color_shift_range=[-10, 10],
            intermittent_edges_canny_thresholds2=[50, 150],
            intermittent_edges_length_range2=[1, 3],
            intermittent_edges_color_shift_range2=[-10, 10],
            rainbow_effects_edge_width=10,
            rainbow_effects_pixel_randomness_rate=0.1,
            rainbow_effects_edge_width2=10,
            rainbow_effects_pixel_randomness_rate2=0.1
    ):
        self.scale = scale
        self.device = device

        # Initialize components
        self.jpeger = DiffJPEG(differentiable=True).to(device)
        self.usm_sharpener = USMSharp().to(device)

        # Default parameters from RealESRGAN
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]
        self.pulse_tensor = torch.zeros(21, 21).float().to(device)
        self.pulse_tensor[10, 10] = 1

        # First degradation parameters
        self.intermittent_edges_prob = intermittent_edges_prob
        self.rainbow_effects_prob = rainbow_effects_prob
        self.rainbow_effects_edge_width = rainbow_effects_edge_width
        self.rainbow_effects_edge_threshold = rainbow_effects_edge_threshold
        self.rainbow_effects_channel_shift = rainbow_effects_channel_shift
        self.rainbow_effects_pixel_randomness_rate = rainbow_effects_pixel_randomness_rate
        self.intermittent_edges_canny_thresholds = intermittent_edges_canny_thresholds
        self.intermittent_edges_length_range = intermittent_edges_length_range
        self.intermittent_edges_color_shift_range = intermittent_edges_color_shift_range
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.betag_range = betag_range
        self.betap_range = betap_range
        self.sinc_prob = sinc_prob

        # Second degradation parameters
        self.intermittent_edges_prob2 = intermittent_edges_prob2
        self.rainbow_effects_prob2 = rainbow_effects_prob2
        self.rainbow_effects_edge_width2 = rainbow_effects_edge_width2
        self.rainbow_effects_edge_threshold2 = rainbow_effects_edge_threshold2
        self.rainbow_effects_channel_shift2 = rainbow_effects_channel_shift2
        self.rainbow_effects_pixel_randomness_rate2 = rainbow_effects_pixel_randomness_rate2
        self.intermittent_edges_canny_thresholds2 = intermittent_edges_canny_thresholds2
        self.intermittent_edges_length_range2 = intermittent_edges_length_range2
        self.intermittent_edges_color_shift_range2 = intermittent_edges_color_shift_range2
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
            sinc_kernel = self.pulse_tensor.clone()
        return sinc_kernel.unsqueeze(0).repeat(batch_size, 1, 1)

    def add_intermittent_edges(self, image, canny_thresholds=[50, 150], length_range=[1, 3],
                               color_shift_range=[-10, 10]):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect edges using Canny
        edges = cv2.Canny(gray, *canny_thresholds)

        # Get edge pixel coordinates
        edge_coords = np.column_stack(np.where(edges > 0))

        # Randomly select subset of edge points to reduce processing
        if len(edge_coords) > 1000:
            edge_coords = edge_coords[np.random.choice(len(edge_coords), 1000, replace=False)]

        # Prepare output image
        output = image.copy()

        # Vectorized random operations
        lengths = np.random.randint(length_range[0], length_range[1] + 1, size=len(edge_coords))
        angles = np.random.uniform(0, 2 * np.pi, size=len(edge_coords))

        # Compute end points
        dx = (lengths * np.cos(angles)).astype(int)
        dy = (lengths * np.sin(angles)).astype(int)

        end_coords_x = np.clip(edge_coords[:, 1] + dx, 0, image.shape[1] - 1)
        end_coords_y = np.clip(edge_coords[:, 0] + dy, 0, image.shape[0] - 1)

        # Vectorized color shifts
        color_shifts = np.random.randint(color_shift_range[0], color_shift_range[1] + 1, size=(len(edge_coords), 3))

        # Process each point
        for (y, x), end_y, end_x, shift in zip(edge_coords, end_coords_y, end_coords_x, color_shifts):
            color = np.clip(image[y, x] + shift, 0, 255).astype(np.uint8)
            cv2.line(output, (x, y), (end_x, end_y), color.tolist(), 1)

        return output

    def add_rainbow_effects(self, image,
                            edge_width=10,
                            edge_threshold=[50, 100],
                            channel_shift=[50, 150],
                            pixel_randomness_rate=0.1):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to control edge width
        blurred = cv2.GaussianBlur(gray, (edge_width * 2 + 1, edge_width * 2 + 1), 0)

        # Sobel edge detection with adjustable width
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1)
        edges = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        # Normalize edges
        edges_normalized = edges / edges.max()

        # Random thresholds
        edge_threshold_val = np.random.randint(*edge_threshold)

        # Create soft edge mask with gradient
        edge_mask = edges_normalized > (edge_threshold_val / 255.0)
        edge_intensity = edges_normalized * edge_mask

        # Split channels
        b, g, r = cv2.split(image)

        # Создаем маску для случайных смещений
        random_mask = np.random.random(b.shape) < pixel_randomness_rate

        # Create unique random shifts для части пикселей
        b_shift = np.zeros_like(b, dtype=np.float32)
        g_shift = np.zeros_like(g, dtype=np.float32)
        r_shift = np.zeros_like(r, dtype=np.float32)

        b_shift[random_mask] = np.random.randint(-channel_shift[1], channel_shift[1] + 1,
                                                 size=b_shift[random_mask].shape)
        g_shift[random_mask] = np.random.randint(-channel_shift[1], channel_shift[1] + 1,
                                                 size=g_shift[random_mask].shape)
        r_shift[random_mask] = np.random.randint(-channel_shift[1], channel_shift[1] + 1,
                                                 size=r_shift[random_mask].shape)

        # Умножаем смещения на интенсивность края
        b_shift *= edge_intensity
        g_shift *= edge_intensity
        r_shift *= edge_intensity

        # Apply shifts with gradient
        b_mod = np.clip(b + b_shift, 0, 255).astype(np.uint8)
        g_mod = np.clip(g + g_shift, 0, 255).astype(np.uint8)
        r_mod = np.clip(r + r_shift, 0, 255).astype(np.uint8)

        # Merge channels
        result = cv2.merge([b_mod, g_mod, r_mod])
        return result

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
        # intermittent_edges and rainbow_effects
        out = gt_usm
        out_np = out.permute(0, 2, 3, 1).detach().cpu().numpy()  # (B, H, W, C)
        out_np = (out_np * 255).astype(np.uint8)
        for i in range(batch_size):
            img = out_np[i]
            # intermittent_edges
            if np.random.uniform() < self.intermittent_edges_prob:
                img = self.add_intermittent_edges(img, canny_thresholds=self.intermittent_edges_canny_thresholds, length_range=self.intermittent_edges_length_range, color_shift_range=self.intermittent_edges_color_shift_range)
            # rainbow_effects
            if np.random.uniform() < self.rainbow_effects_prob:
                img = self.add_rainbow_effects(img, edge_width=self.rainbow_effects_edge_width , edge_threshold=self.rainbow_effects_edge_threshold, channel_shift=self.rainbow_effects_channel_shift, pixel_randomness_rate= self.rainbow_effects_pixel_randomness_rate)
            out_np[i] = img
        out_np = out_np.astype(np.float32) / 255.0
        out = torch.from_numpy(out_np).permute(0, 3, 1, 2).to(self.device)

        # blur
        temp_out = torch.zeros_like(out)
        for i in range(batch_size):
            temp_out[i] = filter2D(out[i:i + 1], kernel2[i:i + 1])
        out = temp_out

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
        out = torch.clamp(out, 0, 1)
        out = self.jpeger(out, quality=jpeg_p)

        # ----------------------- The second degradation process ----------------------- #
        # intermittent_edges and rainbow_effects
        out_np = out.permute(0, 2, 3, 1).detach().cpu().numpy()  # (B, H, W, C)
        out_np = (out_np * 255).astype(np.uint8)
        for i in range(batch_size):
            img = out_np[i]
            # intermittent_edges
            if np.random.uniform() < self.intermittent_edges_prob2:
                img = self.add_intermittent_edges(img, canny_thresholds=self.intermittent_edges_canny_thresholds2, length_range=self.intermittent_edges_length_range2, color_shift_range=self.intermittent_edges_color_shift_range2)
            # rainbow_effects
            if np.random.uniform() < self.rainbow_effects_prob2:
                img = self.add_rainbow_effects(img, edge_width=self.rainbow_effects_edge_width2 , edge_threshold=self.rainbow_effects_edge_threshold2, channel_shift=self.rainbow_effects_channel_shift2, pixel_randomness_rate= self.rainbow_effects_pixel_randomness_rate2)
            out_np[i] = img
        out_np = out_np.astype(np.float32) / 255.0
        out = torch.from_numpy(out_np).permute(0, 3, 1, 2).to(self.device)

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

        # # ----------------------- Final steps ----------------------- #
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
    degrader_batch = ImageDegradationPipeline_custom(scale=2, device=device)

    # Create a random batch for demonstration
    dummy_batch = torch.rand(4, 3, 256, 256).to(device)  # batch_size=4, channels=3, height=256, width=256
    lr_batch = degrader_batch.process_batch(dummy_batch)
    print(f"Processed batch shape: {lr_batch.shape}")


if __name__ == '__main__':
    main()