'''
This is the original degradation method
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
            intermittent_edges_prob=0.5,
            intermittent_edges_canny_thresholds=[50, 150],
            intermittent_edges_length_range=[1, 3],
            intermittent_edges_color_shift_range=[-10, 10],
            rainbow_effects_prob=0.5,
            rainbow_effects_edge_threshold=[10, 40],
            rainbow_effects_channel_shift=[20, 50],
            rainbow_effects_edge_width=10,
            rainbow_effects_pixel_randomness_rate=0.1,
            compression_artifacts_prob=1,
            blur_prob=1,
            kernel_list=['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
            kernel_prob=[0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
            blur_sigma=[0.2, 3],
            betag_range=[0.5, 4],
            betap_range=[1, 2],
            sinc_prob=0.1,
            noise_prob=1,
            gaussian_noise_prob=0.5,
            noise_range=[1, 30],
            poisson_scale_range=[0.05, 3],
            gray_noise_prob=0.4,
            sinc_layer_prob=1,
            jpeg_prob=1,
            jpeg_range=[30, 95],
    ):

        # Device (use "cuda" for fast inference)
        self.device = device

        # Scale factor
        self.scale = scale

        # Initialize components
        self.jpeger = DiffJPEG(differentiable=True).to(device)
        self.usm_sharpener = USMSharp().to(device)

        # Default parameters from RealESRGAN
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]

        # Layer intermittent edges
        self.intermittent_edges_prob = intermittent_edges_prob
        self.intermittent_edges_canny_thresholds = intermittent_edges_canny_thresholds
        self.intermittent_edges_length_range = intermittent_edges_length_range
        self.intermittent_edges_color_shift_range = intermittent_edges_color_shift_range

        # Layer rainbow effects
        self.rainbow_effects_prob = rainbow_effects_prob
        self.rainbow_effects_edge_width = rainbow_effects_edge_width
        self.rainbow_effects_edge_threshold = rainbow_effects_edge_threshold
        self.rainbow_effects_channel_shift = rainbow_effects_channel_shift
        self.rainbow_effects_pixel_randomness_rate = rainbow_effects_pixel_randomness_rate

        # Layer compression artifacts
        self.compression_artifacts_prob = compression_artifacts_prob

        # Layer blur
        self.blur_prob = blur_prob
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.betag_range = betag_range
        self.betap_range = betap_range
        self.sinc_prob = sinc_prob

        # Layer noise
        self.noise_prob = noise_prob
        self.gaussian_noise_prob = gaussian_noise_prob
        self.noise_range = noise_range
        self.poisson_scale_range = poisson_scale_range
        self.gray_noise_prob = gray_noise_prob

        # Layer sinc
        self.sinc_layer_prob = sinc_layer_prob

        # Layer jpeg
        self.jpeg_prob = jpeg_prob
        self.jpeg_range = jpeg_range


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


    def generate_sinc_kernel(self, batch_size):
        kernel_size = random.choice(self.kernel_range)
        omega_c = np.random.uniform(np.pi / 3, np.pi)
        sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
        sinc_kernel = torch.FloatTensor(sinc_kernel).to(self.device)
        return sinc_kernel.unsqueeze(0).repeat(batch_size, 1, 1)


    def add_intermittent_edges(self, x, canny_thresholds=[50, 150], length_range=[1, 3],
                                          color_shift_range=[-10, 10]):
        # Convert tensor to numpy for processing
        batch_size = x.size(0)
        device = x.device
        x_np = (x.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype(np.uint8)

        # Process all images in batch simultaneously
        # Create output array
        output = x_np.copy()

        for i in range(batch_size):
            image = x_np[i]
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, *canny_thresholds)

            # Get edge coordinates more efficiently
            edge_coords = np.nonzero(edges)
            edge_y, edge_x = edge_coords

            # Subsample if needed
            if len(edge_y) > 1000:
                idx = np.random.choice(len(edge_y), 1000, replace=False)
                edge_y = edge_y[idx]
                edge_x = edge_x[idx]

            # Vectorized calculations for all points at once
            lengths = np.random.randint(length_range[0], length_range[1] + 1, size=len(edge_y))
            angles = np.random.uniform(0, 2 * np.pi, size=len(edge_y))

            # Calculate all end points at once
            end_x = np.clip(edge_x + (lengths * np.cos(angles)).astype(int), 0, image.shape[1] - 1)
            end_y = np.clip(edge_y + (lengths * np.sin(angles)).astype(int), 0, image.shape[0] - 1)

            # Generate color shifts for all points at once
            color_shifts = np.random.randint(color_shift_range[0], color_shift_range[1] + 1,
                                             size=(len(edge_y), 3))

            # Get original colors for all points
            colors = image[edge_y, edge_x] + color_shifts
            colors = np.clip(colors, 0, 255).astype(np.uint8)

            # Draw lines using optimized cv2.line
            for j in range(len(edge_y)):
                cv2.line(output[i], (edge_x[j], edge_y[j]), (end_x[j], end_y[j]),
                         colors[j].tolist(), 1)

        # Convert back to tensor
        x = torch.from_numpy(output.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)
        return x

    def add_rainbow_effects(self, x, edge_width=10, edge_threshold=[50, 100],
                            channel_shift=[50, 150], pixel_randomness_rate=0.1):
        # Convert tensor to numpy for processing
        batch_size = x.size(0)
        device = x.device
        x_np = (x.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype(np.uint8)

        # Process all images in batch
        output = np.zeros_like(x_np)

        for i in range(batch_size):
            image = x_np[i]
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (edge_width * 2 + 1, edge_width * 2 + 1), 0)

            # Sobel edge detection
            sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0)
            sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1)
            edges = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

            # Normalize edges
            edges_normalized = edges / np.maximum(edges.max(), 1e-10)

            # Create edge mask
            edge_threshold_val = np.random.randint(*edge_threshold)
            edge_mask = edges_normalized > (edge_threshold_val / 255.0)
            edge_intensity = edges_normalized * edge_mask

            # Split channels
            b, g, r = cv2.split(image)

            # Create random mask based on edge intensity
            base_probability = np.clip(pixel_randomness_rate * edges_normalized, 0, 1)
            random_mask = np.random.random(b.shape) < base_probability

            # Initialize shift arrays
            b_shift = np.zeros_like(b, dtype=np.float32)
            g_shift = np.zeros_like(g, dtype=np.float32)
            r_shift = np.zeros_like(r, dtype=np.float32)

            # Apply random shifts only to masked pixels
            b_shift[random_mask] = np.random.randint(-channel_shift[1], channel_shift[1] + 1,
                                                     size=b_shift[random_mask].shape)
            g_shift[random_mask] = np.random.randint(-channel_shift[1], channel_shift[1] + 1,
                                                     size=g_shift[random_mask].shape)
            r_shift[random_mask] = np.random.randint(-channel_shift[1], channel_shift[1] + 1,
                                                     size=r_shift[random_mask].shape)

            # Multiply shifts by edge intensity
            b_shift *= edge_intensity
            g_shift *= edge_intensity
            r_shift *= edge_intensity

            # Apply shifts and clip values
            b_mod = np.clip(b + b_shift, 0, 255).astype(np.uint8)
            g_mod = np.clip(g + g_shift, 0, 255).astype(np.uint8)
            r_mod = np.clip(r + r_shift, 0, 255).astype(np.uint8)

            # Merge channels
            output[i] = cv2.merge([b_mod, g_mod, r_mod])

        # Convert back to tensor
        x = torch.from_numpy(output.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)
        return x


    def compress_with_detail_preservation(self, image_tensor, block_size=8, quality=30):
        """
        Compress image tensor using DCT transformation with detail preservation.

        Args:
            image_tensor (torch.Tensor): Input tensor in (B, C, H, W) format
            block_size (int): Size of compression blocks (default: 8)
            quality (int): Compression quality from 0 (worst) to 100 (best)

        Returns:
            torch.Tensor: Compressed image tensor in the same format
        """
        if not isinstance(image_tensor, torch.Tensor):
            raise TypeError("Expected PyTorch tensor in (B, C, H, W) format")

        # Validate and normalize quality parameter
        quality = max(0, min(100, int(quality)))  # Ensure quality is between 0 and 100

        # Convert tensor to numpy array
        image = image_tensor.permute(0, 2, 3, 1).detach().cpu().numpy()
        image = (image * 255).astype(np.uint8)

        batch_size, h, w, c = image.shape
        output_images = np.zeros_like(image)

        def create_quantization_matrices(quality):
            # Standard JPEG luminance quantization matrix
            standard_luminance_matrix = np.array([
                [16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99]
            ])

            # Standard JPEG chrominance quantization matrix
            standard_chrominance_matrix = np.array([
                [17, 18, 24, 47, 99, 99, 99, 99],
                [18, 21, 26, 66, 99, 99, 99, 99],
                [24, 26, 56, 99, 99, 99, 99, 99],
                [47, 66, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99]
            ])

            # Handle quality = 0 case
            if quality == 0:
                return np.ones_like(standard_luminance_matrix) * 255, np.ones_like(standard_chrominance_matrix) * 255

            # Standard JPEG quality scaling
            if quality < 50:
                scale = 5000 / quality
            else:
                scale = 200 - 2 * quality

            # Scale matrices
            scaled_luminance = np.floor((standard_luminance_matrix * scale + 50) / 100)
            scaled_chrominance = np.floor((standard_chrominance_matrix * scale + 50) / 100)

            # Ensure valid quantization values (1-255)
            scaled_luminance = np.clip(scaled_luminance, 1, 255).astype(np.uint8)
            scaled_chrominance = np.clip(scaled_chrominance, 1, 255).astype(np.uint8)

            return scaled_luminance, scaled_chrominance

        def process_channel(channel, quant_matrix, h_p, w_p, block_size):
            # Prepare for vectorized processing
            blocks_v = h_p // block_size
            blocks_h = w_p // block_size

            # Split into blocks using stride tricks
            strided_channel = np.lib.stride_tricks.as_strided(
                channel,
                shape=(blocks_v, blocks_h, block_size, block_size),
                strides=(channel.strides[0] * block_size,
                         channel.strides[1] * block_size,
                         channel.strides[0],
                         channel.strides[1])
            ).astype(np.float32)

            # Subtract DC offset
            strided_channel -= 128

            # Apply DCT to all blocks
            dct_blocks = np.zeros_like(strided_channel)
            for i in range(blocks_v):
                for j in range(blocks_h):
                    dct_blocks[i, j] = cv2.dct(strided_channel[i, j])

            # Quantization with quality-adjusted matrices
            quantized_blocks = np.round(dct_blocks / quant_matrix) * quant_matrix

            # Inverse DCT
            reconstructed_blocks = np.zeros_like(quantized_blocks)
            for i in range(blocks_v):
                for j in range(blocks_h):
                    reconstructed_blocks[i, j] = cv2.idct(quantized_blocks[i, j])

            # Reassemble image
            reconstructed = np.zeros((h_p, w_p))
            for i in range(blocks_v):
                for j in range(blocks_h):
                    reconstructed[i * block_size:(i + 1) * block_size,
                    j * block_size:(j + 1) * block_size] = reconstructed_blocks[i, j]

            return np.clip(reconstructed + 128, 0, 255)

        # Get quantization matrices
        quant_matrix_y, quant_matrix_c = create_quantization_matrices(quality)

        for i in range(batch_size):
            img = image[i]
            # Calculate padding if needed
            pad_h = (block_size - h % block_size) % block_size
            pad_w = (block_size - w % block_size) % block_size

            if pad_h > 0 or pad_w > 0:
                img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')

            h_p, w_p = img.shape[:2]
            img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(img_ycrcb)

            # Process each channel with appropriate quantization matrix
            y_compressed = process_channel(y, quant_matrix_y, h_p, w_p, block_size)
            cr_compressed = process_channel(cr, quant_matrix_c, h_p, w_p, block_size)
            cb_compressed = process_channel(cb, quant_matrix_c, h_p, w_p, block_size)

            # Merge channels
            compressed_image_ycrcb = cv2.merge([y_compressed, cr_compressed, cb_compressed])
            compressed_image = cv2.cvtColor(compressed_image_ycrcb.astype(np.uint8),
                                            cv2.COLOR_YCrCb2BGR)

            # Remove padding if it was added
            if pad_h > 0 or pad_w > 0:
                compressed_image = compressed_image[:h, :w]

            output_images[i] = compressed_image

        # Convert back to tensor format
        output_images = output_images.astype(np.float32) / 255.0
        return torch.from_numpy(output_images).permute(0, 3, 1, 2).to(image_tensor.device)

    def process_batch(self, gt_batch):
        batch_size = gt_batch.size(0)
        gt = gt_batch

        # Generate kernels with proper batch size
        kernel1 = self.generate_kernel1(batch_size)
        sinc_kernel = self.generate_sinc_kernel(batch_size)

        # interpolation
        ori_h, ori_w = gt.size()[2:4]
        out = gt
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, size=(ori_h // self.scale, ori_w // self.scale), mode=mode)

        # Define operations as functions
        def compression_artifacts(image):
            if np.random.uniform() < self.compression_artifacts_prob:
                return self.compress_with_detail_preservation(image)
            else:
                return image

        def intermittent_edges(image):
            if np.random.uniform() < self.intermittent_edges_prob:
                return self.add_intermittent_edges(
                    image,
                    self.intermittent_edges_canny_thresholds,
                    self.intermittent_edges_length_range,
                    self.intermittent_edges_color_shift_range,
                )
            else:
                return image

        def rainbow_effects(image):
            if np.random.uniform() < self.rainbow_effects_prob:
                return self.add_rainbow_effects(
                    image,
                    self.rainbow_effects_edge_width,
                    self.rainbow_effects_edge_threshold,
                    self.rainbow_effects_channel_shift,
                    self.rainbow_effects_pixel_randomness_rate
                )
            else:
                return image

        def blur_operation(image):
            if np.random.uniform() < self.blur_prob:
                temp_out = torch.zeros_like(image)
                for i in range(batch_size):
                    temp_out[i] = filter2D(image[i:i + 1], kernel1[i:i + 1])
                return temp_out
            else:
                return image

        def noise_operation(image):
            if np.random.uniform() < self.noise_prob:
                if np.random.uniform() < self.gaussian_noise_prob:
                    return random_add_gaussian_noise_pt(
                        image, sigma_range=self.noise_range, clip=True,
                        rounds=False, gray_prob=self.gray_noise_prob)
                else:
                    return random_add_poisson_noise_pt(
                        image, scale_range=self.poisson_scale_range,
                        gray_prob=self.gray_noise_prob, clip=True, rounds=False)
            else:
                return image

        def sinc_operation(image):
            if np.random.uniform() < self.sinc_layer_prob:
                temp_out = torch.zeros_like(image)
                for i in range(batch_size):
                    temp_out[i] = filter2D(image[i:i + 1], sinc_kernel[i:i + 1])
                return temp_out
            else:
                return image

        def jpeg_operation(image):
            if np.random.uniform() < self.jpeg_prob:
                jpeg_p = image.new_zeros(image.size(0)).uniform_(*self.jpeg_range)
                image = torch.clamp(image, 0, 1)
                return self.jpeger(image, quality=jpeg_p)
            else:
                return image

        # Create list of operations with random weights
        operations = [
            (np.random.random(), compression_artifacts),
            (np.random.random(), intermittent_edges),
            (np.random.random(), rainbow_effects),
            (np.random.random(), blur_operation),
            (np.random.random(), noise_operation),
            (np.random.random(), sinc_operation),
            (np.random.random(), jpeg_operation)
        ]

        # Sort operations by their random weights in descending order
        operations.sort(reverse=True, key=lambda x: x[0])

        # Apply operations in the sorted order
        for _, operation in operations:
            out = operation(out)

        # Final clamp and round
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