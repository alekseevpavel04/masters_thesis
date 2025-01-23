'''
This is the degradation method used in paper:
"APISR: Anime Production Inspired Real-World Anime Super-Resolution"
https://github.com/Kiteretsu77/APISR

@inproceedings{wang2024apisr,
  title={APISR: Anime Production Inspired Real-World Anime Super-Resolution},
  author={Wang, Boyang and Yang, Fengyu and Yu, Xihang and Zhang, Chao and Zhao, Hanbin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={25574--25584},
  year={2024}
}
'''


import argparse
import cv2
import torch
import numpy as np
import os, shutil, time
import sys, random
from multiprocessing import Process, Queue, Pool
from os import path as osp
from tqdm import tqdm
import copy
import warnings
import gc
import math
from math import log10, sqrt
import torch.nn.functional as F
from scipy import special
from scipy.stats import multivariate_normal
from torchvision.transforms.functional import rgb_to_grayscale

warnings.filterwarnings("ignore")

# Utility functions
def np2tensor(np_frame):
    return torch.from_numpy(np.transpose(np_frame, (2, 0, 1))).unsqueeze(0).cuda().float() / 255

def tensor2np(tensor):
    return (np.transpose(tensor.detach().squeeze(0).cpu().numpy(), (1, 2, 0))) * 255

def mass_tensor2np(tensor):
    return (np.transpose(tensor.detach().squeeze(0).cpu().numpy(), (0, 2, 3, 1))) * 255

def filter2D(img, kernel):
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

def generate_kernels(opt):
    kernel_range = [2 * v + 1 for v in range(opt["kernel_range"][0], opt["kernel_range"][1])]
    kernel_size = random.choice(kernel_range)
    if np.random.uniform() < opt['sinc_prob']:
        if kernel_size < 13:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c = np.random.uniform(np.pi / 5, np.pi)
        kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
    else:
        kernel = random_mixed_kernels(
            opt['kernel_list'],
            opt['kernel_prob'],
            kernel_size,
            opt['blur_sigma'],
            opt['blur_sigma'], [-math.pi, math.pi],
            opt['betag_range'],
            opt['betap_range'],
            noise_range=None)
    pad_size = (21 - kernel_size) // 2
    kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

    kernel_size = random.choice(kernel_range)
    if np.random.uniform() < opt['sinc_prob2']:
        if kernel_size < 13:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c = np.random.uniform(np.pi / 5, np.pi)
        kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
    else:
        kernel2 = random_mixed_kernels(
            opt['kernel_list2'],
            opt['kernel_prob2'],
            kernel_size,
            opt['blur_sigma2'],
            opt['blur_sigma2'], [-math.pi, math.pi],
            opt['betag_range2'],
            opt['betap_range2'],
            noise_range=None)
    pad_size = (21 - kernel_size) // 2
    kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))
    kernel = torch.FloatTensor(kernel)
    kernel2 = torch.FloatTensor(kernel2)
    return (kernel, kernel2)

def circular_lowpass_kernel(cutoff, kernel_size, pad_to=0):
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    kernel = np.fromfunction(
        lambda x, y: cutoff * special.j1(cutoff * np.sqrt(
            (x - (kernel_size - 1) / 2)**2 + (y - (kernel_size - 1) / 2)**2)) / (2 * np.pi * np.sqrt(
                (x - (kernel_size - 1) / 2)**2 + (y - (kernel_size - 1) / 2)**2)), [kernel_size, kernel_size])
    kernel[(kernel_size - 1) // 2, (kernel_size - 1) // 2] = cutoff**2 / (4 * np.pi)
    kernel = kernel / np.sum(kernel)
    if pad_to > kernel_size:
        pad_size = (pad_to - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
    return kernel

def random_mixed_kernels(kernel_list, kernel_prob, kernel_size=21, sigma_x_range=(0.6, 5), sigma_y_range=(0.6, 5), rotation_range=(-math.pi, math.pi), betag_range=(0.5, 8), betap_range=(0.5, 8), noise_range=None):
    kernel_type = random.choices(kernel_list, kernel_prob)[0]
    if kernel_type == 'iso':
        kernel = random_bivariate_Gaussian(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range, noise_range=noise_range, isotropic=True)
    elif kernel_type == 'aniso':
        kernel = random_bivariate_Gaussian(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range, noise_range=noise_range, isotropic=False)
    elif kernel_type == 'generalized_iso':
        kernel = random_bivariate_generalized_Gaussian(
            kernel_size,
            sigma_x_range,
            sigma_y_range,
            rotation_range,
            betag_range,
            noise_range=noise_range,
            isotropic=True)
    elif kernel_type == 'generalized_aniso':
        kernel = random_bivariate_generalized_Gaussian(
            kernel_size,
            sigma_x_range,
            sigma_y_range,
            rotation_range,
            betag_range,
            noise_range=noise_range,
            isotropic=False)
    elif kernel_type == 'plateau_iso':
        kernel = random_bivariate_plateau(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range, betap_range, noise_range=None, isotropic=True)
    elif kernel_type == 'plateau_aniso':
        kernel = random_bivariate_plateau(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range, betap_range, noise_range=None, isotropic=False)
    return kernel

def random_bivariate_Gaussian(kernel_size, sigma_x_range, sigma_y_range, rotation_range, noise_range=None, isotropic=True):
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
    if isotropic is False:
        sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
        rotation = np.random.uniform(rotation_range[0], rotation_range[1])
    else:
        sigma_y = sigma_x
        rotation = 0
    kernel = bivariate_Gaussian(kernel_size, sigma_x, sigma_y, rotation, isotropic=isotropic)
    if noise_range is not None:
        noise = np.random.uniform(noise_range[0], noise_range[1], size=kernel.shape)
        kernel = kernel * noise
    kernel = kernel / np.sum(kernel)
    return kernel

def bivariate_Gaussian(kernel_size, sig_x, sig_y, theta, grid=None, isotropic=True):
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]])
    else:
        sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)
    kernel = pdf2(sigma_matrix, grid)
    kernel = kernel / np.sum(kernel)
    return kernel

def mesh_grid(kernel_size):
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    xy = np.hstack((xx.reshape((kernel_size * kernel_size, 1)), yy.reshape(kernel_size * kernel_size, 1))).reshape(kernel_size, kernel_size, 2)
    return xy, xx, yy

def sigma_matrix2(sig_x, sig_y, theta):
    d_matrix = np.array([[sig_x**2, 0], [0, sig_y**2]])
    u_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.dot(u_matrix, np.dot(d_matrix, u_matrix.T))

def pdf2(sigma_matrix, grid):
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.sum(np.dot(grid, inverse_sigma) * grid, 2))
    return kernel

def random_bivariate_generalized_Gaussian(kernel_size, sigma_x_range, sigma_y_range, rotation_range, beta_range, noise_range=None, isotropic=True):
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
    if isotropic is False:
        sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
        rotation = np.random.uniform(rotation_range[0], rotation_range[1])
    else:
        sigma_y = sigma_x
        rotation = 0
    if np.random.uniform() < 0.5:
        beta = np.random.uniform(beta_range[0], 1)
    else:
        beta = np.random.uniform(1, beta_range[1])
    kernel = bivariate_generalized_Gaussian(kernel_size, sigma_x, sigma_y, rotation, beta, isotropic=isotropic)
    if noise_range is not None:
        noise = np.random.uniform(noise_range[0], noise_range[1], size=kernel.shape)
        kernel = kernel * noise
    kernel = kernel / np.sum(kernel)
    return kernel

def bivariate_generalized_Gaussian(kernel_size, sig_x, sig_y, theta, beta, grid=None, isotropic=True):
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]])
    else:
        sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.power(np.sum(np.dot(grid, inverse_sigma) * grid, 2), beta))
    kernel = kernel / np.sum(kernel)
    return kernel

def random_bivariate_plateau(kernel_size, sigma_x_range, sigma_y_range, rotation_range, beta_range, noise_range=None, isotropic=True):
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
    if isotropic is False:
        sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
        rotation = np.random.uniform(rotation_range[0], rotation_range[1])
    else:
        sigma_y = sigma_x
        rotation = 0
    if np.random.uniform() < 0.5:
        beta = np.random.uniform(beta_range[0], 1)
    else:
        beta = np.random.uniform(1, beta_range[1])
    kernel = bivariate_plateau(kernel_size, sigma_x, sigma_y, rotation, beta, isotropic=isotropic)
    if noise_range is not None:
        noise = np.random.uniform(noise_range[0], noise_range[1], size=kernel.shape)
        kernel = kernel * noise
    kernel = kernel / np.sum(kernel)
    return kernel

def bivariate_plateau(kernel_size, sig_x, sig_y, theta, beta, grid=None, isotropic=True):
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]])
    else:
        sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.reciprocal(np.power(np.sum(np.dot(grid, inverse_sigma) * grid, 2), beta) + 1)
    kernel = kernel / np.sum(kernel)
    return kernel

def random_generate_gaussian_noise_pt(img, sigma_range=(0, 10), gray_prob=0):
    sigma = torch.rand(img.size(0), dtype=img.dtype, device=img.device) * (sigma_range[1] - sigma_range[0]) + sigma_range[0]
    gray_noise = torch.rand(img.size(0), dtype=img.dtype, device=img.device)
    gray_noise = (gray_noise < gray_prob).float()
    return generate_gaussian_noise_pt(img, sigma, gray_noise)

def generate_gaussian_noise_pt(img, sigma=10, gray_noise=0):
    b, _, h, w = img.size()
    if not isinstance(sigma, (float, int)):
        sigma = sigma.view(img.size(0), 1, 1, 1)
    if isinstance(gray_noise, (float, int)):
        cal_gray_noise = gray_noise > 0
    else:
        gray_noise = gray_noise.view(b, 1, 1, 1)
        cal_gray_noise = torch.sum(gray_noise) > 0
    if cal_gray_noise:
        noise_gray = torch.randn(*img.size()[2:4], dtype=img.dtype, device=img.device) * sigma / 255.
        noise_gray = noise_gray.view(b, 1, h, w)
    noise = torch.randn(*img.size(), dtype=img.dtype, device=img.device) * sigma / 255.
    if cal_gray_noise:
        noise = noise * (1 - gray_noise) + noise_gray * gray_noise
    return noise

def random_add_gaussian_noise_pt(img, sigma_range=(0, 1.0), gray_prob=0, clip=True, rounds=False):
    noise = random_generate_gaussian_noise_pt(img, sigma_range, gray_prob)
    out = img + noise
    if clip and rounds:
        out = torch.clamp((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = torch.clamp(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.
    return out

def random_generate_poisson_noise_pt(img, scale_range=(0, 1.0), gray_prob=0):
    scale = torch.rand(img.size(0), dtype=img.dtype, device=img.device) * (scale_range[1] - scale_range[0]) + scale_range[0]
    gray_noise = torch.rand(img.size(0), dtype=img.dtype, device=img.device)
    gray_noise = (gray_noise < gray_prob).float()
    return generate_poisson_noise_pt(img, scale, gray_noise)

def generate_poisson_noise_pt(img, scale=1.0, gray_noise=0):
    b, _, h, w = img.size()
    if isinstance(gray_noise, (float, int)):
        cal_gray_noise = gray_noise > 0
    else:
        gray_noise = gray_noise.view(b, 1, 1, 1)
        cal_gray_noise = torch.sum(gray_noise) > 0
    if cal_gray_noise:
        img_gray = rgb_to_grayscale(img, num_output_channels=1)
        img_gray = torch.clamp((img_gray * 255.0).round(), 0, 255) / 255.
        vals_list = [len(torch.unique(img_gray[i, :, :, :])) for i in range(b)]
        vals_list = [2**np.ceil(np.log2(vals)) for vals in vals_list]
        vals = img_gray.new_tensor(vals_list).view(b, 1, 1, 1)
        out = torch.poisson(img_gray * vals) / vals
        noise_gray = out - img_gray
        noise_gray = noise_gray.expand(b, 3, h, w)
    img = torch.clamp((img * 255.0).round(), 0, 255) / 255.
    vals_list = [len(torch.unique(img[i, :, :, :])) for i in range(b)]
    vals_list = [2**np.ceil(np.log2(vals)) for vals in vals_list]
    vals = img.new_tensor(vals_list).view(b, 1, 1, 1)
    out = torch.poisson(img * vals) / vals
    noise = out - img
    if cal_gray_noise:
        noise = noise * (1 - gray_noise) + noise_gray * gray_noise
    if not isinstance(scale, (float, int)):
        scale = scale.view(b, 1, 1, 1)
    return noise * scale

def random_add_poisson_noise_pt(img, scale_range=(0, 1.0), gray_prob=0, clip=True, rounds=False):
    noise = random_generate_poisson_noise_pt(img, scale_range, gray_prob)
    out = img + noise
    if clip and rounds:
        out = torch.clamp((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = torch.clamp(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.
    return out

def downsample_1st(out, opt):
    updown_type = random.choices(['up', 'down', 'keep'], opt['resize_prob'])[0]
    if updown_type == 'up':
        scale = np.random.uniform(1, opt['resize_range'][1])
    elif updown_type == 'down':
        scale = np.random.uniform(opt['resize_range'][0], 1)
    else:
        scale = 1
    mode = random.choice(opt['resize_options'])
    out = F.interpolate(out, scale_factor=scale, mode=mode)
    return out

def downsample_2nd(out, opt, ori_h, ori_w):
    if opt['scale'] == 4:
        updown_type = random.choices(['up', 'down', 'keep'], opt['resize_prob2'])[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, opt['resize_range2'][1])
        elif updown_type == 'down':
            scale = np.random.uniform(opt['resize_range2'][0], 1)
        else:
            scale = 1
        mode = random.choice(opt['resize_options'])
        out = F.interpolate(
            out, size=(int(ori_h / opt['scale'] * scale), int(ori_w / opt['scale'] * scale)), mode=mode
        )
    return out

def common_degradation(out, opt, kernels, process_id, verbose=False):
    kernel1, kernel2 = kernels
    downsample_1st_position = random.choices([0, 1, 2])[0]
    if opt['scale'] == 4:
        downsample_2nd_position = random.choices([0, 1, 2])[0]
    else:
        downsample_2nd_position = -1

    ####---------------------------- Frist Degradation ----------------------------------####
    batch_size, _, ori_h, ori_w = out.size()

    if downsample_1st_position == 0:
        out = downsample_1st(out, opt)

    # Bluring kernel
    out = filter2D(out, kernel1)
    if verbose: print(f"(1st) blur noise")

    if downsample_1st_position == 1:
        out = downsample_1st(out, opt)

    # Noise effect (gaussian / poisson)
    gray_noise_prob = opt['gray_noise_prob']
    if np.random.uniform() < opt['gaussian_noise_prob']:
        # Gaussian noise
        out = random_add_gaussian_noise_pt(
            out, sigma_range=opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
        name = "gaussian_noise"
    else:
        # Poisson noise
        out = random_add_poisson_noise_pt(
            out,
            scale_range=opt['poisson_scale_range'],
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False)
        name = "poisson_noise"
    if verbose: print("(1st) " + str(name))

    if downsample_1st_position == 2:
        out = downsample_1st(out, opt)

    # Choose an image compression codec (All degradation batch use the same codec)
    image_codec = random.choices(opt['compression_codec1'], opt['compression_codec_prob1'])[0]
    if image_codec == "jpeg":
        out = JPEG.compress_tensor(out, opt['jpeg_quality_range1'])
    elif image_codec == "webp":
        out = WEBP.compress_tensor(out)
    else:
        raise NotImplementedError("We don't have such image compression designed!")

    # ####---------------------------- Second Degradation ----------------------------------####
    if downsample_2nd_position == 0:
        out = downsample_2nd(out, opt, ori_h, ori_w)

    # Add blur 2nd time
    if np.random.uniform() < opt['second_blur_prob']:
        if verbose: print("(2nd) blur noise")
        out = filter2D(out, kernel2)

    if downsample_2nd_position == 1:
        out = downsample_2nd(out, opt, ori_h, ori_w)

    # Add noise 2nd time
    gray_noise_prob = opt['gray_noise_prob2']
    if np.random.uniform() < opt['gaussian_noise_prob2']:
        # gaussian noise
        if verbose: print("(2nd) gaussian noise")
        out = random_add_gaussian_noise_pt(
            out, sigma_range=opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
        name = "gaussian_noise"
    else:
        # poisson noise
        if verbose: print("(2nd) poisson noise")
        out = random_add_poisson_noise_pt(
            out, scale_range=opt['poisson_scale_range2'], gray_prob=gray_noise_prob, clip=True, rounds=False)
        name = "poisson_noise"

    if downsample_2nd_position == 2:
        out = downsample_2nd(out, opt, ori_h, ori_w)

    # Choose an image compression codec (All degradation batch use the same codec)
    image_codec = random.choices(opt['compression_codec2'], opt['compression_codec_prob2'])[0]
    if image_codec == "jpeg":
        out = JPEG.compress_tensor(out, opt['jpeg_quality_range2'])
    elif image_codec == "webp":
        out = WEBP.compress_tensor(out)
    else:
        raise NotImplementedError("We don't have such image compression designed!")

    return out


class JPEG:
    @staticmethod
    def compress_tensor(tensor_frames, quality_range):
        single_frame = tensor2np(tensor_frames)
        # Convert to CV_8U explicitly
        single_frame = (single_frame).astype(np.uint8)
        jpeg_quality = random.randint(*quality_range)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        _, encimg = cv2.imencode('.jpg', single_frame, encode_param)
        decimg = cv2.imdecode(encimg, 1)
        result = np2tensor(decimg)
        return result

class WEBP:
    @staticmethod
    def compress_tensor(tensor_frames):
        single_frame = tensor2np(tensor_frames)
        # Convert to CV_8U explicitly
        single_frame = (single_frame).astype(np.uint8)
        webp_quality = random.randint(30, 95)
        encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), webp_quality]
        _, encimg = cv2.imencode('.webp', single_frame, encode_param)
        decimg = cv2.imdecode(encimg, 1)
        result = np2tensor(decimg)
        return result

class degradation_v1:
    def __init__(self):
        self.kernel1, self.kernel2, self.sinc_kernel = None, None, None
        self.queue_size = 160

        # Init the compression instance
        self.jpeg_instance = JPEG()
        self.webp_instance = WEBP()

    def reset_kernels(self, opt):
        kernel1, kernel2 = generate_kernels(opt)
        self.kernel1 = kernel1.unsqueeze(0).cuda()
        self.kernel2 = kernel2.unsqueeze(0).cuda()

    @torch.no_grad()
    def degradate_process(self, out, opt, store_path, process_id, verbose=False):
        """
        Degrade a single image tensor.

        Args:
            out (torch.Tensor): Input image tensor with shape (1, C, H, W).
            opt (dict): Configuration options.
            store_path (str): Temporary path (not used in this version).
            process_id (int): Process ID (not used in this version).
            verbose (bool): Whether to print debug information.

        Returns:
            torch.Tensor: Degraded image tensor with shape (1, C, H // scale, W // scale).
        """
        batch_size, _, ori_h, ori_w = out.size()

        # Shared degradation until the last step
        resize_mode = random.choice(opt['resize_options'])
        out = common_degradation(out, opt, [self.kernel1, self.kernel2], process_id, verbose=verbose)

        # Resize back
        out = F.interpolate(out, size=(ori_h // opt['scale'], ori_w // opt['scale']), mode=resize_mode)
        out = torch.clamp(out, 0, 1)

        return out

class ImageDegradationPipeline_APISR:
    def __init__(
        self,
        mode='batch',
        scale=2,
        resize_options=['area', 'bilinear', 'bicubic'],
        resize_prob=[0.2, 0.7, 0.1],
        resize_range=[0.1, 1.2],
        gaussian_noise_prob=0.5,
        noise_range=[1, 30],
        poisson_scale_range=[0.05, 3.0],
        gray_noise_prob=0.4,
        second_blur_prob=0.8,
        resize_prob2=[0.2, 0.7, 0.1],
        resize_range2=[0.15, 1.2],
        gaussian_noise_prob2=0.5,
        noise_range2=[1, 25],
        poisson_scale_range2=[0.05, 2.5],
        gray_noise_prob2=0.4,
        kernel_range=[3, 11],
        kernel_list=['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
        kernel_prob=[0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
        sinc_prob=0.1,
        blur_sigma=[0.2, 3.0],
        betag_range=[0.5, 4.0],
        betap_range=[1, 2],
        kernel_list2=['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
        kernel_prob2=[0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
        sinc_prob2=0.1,
        blur_sigma2=[0.2, 1.5],
        betag_range2=[0.5, 4.0],
        betap_range2=[1, 2],
        compression_codec1=['jpeg', 'webp'],
        compression_codec_prob1=[0.85, 0.15],
        jpeg_quality_range1=[20, 95],
        compression_codec2=['jpeg', 'webp'],
        compression_codec_prob2=[0.85, 0.15],
        jpeg_quality_range2=[20, 95]
    ):
        # Основные параметры
        self.mode = mode
        self.scale = scale

        # The first degradation process
        self.resize_options = resize_options
        self.resize_prob = resize_prob
        self.resize_range = resize_range
        self.gaussian_noise_prob = gaussian_noise_prob
        self.noise_range = noise_range
        self.poisson_scale_range = poisson_scale_range
        self.gray_noise_prob = gray_noise_prob

        # The second degradation process
        self.second_blur_prob = second_blur_prob
        self.resize_prob2 = resize_prob2
        self.resize_range2 = resize_range2
        self.gaussian_noise_prob2 = gaussian_noise_prob2
        self.noise_range2 = noise_range2
        self.poisson_scale_range2 = poisson_scale_range2
        self.gray_noise_prob2 = gray_noise_prob2

        # Blur kernel1
        self.kernel_range = kernel_range
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.sinc_prob = sinc_prob
        self.blur_sigma = blur_sigma
        self.betag_range = betag_range
        self.betap_range = betap_range

        # Blur kernel2
        self.kernel_list2 = kernel_list2
        self.kernel_prob2 = kernel_prob2
        self.sinc_prob2 = sinc_prob2
        self.blur_sigma2 = blur_sigma2
        self.betag_range2 = betag_range2
        self.betap_range2 = betap_range2

        # First image compression
        self.compression_codec1 = compression_codec1
        self.compression_codec_prob1 = compression_codec_prob1
        self.jpeg_quality_range1 = jpeg_quality_range1

        # Second image compression
        self.compression_codec2 = compression_codec2
        self.compression_codec_prob2 = compression_codec_prob2
        self.jpeg_quality_range2 = jpeg_quality_range2

        # Инициализация деградера
        self.degrader = degradation_v1()

    def process_batch(self, batch):
        """
        Process a batch of images through the degradation pipeline.

        Args:
            batch (torch.Tensor): Input batch of images with shape (B, C, H, W).

        Returns:
            torch.Tensor: Degraded batch of images with shape (B, C, H // scale, W // scale).
        """
        # Создаем словарь opt, используя атрибуты класса
        opt = {
            'scale': self.scale,
            'resize_options': self.resize_options,
            'resize_prob': self.resize_prob,
            'resize_range': self.resize_range,
            'gaussian_noise_prob': self.gaussian_noise_prob,
            'noise_range': self.noise_range,
            'poisson_scale_range': self.poisson_scale_range,
            'gray_noise_prob': self.gray_noise_prob,
            'second_blur_prob': self.second_blur_prob,
            'resize_prob2': self.resize_prob2,
            'resize_range2': self.resize_range2,
            'gaussian_noise_prob2': self.gaussian_noise_prob2,
            'noise_range2': self.noise_range2,
            'poisson_scale_range2': self.poisson_scale_range2,
            'gray_noise_prob2': self.gray_noise_prob2,
            'kernel_range': self.kernel_range,
            'kernel_list': self.kernel_list,
            'kernel_prob': self.kernel_prob,
            'sinc_prob': self.sinc_prob,
            'blur_sigma': self.blur_sigma,
            'betag_range': self.betag_range,
            'betap_range': self.betap_range,
            'kernel_list2': self.kernel_list2,
            'kernel_prob2': self.kernel_prob2,
            'sinc_prob2': self.sinc_prob2,
            'blur_sigma2': self.blur_sigma2,
            'betag_range2': self.betag_range2,
            'betap_range2': self.betap_range2,
            'compression_codec1': self.compression_codec1,
            'compression_codec_prob1': self.compression_codec_prob1,
            'jpeg_quality_range1': self.jpeg_quality_range1,
            'compression_codec2': self.compression_codec2,
            'compression_codec_prob2': self.compression_codec_prob2,
            'jpeg_quality_range2': self.jpeg_quality_range2
        }

        opt_copy = copy.deepcopy(opt)
        self.degrader.reset_kernels(opt_copy)

        # Process each image in the batch
        lr_batch = []
        for img in batch:
            img = img.unsqueeze(0).cuda()  # Add batch dimension and move to GPU
            lr_img = self.degrader.degradate_process(img, opt_copy, "tmp", 0, verbose=False)
            lr_batch.append(lr_img)

        # Concatenate all processed images into a single batch
        return torch.cat(lr_batch, dim=0)

def main():
    # Example usage for batch processing
    degrader_batch = ImageDegradationPipeline_APISR(mode='batch')
    # Create a random batch for demonstration
    dummy_batch = torch.rand(4, 3, 256, 256)  # batch_size=4, channels=3, height=256, width=256
    lr_batch = degrader_batch.process_batch(dummy_batch)
    print(f"Processed batch shape: {lr_batch.shape}")

if __name__ == '__main__':
    main()