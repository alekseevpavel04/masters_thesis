import numpy as np

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
