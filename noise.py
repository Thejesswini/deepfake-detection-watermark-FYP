from torchvision.transforms import ToTensor, ToPILImage
import random
from PIL import Image
import io
import torch
import torch.nn as nn

# Helper Function 1: Gaussian Noise
def apply_gaussian_noise(images, std=0.075):
    """Adds Gaussian noise to a batch of images."""
    # Ensure the input is a floating-point tensor for noise addition
    images = images.float()
    
    # Generate noise with the same shape as the images
    noise = torch.randn_like(images) * std
    
    # Add noise and clamp the result to the valid [0, 1] range
    noisy_images = images + noise
    #print(1)
    return torch.clamp(noisy_images, 0.0, 1.0)


# Helper Function 2: Simulated JPEG Compression
def apply_jpeg_compression(images):
    """Simulates JPEG compression for a batch of images."""
    # This process is computationally more intensive as it requires a loop
    images = images.cpu()
    compressed_images = []
    
    # Initialize converters
    to_pil = ToPILImage()
    to_tensor = ToTensor()
    
    for img_tensor in images:
        # Convert tensor to PIL Image
        pil_img = to_pil(img_tensor)
        
        # Save to an in-memory buffer with random JPEG quality
        buffer = io.BytesIO()
        quality = random.randint(50, 95) # Random quality between 50 and 95
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        
        # Read back from buffer and convert to tensor
        img_compressed_pil = Image.open(buffer)
        img_compressed_tensor = to_tensor(img_compressed_pil)
        compressed_images.append(img_compressed_tensor)
        
    
    return torch.stack(compressed_images).to(images.device)

import torch.nn.functional as F

def apply_diff_jpeg_approximation(images):
    """
    Differentiable approximation of JPEG:
    1. Downscales the image (loss of detail).
    2. Adds quantization noise (rounding error).
    3. Upscales it back (blurriness).
    """
    # Randomly choose a "quality" factor (scale)
    # Lower scale = Lower quality (more pixelated)
    scale = random.uniform(0.5, 0.9) 
    
    # 1. Downsample (Loss of high-frequency info)
    # mode='area' or 'bilinear' creates gradients
    small_images = F.interpolate(images, scale_factor=scale, mode='bilinear', align_corners=False)
    
    # 2. Add Quantization Noise (Simulate 8-bit rounding loss)
    # We add noise to the small image before resizing back
    noise = (torch.rand_like(small_images) - 0.5) / 255.0
    small_images = small_images + noise
    
    # 3. Upsample back to original size
    restored_images = F.interpolate(small_images, size=(images.shape[2], images.shape[3]), mode='bilinear', align_corners=False)
    
    return torch.clamp(restored_images, 0.0, 1.0)

# Helper Function 3: Sharpening Filter
def apply_sharpening(images):
    """Applies a sharpening kernel to a batch of images."""
    num_channels = images.size(1)
    
    # Define a classic 3x3 sharpening kernel
    kernel_base = torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=torch.float32)
    
    kernel = kernel_base.view(1, 1, 3, 3).repeat(num_channels, 1, 1, 1)
    
    
    kernel = kernel.to(images.device)
    
    conv = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, 
                     groups=num_channels, padding=1, bias=False)
    
    # Set the layer's weight to be our sharpening kernel
    conv.weight = nn.Parameter(kernel, requires_grad=False)
    
    # Apply the convolution and clamp the output
    sharpened_images = conv(images)
    #print(3)
    return torch.clamp(sharpened_images, 0.0, 1.0)

def apply_quantization_noise(images):
        # 1. Scale to 0-255 (simulating standard image format)
        images = images * 255.0
        
        # 2. Add random uniform noise between -0.5 and 0.5
        # This simulates the error introduced by rounding to the nearest integer
        # without breaking the gradient flow (unlike actual rounding).
        noise = (torch.rand_like(images) - 0.5)
        images = images + noise
        
        # 3. Scale back to 0-1
        return images / 255.0

def apply_corruptions(images):
    """
    Randomly applies one of three corruptions to a batch of images:
    Gaussian noise, JPEG compression, or sharpening.
    
    Args:
        images (torch.Tensor): A batch of images with shape [B, C, H, W]
                               and pixel values in the [0, 1] range.
                               
    Returns:
        torch.Tensor: The batch of corrupted images.
    """
    corruption_funcs = [
        apply_gaussian_noise, #---
        # apply_jpeg_compression,
        apply_sharpening,
        apply_quantization_noise,
        apply_diff_jpeg_approximation
    ]
    
    
    chosen_corruption = random.choice(corruption_funcs)
    
    return chosen_corruption(images)