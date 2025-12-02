# import os
# import torch
# from torchvision.io import read_image
# from torchvision.transforms import Resize
# from torchvision.utils import save_image
# from revnet_model import RevNet3
# import torch.nn.functional as F
# import numpy as np
# import pandas as pd
# from skimage.metrics import structural_similarity as ssim

# WMARKED_DIR = "outputs/watermarked"     # your saved watermarked images
# DEEPFAKE_DIR = "deepfakes"             # deepfake images
# MODEL_PATH = "model_weights.pth"
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# IMAGE_SIZE = (218, 178)

# deepfaked = [4, 5, 8, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 24, 31,32,34,37,38,42,39,45,46,48,47,49,51,53,56,55,58,60,41,61,62,63,64,65,66]
# def preprocess_image(path):
#     """Read and normalize image as tensor [1,3,H,W]"""
#     img = read_image(path).float() / 255.0
#     transform = Resize(IMAGE_SIZE)
#     img = transform(img)
#     img = img.unsqueeze(0)
#     return img.to(DEVICE)

# def extract_watermark(image_tensor):
#     """Extract watermark using model inverse"""
#     zeros = torch.zeros_like(image_tensor).to(DEVICE)
#     input_tensor = torch.cat([image_tensor, zeros], dim=1)
#     with torch.no_grad():
#         recovered = model.inverse(input_tensor)
#         _, recovered_wm = torch.chunk(recovered, 2, dim=1)
#     return recovered_wm

# def safe_3d_ssim(original_tensor, extracted_tensor):
#     """
#     Safely calculates SSIM for 3D floating-point tensors (C, H, W).
#     """
    
#     # --- STEP 1: Safe Conversion ---
#     # Detach from graph, move to CPU, convert to numpy
#     # We iterate to handle both PyTorch Tensors and Numpy arrays safely
#     if torch.is_tensor(original_tensor):
#         img_orig = original_tensor.detach().cpu().numpy()
#     else:
#         img_orig = original_tensor
        
#     if torch.is_tensor(extracted_tensor):
#         img_ext = extracted_tensor.detach().cpu().numpy()
#     else:
#         img_ext = extracted_tensor

#     # Your data is approx -0.015 to 0.835, so the range is ~0.85.
#     # We calculate it dynamically to be safe.
#     dynamic_range = img_orig.max() - img_orig.min()

#     # --- STEP 3: Calculate SSIM ---
#     # channel_axis=0 tells skimage that dimension 0 is the "Depth" or "Channels"
#     # This forces it to calculate 2D SSIM for each slice and average them.
#     # win_size=3 is safer than default (7) if your spatial dims (H,W) are small (<7 pixels).
    
#     try:
#         score = ssim(
#             img_orig, 
#             img_ext, 
#             data_range=dynamic_range, 
#             channel_axis=0  # Assumes input is (Depth, Height, Width)
#         )
#     except ValueError as e:
#         # Fallback if spatial dimensions are tiny (smaller than 7x7)
#         print(f"Standard SSIM failed ({e}). Retrying with smaller window...")
#         score = ssim(
#             img_orig, 
#             img_ext, 
#             data_range=dynamic_range, 
#             channel_axis=0,
#             win_size=3 # Smaller window for small feature maps
#         )

#     return score, dynamic_range

# def ncc(original_tensor, extracted_tensor):
# # --- Metric B: Normalized Cross-Correlation (NCC) ---
#     # Flattens the 3D volume to 1D vectors to check alignment
#     # Returns 1.0 for perfect match, closer to 0 for random noise.
#     if torch.is_tensor(original_tensor):
#         img_orig = original_tensor.detach().cpu().numpy()
#     else:
#         img_orig = original_tensor
        
#     if torch.is_tensor(extracted_tensor):
#         img_ext = extracted_tensor.detach().cpu().numpy()
#     else:
#         img_ext = extracted_tensor
        
#     flat_orig = img_orig.flatten()
#     flat_ext = img_ext.flatten()
    
#     # Zero-mean normalization (optional but recommended for signal matching)
#     # This removes "brightness" shifts and focuses purely on the PATTERN.
#     flat_orig_centered = flat_orig - np.mean(flat_orig)
#     flat_ext_centered = flat_ext - np.mean(flat_ext)
    
#     numerator = np.sum(flat_orig_centered * flat_ext_centered)
#     denominator = np.sqrt(np.sum(flat_orig_centered**2) * np.sum(flat_ext_centered**2))
    
#     final_ncc = numerator / denominator
    
#     return final_ncc

# model = RevNet3(channels=6).to(DEVICE)
# model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
# model.eval()

# maxi_o = 0
# mini_o = 0

# maxi_d = 0
# mini_d = 0

# ct = 0

# deepfaked = [str(i) for i in deepfaked]
# total_ssim_for_original = 0
# total_ssim_for_deepfake = 0
# total_ncc_for_original = 0
# total_ncc_for_deepfake = 0

# for fname in os.listdir(WMARKED_DIR):
#     wmk_path = os.path.join(WMARKED_DIR, fname)
#     if any(s in fname for s in deepfaked):
#         flag = True
#         ct+=1
#     else:
#         flag = False
        
#     deepfake_path = os.path.join(DEEPFAKE_DIR, fname)
    
#     # Load images
#     watermarked_img = preprocess_image(wmk_path)
#     deepfake_img = preprocess_image(deepfake_path)

#     # Extract watermark from deepfake
#     extracted_wm = extract_watermark(deepfake_img)
    
#     # Watermark from original watermarked image
#     orig_wm = extract_watermark(watermarked_img)
    
#     orig_wm = orig_wm.squeeze(0)
#     extracted_wm = extracted_wm.squeeze(0)
    
#     # maxi_o += orig_wm.max()
#     # mini_o += orig_wm.min()
#     # maxi_d += extracted_wm.max()
#     # mini_d += extracted_wm.min()
    
#     # 60
#     # orig wm -> max: tensor(0.8353) mini: tensor(-0.0151)
#     # 44
#     # extracted wm -> max: tensor(0.8366) mini: tensor(-0.0178)
#     ssim_score, _ = safe_3d_ssim(orig_wm, extracted_wm)
#     ncc_score = ncc(orig_wm, extracted_wm)
    
#     if flag:
#         total_ssim_for_deepfake+=ssim_score
#         total_ncc_for_deepfake+=ncc_score
#     else:
#         total_ssim_for_original+=ssim_score
#         total_ncc_for_original+=ncc_score

# print('ssim')
# print("original:",total_ssim_for_original/(66-ct), "count:",66-ct)
# print("deepfake:",total_ssim_for_deepfake/ct, "count:",ct)
# print('ncc')
# print("original:",total_ncc_for_original/(66-ct), "count:",66-ct)
# print("deepfake:",total_ncc_for_deepfake/ct, "count:",ct)

# #print('orig wm -> max:', maxi_o/ct, 'mini:',mini_o/ct)
# #print('extracted wm -> max:', maxi_d/ct, 'mini:',mini_d/ct)

# import os

# deepfaked = [1,4, 5, 8, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 24, 31,32,34,37,38,42,39,45,46,48,47,49,51,53,56,55,58,60,41,61,62,63,64,65,66]
# deepfaked.sort()
# deepfaked = ["_"+str(i)+"." for i in deepfaked]
# print(deepfaked)
# print(len(deepfaked))
# ct=0
# for fname in os.listdir("./outputs/watermarked"):
#     if any(s in fname for s in deepfaked):
#         print(fname)
#         ct+=1
# print(ct)

import torch
from revnet_model import RevNet3
import torch.optim as optim
from watermark_generation import generate_watermark_matrix
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

def extract_and_visualize_single(model, image_path):
    """
    Embeds a watermark into a single image, extracts it, and displays:
    1. Original Target Watermark
    2. The Watermarked Image (Container)
    3. The Extracted Watermark
    
    Args:
        model: The trained RevNet model
        image_path: Path to the input image
        device: 'cuda' or 'cpu'
    """
    model.eval()

    # 1. Prepare the Image
    # Ensure this transform matches your training exactly
    transform = transforms.Compose([
        transforms.ToTensor(),         # Converts to [0, 1]
    ])

    try:
        original_image = Image.open(image_path).convert('RGB')
        img_tensor = transform(original_image).unsqueeze(0).to(device) # Add batch dim: [1, 3, H, W]
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # 2. Generate the Target Watermark
    # Uses the helper function from your training context
    # Note: We assume generate_watermark_matrix is defined in your scope
    height, width = img_tensor.shape[2], img_tensor.shape[3]
    target_watermark = generate_watermark_matrix(1, height, width).to(device)

    # 3. Embedding Process (Forward Pass)
    with torch.no_grad():
        # Concat image and watermark
        input_tensor = torch.cat([img_tensor, target_watermark], dim=1)
        
        # Forward pass
        embedded_output = model(input_tensor)
        embedded_image, _ = torch.chunk(embedded_output, 2, dim=1)
        
        # 4. Simulation of "Saving/Transmission" (Optional but recommended)
        # This clamps values to valid image range to see real-world performance
        embedded_image = torch.clamp(embedded_image, 0, 1)

    # 5. Extraction Process (Inverse Pass)
    with torch.no_grad():
        # Prepare input for inverse: [Watermarked Image, Zero Matrix]
        # We use a zero matrix because we don't have the watermark at extraction time
        zeros = torch.zeros_like(target_watermark).to(device)
        extraction_input = torch.cat([embedded_image, zeros], dim=1)
        
        # Inverse pass
        recovered_output = model.inverse(extraction_input)
        _, recovered_watermark = torch.chunk(recovered_output, 2, dim=1)

    # 6. Visualization
    # Helper to convert tensors to numpy for plotting
    def to_numpy(tensor):
        return tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Create the plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # A. Target Watermark (Ground Truth)
    # If watermark is 3 channels, show as RGB. If 1 channel, show as gray.
    target_np = to_numpy(target_watermark)
    if target_np.shape[2] == 1:
        axes[0].imshow(target_np.squeeze(), cmap='gray')
    else:
        axes[0].imshow(target_np)
    axes[0].set_title("Original Watermark (Target)")
    axes[0].axis('off')

    # B. Watermarked Image
    axes[1].imshow(to_numpy(embedded_image))
    axes[1].set_title("Watermarked Image")
    axes[1].axis('off')

    # C. Extracted Watermark
    extracted_np = to_numpy(recovered_watermark)
    # Clamp to remove slight artifacts for better visualization
    extracted_np = extracted_np.clip(0, 1) 
    
    if extracted_np.shape[2] == 1:
        axes[2].imshow(extracted_np.squeeze(), cmap='gray')
    else:
        axes[2].imshow(extracted_np)
    axes[2].set_title("Extracted Watermark")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model
model = RevNet3(channels=6).to(device)
model.load_state_dict(torch.load("model_weights_n.pth", map_location=device))

extract_and_visualize_single(model=model, image_path=r"D:\SSN\DEEPFAKE\code\celebA\img_align_celeba\img_align_celeba\000002.jpg")

