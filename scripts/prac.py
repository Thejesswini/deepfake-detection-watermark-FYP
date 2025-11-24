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

import os

deepfaked = [1,4, 5, 8, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 24, 31,32,34,37,38,42,39,45,46,48,47,49,51,53,56,55,58,60,41,61,62,63,64,65,66]
deepfaked.sort()
deepfaked = ["_"+str(i)+"." for i in deepfaked]
print(deepfaked)
print(len(deepfaked))
ct=0
for fname in os.listdir("./outputs/watermarked"):
    if any(s in fname for s in deepfaked):
        print(fname)
        ct+=1
print(ct)