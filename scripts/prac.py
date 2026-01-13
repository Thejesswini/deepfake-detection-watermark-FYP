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

# import torch
# from revnet_model import RevNet3
# import torch.optim as optim
# from watermark_generation import generate_watermark_matrix
# import torch.nn as nn
# import matplotlib.pyplot as plt
# from PIL import Image
# from torchvision import transforms

# def extract_and_visualize_single(model, image_path):
#     """
#     Embeds a watermark into a single image, extracts it, and displays:
#     1. Original Target Watermark
#     2. The Watermarked Image (Container)
#     3. The Extracted Watermark
    
#     Args:
#         model: The trained RevNet model
#         image_path: Path to the input image
#         device: 'cuda' or 'cpu'
#     """
#     model.eval()

#     # 1. Prepare the Image
#     # Ensure this transform matches your training exactly
#     transform = transforms.Compose([
#         transforms.ToTensor(),         # Converts to [0, 1]
#     ])

#     try:
#         original_image = Image.open(image_path).convert('RGB')
#         img_tensor = transform(original_image).unsqueeze(0).to(device) # Add batch dim: [1, 3, H, W]
#     except Exception as e:
#         print(f"Error loading image: {e}")
#         return

#     # 2. Generate the Target Watermark
#     # Uses the helper function from your training context
#     # Note: We assume generate_watermark_matrix is defined in your scope
#     height, width = img_tensor.shape[2], img_tensor.shape[3]
#     target_watermark = generate_watermark_matrix(1, height, width).to(device)

#     # 3. Embedding Process (Forward Pass)
#     with torch.no_grad():
#         # Concat image and watermark
#         input_tensor = torch.cat([img_tensor, target_watermark], dim=1)
        
#         # Forward pass
#         embedded_output = model(input_tensor)
#         embedded_image, _ = torch.chunk(embedded_output, 2, dim=1)
        
#         # 4. Simulation of "Saving/Transmission" (Optional but recommended)
#         # This clamps values to valid image range to see real-world performance
#         embedded_image = torch.clamp(embedded_image, 0, 1)

#     # 5. Extraction Process (Inverse Pass)
#     with torch.no_grad():
#         # Prepare input for inverse: [Watermarked Image, Zero Matrix]
#         # We use a zero matrix because we don't have the watermark at extraction time
#         zeros = torch.zeros_like(target_watermark).to(device)
#         extraction_input = torch.cat([embedded_image, zeros], dim=1)
        
#         # Inverse pass
#         recovered_output = model.inverse(extraction_input)
#         _, recovered_watermark = torch.chunk(recovered_output, 2, dim=1)

#     # 6. Visualization
#     # Helper to convert tensors to numpy for plotting
#     def to_numpy(tensor):
#         return tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

#     # Create the plot
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))

#     # A. Target Watermark (Ground Truth)
#     # If watermark is 3 channels, show as RGB. If 1 channel, show as gray.
#     target_np = to_numpy(target_watermark)
#     if target_np.shape[2] == 1:
#         axes[0].imshow(target_np.squeeze(), cmap='gray')
#     else:
#         axes[0].imshow(target_np)
#     axes[0].set_title("Original Watermark (Target)")
#     axes[0].axis('off')

#     # B. Watermarked Image
#     axes[1].imshow(to_numpy(embedded_image))
#     axes[1].set_title("Watermarked Image")
#     axes[1].axis('off')

#     # C. Extracted Watermark
#     extracted_np = to_numpy(recovered_watermark)
#     # Clamp to remove slight artifacts for better visualization
#     extracted_np = extracted_np.clip(0, 1) 
    
#     if extracted_np.shape[2] == 1:
#         axes[2].imshow(extracted_np.squeeze(), cmap='gray')
#     else:
#         axes[2].imshow(extracted_np)
#     axes[2].set_title("Extracted Watermark")
#     axes[2].axis('off')

#     plt.tight_layout()
#     plt.show()


# Device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Model
# model = RevNet3(channels=6).to(device)
# model.load_state_dict(torch.load("model_weights_n.pth", map_location=device))

# extract_and_visualize_single(model=model, image_path=r"D:\SSN\DEEPFAKE\code\celebA\img_align_celeba\img_align_celeba\000002.jpg")
import pandas as pd
d = {
 'Epoch': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100],
 'Loss': [0.396581, 0.195288, 0.112787, 0.07852, 0.060984, 0.048297, 0.037176, 0.027476, 0.021246, 0.016791, 0.012583, 0.011698, 0.010736, 0.010104, 0.009489, 0.008998, 0.008725, 0.007957, 0.007463, 0.008041, 0.03707, 0.030672, 0.028868, 0.026366, 0.025812, 0.023463, 0.022305, 0.021558, 0.020146, 0.019931, 0.020295, 0.020058, 0.020723, 0.019259, 0.019938, 0.018996, 0.018578, 0.018703, 0.017845, 0.017591, 0.017555, 0.017962, 0.017784, 0.019152, 0.017514, 0.017138, 0.019574, 0.017928, 0.018006, 0.01843, 0.016754, 0.016861, 0.018112, 0.017243, 0.017915, 0.018105, 0.017583, 0.016754, 0.017332, 0.016995, 0.058927, 0.05984, 0.058239, 0.058753, 0.060141, 0.059209, 0.058102, 0.0565, 0.058749, 0.057102, 0.056335, 0.057172, 0.056622, 0.057412, 0.057633, 0.056572, 0.057791, 0.057201, 0.056421, 0.058029, 0.058586, 0.058625, 0.05764, 0.056086, 0.057112, 0.05671, 0.057922, 0.056875, 0.056778, 0.054859, 0.057363, 0.056037, 0.056832, 0.059068, 0.057328, 0.055998, 0.057123, 0.055938, 0.057837, 0.055423],
 'PSNR': [6.55, 9.27, 11.49, 13.01, 14.37, 15.55, 16.72, 18.45, 19.64, 20.8, 22.04, 22.46, 22.94, 23.24, 23.53, 23.83, 24.07, 24.12, 24.64, 24.28, 28.38, 28.25, 28.08, 27.98, 28.94, 29.5, 29.81, 29.91, 29.86, 30.15, 30.02, 30.91, 30.65, 30.88, 30.75, 31.09, 31.01, 31.12, 31.31, 31.23, 31.37, 31.36, 31.29, 31.21, 31.36, 31.47, 31.15, 31.35, 31.32, 31.28, 31.36, 31.43, 31.37, 31.39, 31.32, 31.33, 31.36, 31.41, 31.42, 31.43, 32.24, 32.27, 32.37, 32.39, 32.37, 32.41, 32.46, 32.48, 32.47, 32.48, 32.51, 32.52, 32.52, 32.53, 32.53, 32.53, 32.53, 32.53, 32.54, 32.54, 32.53, 32.53, 32.53, 32.54, 32.54, 32.54, 32.54, 32.54, 32.55, 32.55, 32.55, 32.55, 32.55, 32.55, 32.55, 32.55, 32.55, 32.55, 32.55, 32.55],
 'Ext_MSE': [0.014822, 0.006151, 0.011388, 0.009274, 0.009048, 0.007627, 0.004951, 0.007465, 0.002078, 0.008569, 0.001444, 0.001621, 0.008377, 0.00156, 0.001702, 0.009425, 0.001101, 0.002148, 0.001661, 0.001236, 0.004863, 0.004605, 0.018328, 0.005716, 0.003681, 0.003775, 0.006092, 0.020194, 0.003351, 0.003369, 0.003675, 0.002489, 0.003466, 0.003591, 0.003384, 0.002729, 0.002182, 0.003312, 0.016777, 0.003365, 0.00186, 0.002151, 0.001813, 0.015631, 0.001826, 0.016955, 0.001991, 0.002105, 0.001854, 0.003065, 0.003155, 0.001766, 0.015997, 0.001768, 0.003013, 0.003002, 0.00204, 0.003001, 0.002022, 0.001725, 0.004869, 0.003311, 0.0041, 0.005182, 0.003509, 0.003987, 0.005333, 0.004212, 0.023638, 0.004174, 0.005505, 0.024169, 0.005511, 0.003922, 0.024377, 0.003912, 0.005499, 0.004237, 0.003929, 0.024698, 0.003912, 0.00422, 0.005481, 0.003925, 0.024093, 0.003921, 0.005504, 0.004396, 0.005513, 0.024576, 0.024159, 0.024443, 0.00434, 0.003935, 0.004236, 0.004369, 0.0243, 0.023974, 0.024478, 0.004568],
 'Sanity_Check_MSE': [None, None, None, None, None, None, None, None, None, 0.021, None, None, None, None, None, None, None, None, None, 0.05802, None, None, None, None, None, None, None, None, None, 0.07603, None, None, None, None, None, None, None, None, None, 0.0773, None, None, None, None, None, None, None, None, None, 0.07386, None, None, None, None, None, None, None, None, None, 0.07819, None, None, None, None, None, None, None, None, None, 0.08298, None, None, None, None, None, None, None, None, None, 0.08273, None, None, None, None, None, None, None, None, None, 0.08267, None, None, None, None, None, None, None, None, None, 0.0827]
}

d = pd.DataFrame(d)
d.to_csv("revnet2_ChannelMixing_tanh_Deep_metrics.csv")