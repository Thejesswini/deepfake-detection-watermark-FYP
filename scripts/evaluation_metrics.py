import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from revnet_model import RevNet3
from watermark_generation import generate_watermark_matrix

def extract_watermark(model, image_tensor, DEVICE):
    """Extract watermark using model inverse"""
    zeros = torch.zeros_like(image_tensor).to(DEVICE)
    input_tensor = torch.cat([image_tensor, zeros], dim=1)
    print(input_tensor.shape)
    with torch.no_grad():
        recovered = model.inverse(input_tensor)
        _, recovered_wm = torch.chunk(recovered, 2, dim=1)
    return recovered_wm

def safe_3d_ssim(img_orig, img_ext):
    """
    Safely calculates SSIM for 3D floating-point tensors (C, H, W).
    """

    # Your data is approx -0.015 to 0.835, so the range is ~0.85.
    # We calculate it dynamically to be safe.
    dynamic_range = img_orig.max() - img_orig.min()

    # --- STEP 3: Calculate SSIM ---
    # channel_axis=0 tells skimage that dimension 0 is the "Depth" or "Channels"
    # This forces it to calculate 2D SSIM for each slice and average them.
    # win_size=3 is safer than default (7) if your spatial dims (H,W) are small (<7 pixels).
    
    try:
        score = ssim(
            img_orig, 
            img_ext, 
            data_range=dynamic_range, 
            channel_axis=0  # Assumes input is (Depth, Height, Width)
        )
    except ValueError as e:
        # Fallback if spatial dimensions are tiny (smaller than 7x7)
        print(f"Standard SSIM failed ({e}). Retrying with smaller window...")
        score = ssim(
            img_orig, 
            img_ext, 
            data_range=dynamic_range, 
            channel_axis=0,
            win_size=3 # Smaller window for small feature maps
        )

    return score, dynamic_range

def ncc_score(img_orig, img_ext):
    """
    input shape: [3,h,w]
    """
    # --- Metric B: Normalized Cross-Correlation (NCC) ---
    # Flattens the 3D volume to 1D vectors to check alignment
    # Returns 1.0 for perfect match, closer to 0 for random noise.
        
    flat_orig = img_orig.flatten()
    flat_ext = img_ext.flatten()
    
    # Zero-mean normalization (optional but recommended for signal matching)
    # This removes "brightness" shifts and focuses purely on the PATTERN.
    flat_orig_centered = flat_orig - np.mean(flat_orig)
    flat_ext_centered = flat_ext - np.mean(flat_ext)
    
    numerator = np.sum(flat_orig_centered * flat_ext_centered)
    denominator = np.sqrt(np.sum(flat_orig_centered**2) * np.sum(flat_ext_centered**2))
    
    final_ncc = numerator / denominator
    
    return final_ncc

def mse_psnr(extracted, original):
    """Compute similarity metric (MSE & PSNR)
    takes input in the form of [1,3,h,w]
    """
    eps = 1e-10
    mse = F.mse_loss(extracted, original).item()
    psnr = 10 * np.log10(1.0 / (mse + eps))  # add eps to avoid div by zero
    return mse, psnr

def compare_watermarks(extracted, original):
    mse, psnr = mse_psnr(extracted=extracted, original=original)
    
    original_tensor = original.squeeze(0)
    extracted_tensor = extracted.squeeze(0)
    
    # --- STEP 1: Safe Conversion ---
    # Detach from graph, move to CPU, convert to numpy
    # We iterate to handle both PyTorch Tensors and Numpy arrays safely
    if torch.is_tensor(original_tensor):
        orig = original_tensor.detach().cpu().numpy()
    else:
        orig = original_tensor
        
    if torch.is_tensor(extracted_tensor):
        ext = extracted_tensor.detach().cpu().numpy()
    else:
        ext = extracted_tensor
        
    ssim_score, _ = safe_3d_ssim(orig, ext)
    ncc = ncc_score(orig, ext)
    
    return mse, psnr, ssim_score, ncc

if __name__=='__main__':
    from torchvision.io import read_image
    path1 = r"D:\SSN\DEEPFAKE\code\varied_bg.jpg"
    path2 = r"c:\Users\theju\Downloads\varied_bg_deepfaked.png"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img1 = read_image(path1).float() / 255.0
    img1 = img1.unsqueeze(0)
    img1.to(DEVICE)
    
    from PIL import Image
    from torchvision import transforms

    img2 = Image.open(path2).convert("RGB")
    transform = transforms.ToTensor()
    img2 = transform(img2)
    img2 = img2.unsqueeze(0)
    img2.to(DEVICE)
    
    model = RevNet3(channels=6).to(DEVICE)
    model.load_state_dict(torch.load("model_weights.pth", map_location=DEVICE))
    model.eval()
    
    wm1 = extract_watermark(model, img1, DEVICE)
    wm2 = extract_watermark(model, img2, DEVICE)
    # print(img2.shape)
    # og_peano_curve = generate_watermark_matrix(batch_size=1, height=218, width=178)
    
    mse, psnr, ssim_, ncc = compare_watermarks(wm2, wm1)
    print(f"MSE: {mse:.4f}; PSNR: {psnr:.4f}dB; SSIM: {ssim_:.4f}; NCC: {ncc:.4f}")
    