import os
import torch
from torchvision.io import read_image
from torchvision.transforms import Resize
from torchvision.utils import save_image
from revnet_model import RevNet3
import torch.nn.functional as F
import numpy as np

# --------- CONFIG ----------
WMARKED_DIR = "outputs/watermarked"     # your saved watermarked images
DEEPFAKE_DIR = "deepfakes"             # deepfake images
OUT_DIR = "outputs/extracted_watermarks"
MODEL_PATH = "model_weights.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = (218, 178)

os.makedirs(OUT_DIR, exist_ok=True)

# --------- MODEL ----------
model = RevNet3(channels=6).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# --------- FUNCTIONS ----------

def preprocess_image(path):
    """Read and normalize image as tensor [1,3,H,W]"""
    img = read_image(path).float() / 255.0
    transform = Resize(IMAGE_SIZE)
    img = transform(img)
    img = img.unsqueeze(0)
    return img.to(DEVICE)

def extract_watermark(image_tensor):
    """Extract watermark using model inverse"""
    zeros = torch.zeros_like(image_tensor).to(DEVICE)
    input_tensor = torch.cat([image_tensor, zeros], dim=1)
    with torch.no_grad():
        recovered = model.inverse(input_tensor)
        _, recovered_wm = torch.chunk(recovered, 2, dim=1)
    return recovered_wm

def compare_watermarks(extracted, original):
    """Compute similarity metric (MSE & PSNR)"""
    eps = 1e-10
    mse = F.mse_loss(extracted, original).item()
    psnr = 10 * np.log10(1.0 / (mse + eps))  # add eps to avoid div by zero
    return mse, psnr

# --------- MAIN LOOP ----------

for fname in os.listdir(WMARKED_DIR):
    wmk_path = os.path.join(WMARKED_DIR, fname)
    deepfake_path = os.path.join(DEEPFAKE_DIR, fname)

    if not os.path.exists(deepfake_path):
        print(f"Skipping {fname} as deepfake not found")
        continue

    # Load images
    watermarked_img = preprocess_image(wmk_path)
    deepfake_img = preprocess_image(deepfake_path)

    # Extract watermark from deepfake
    extracted_wm = extract_watermark(deepfake_img)

    # Watermark from original watermarked image
    orig_wm = extract_watermark(watermarked_img)

    # Compare
    mse, psnr = compare_watermarks(extracted_wm, orig_wm)
    print(f"{fname}: MSE={mse:.6f}, PSNR={psnr:.2f} dB")

    # Save extracted watermark
    save_path = os.path.join(OUT_DIR, f"{fname}_wm.png")
    save_image(extracted_wm, save_path)
    print(f"Saved extracted watermark: {save_path}")
