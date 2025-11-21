import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.io import read_image, write_png
from torchvision.transforms import Resize
from torchvision.utils import save_image
from watermark_generation import generate_watermark_matrix

# CONFIG
EXTRACTED_DIR = "outputs/extracted_watermarks"
HEATMAP_DIR = "outputs/localization"
OVERLAY_DIR = "outputs/overlay"
os.makedirs(HEATMAP_DIR, exist_ok=True)
os.makedirs(OVERLAY_DIR, exist_ok=True)
IMAGE_SIZE = (218, 178)
THRESHOLD = 0.05  # differences below this are ignored

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Preprocessing ---
def preprocess_image(path):
    img = read_image(path).float() / 255.0  # normalize 0-1
    transform = Resize(IMAGE_SIZE)
    img = transform(img).unsqueeze(0)  # add batch dim
    return img.to(DEVICE)

# --- Overlay function ---
def overlay_heatmap_on_image(image, heatmap, alpha=0.5):
    """
    image: (C,H,W) torch tensor [0-1]
    heatmap: (1,H,W) torch tensor [0-1]
    alpha: blending factor
    returns: overlayed image (C,H,W) [0-1]
    """
    import matplotlib.cm as cm
    heatmap_np = heatmap.squeeze().cpu().numpy()
    heatmap_color = cm.hot(heatmap_np)[..., :3]  # get RGB from colormap
    heatmap_color = torch.tensor(heatmap_color).permute(2,0,1).float().to(DEVICE)
    overlay = alpha * heatmap_color + (1-alpha) * image
    overlay = overlay.clamp(0,1)
    return overlay

# --- Main Loop ---
for fname in os.listdir(EXTRACTED_DIR):
    wm_path = os.path.join(EXTRACTED_DIR, fname)
    extracted_wm = preprocess_image(wm_path)

    # Generate original watermark
    _, C, H, W = extracted_wm.shape
    original_wm = generate_watermark_matrix(1, H, W).to(DEVICE)

    # Compute tamper map
    tamper_map = torch.abs(extracted_wm - original_wm)
    tamper_map = tamper_map.mean(dim=1, keepdim=True)  # average over channels

    # Threshold small differences
    tamper_map = torch.where(tamper_map > THRESHOLD, tamper_map, torch.zeros_like(tamper_map))

    # Normalize
    if tamper_map.max() > 0:
        tamper_map = (tamper_map - tamper_map.min()) / (tamper_map.max() - tamper_map.min() + 1e-8)

    # Save standalone heatmap
    heatmap_path = os.path.join(HEATMAP_DIR, f"{fname}_heatmap.png")
    save_image(tamper_map, heatmap_path)
    print(f"Saved heatmap: {heatmap_path}")

    # Overlay on original image
    # Preprocess original image for overlay
    orig_image_path = wm_path  # if original image is same as extracted, adjust path if needed
    orig_image = preprocess_image(orig_image_path).squeeze(0)
    overlay_img = overlay_heatmap_on_image(orig_image, tamper_map, alpha=0.5)

    overlay_path = os.path.join(OVERLAY_DIR, f"{fname}_overlay.png")
    save_image(overlay_img, overlay_path)
    print(f"Saved overlay image: {overlay_path}")

    # Optional: visualize
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(tamper_map.squeeze().cpu(), cmap='hot')
    plt.title(f"Tamper Map: {fname}")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(overlay_img.squeeze().cpu().permute(1,2,0))
    plt.title(f"Overlay: {fname}")
    plt.axis("off")
    plt.show()
