



import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision.transforms import Resize
from torchvision.utils import save_image
import matplotlib.cm as cm
import torch.nn.functional as F

# --------- SINGLE IMAGE CONFIG ----------
EXTRACTED_DEEPFAKE_WM = "unwatermarked/deepfake_unwatermarked_wm.png"
EXTRACTED_ORIGINAL_WM = "unwatermarked/orig_unwatermarked_wm.png"
DEEPFAKE_HOST_IMAGE   = "unwatermarked/unw_2.jpg"

HEATMAP_PATH = "output/localization/img1_heatmap.png"
OVERLAY_PATH = "output/overlay/img1_overlay.png"

os.makedirs("output/localization", exist_ok=True)
os.makedirs("output/overlay", exist_ok=True)

IMAGE_SIZE = (218, 178)
THRESHOLD = 0.05
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Preprocessing ---
def preprocess_image(path):
    img = read_image(path).float() / 255.0  
    img = Resize(IMAGE_SIZE)(img)
    return img.unsqueeze(0).to(DEVICE)

# --- Overlay ---
def overlay_heatmap_on_image(image, heatmap, alpha=0.5):
    heatmap_np = heatmap.squeeze().cpu().numpy()
    heatmap_color = cm.hot(heatmap_np)[..., :3]
    heatmap_color = torch.tensor(heatmap_color).permute(2,0,1).float().to(DEVICE)
    overlay = alpha * heatmap_color + (1 - alpha) * image.squeeze(0)
    return overlay.clamp(0,1)

# --------- PROCESS SINGLE IMAGE ----------
print("\n---- Running Localization for ONE Image ----\n")

# Load extracted WMs
deepfake_wm = preprocess_image(EXTRACTED_DEEPFAKE_WM)
original_wm = preprocess_image(EXTRACTED_ORIGINAL_WM)

# Compute tamper map
tamper_map = torch.abs(deepfake_wm - original_wm).mean(dim=1, keepdim=True)
tamper_map = torch.where(tamper_map > THRESHOLD, tamper_map, torch.zeros_like(tamper_map))

# Normalize for visibility
if tamper_map.max() > 0:
    tamper_map = (tamper_map - tamper_map.min()) / (tamper_map.max() - tamper_map.min() + 1e-8)

# Save heatmap
save_image(tamper_map, HEATMAP_PATH)
print(f"Saved heatmap → {HEATMAP_PATH}")

# Load deepfake host image
host_img = preprocess_image(DEEPFAKE_HOST_IMAGE)

# Create overlay
overlay_img = overlay_heatmap_on_image(host_img, tamper_map)
save_image(overlay_img, OVERLAY_PATH)
print(f"Saved overlay → {OVERLAY_PATH}")

# Show visualization
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(tamper_map.squeeze().cpu(), cmap='hot')
plt.title("Tamper Map")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(overlay_img.cpu().permute(1,2,0))
plt.title("Overlay")
plt.axis("off")

plt.show()
