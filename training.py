import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import utils
import matplotlib.pyplot as plt
import numpy as np

from revnet_model import RevNet3
from dataset_prep import get_data_loaders
from watermark_generation import generate_watermark_matrix
from noise import apply_corruptions
from evaluate import evaluate_model
from training_fns import training_fn_set_trnfrmd_wtmk_2_0
from config import EPOCHS
from torchmetrics.image import PeakSignalNoiseRatio


def show_and_save_watermarking_results(model, loader, device, num_images=30, save_dir="outputs/watermarked"):
    """
    Visualizes and saves original, watermarked, and corrupted images side-by-side.

    Args:
        model (nn.Module): Trained RevNet model.
        loader (DataLoader): DataLoader for validation/test set.
        device (torch.device): CPU or CUDA.
        num_images (int): Number of images to process.
        save_dir (str): Directory to save images.
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # Take one batch
    data_iter = iter(loader)
    images = next(data_iter)

    num_images = min(num_images, images.size(0))
    original_images = images[:num_images].to(device)

    # Generate watermark
    watermarks = generate_watermark_matrix(
        batch_size=original_images.size(0),
        height=original_images.size(2),
        width=original_images.size(3)
    ).to(device)

    # Concatenate original + watermark
    input_tensor = torch.cat([original_images, watermarks], dim=1)

    with torch.no_grad():
        # Embed watermark
        embedded = model(input_tensor)
        embedded_images, _ = torch.chunk(embedded, 2, dim=1)

        # Apply corruption
        corrupted_images = apply_corruptions(embedded_images)

      # ----- Save only watermarked images -----
    for idx, img in enumerate(embedded_images):
        save_path = os.path.join(save_dir, f"watermarked_{idx+1}.png")
        utils.save_image(img, save_path)
        print(f"Saved: {save_path}")

    # ----- Visualize all images in a grid -----
    all_images = torch.cat([original_images, embedded_images, corrupted_images], dim=0)
    grid = utils.make_grid(all_images, nrow=num_images, padding=2, normalize=False)

    plt.figure(figsize=(15, 6))
    np_grid = grid.cpu().numpy()
    plt.imshow(np.transpose(np_grid, (1, 2, 0)))
    plt.title('Top: Original | Middle: Watermarked | Bottom: Corrupted', fontsize=16)
    plt.axis('off')
    plt.show()


# ================== MAIN TRAINING ==================

# Load dataset
train_loader, val_loader, test_loader = get_data_loaders(
    image_directory='/Users/sem5/sem7/deepfake/celebA/img_align_celeba/img_align_celeba',
    total_num=1000,
    train_per=0.8,
    val_per=0.1
)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model
model = RevNet3(channels=6).to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Loss + metric
psnr_metric = nn.MSELoss()
criterion2 = nn.MSELoss()

# Train
training_fn_set_trnfrmd_wtmk_2_0(
    EPOCHS,
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    criterion1=psnr_metric,
    criterion2=criterion2,
    device=device
)


# Load trained model weights
#model.load_state_dict(torch.load("model_weights.pth", map_location=device))
#model.eval()


print("Training complete.")

# Evaluate
evaluate_model(model=model, device=device, val_loader=val_loader)

# Show & save watermarked results
show_and_save_watermarking_results(model, val_loader, device, num_images=30)
