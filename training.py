import torch
import torch.nn as nn
from revnet_model import RevNet3
from dataset_prep import get_data_loaders
import torch.optim as optim
from watermark_generation import generate_watermark_matrix
from noise import apply_corruptions
from evaluate import evaluate_model
import numpy as np
from torchvision import utils
import matplotlib.pyplot as plt
from config import EPOCHS
from training_fns import training_fn_set_trnfrmd_wtmk_2_0
from torchmetrics.image import PeakSignalNoiseRatio


def show_watermarking_results(model, loader, device, num_images=8):
    """
    Visualizes the output of the watermarking process on a batch of images.

    Args:
        model (nn.Module): The trained RevNet model.
        loader (DataLoader): The DataLoader for the test or validation set.
        device (torch.device): The device to run on ('cuda' or 'cpu').
        num_images (int): The number of images to display from the batch.
    """
    model.eval()

    # Take one batch
    data_iter = iter(loader)
    images = next(data_iter)

    # Restrict to requested number
    num_images = min(num_images, images.size(0))
    original_images = images[:num_images].to(device)

    # Generate watermark with same H, W as images
    watermarks = generate_watermark_matrix(
        batch_size=original_images.size(0),
        height=original_images.size(2),
        width=original_images.size(3)
    ).to(device)

    # Concatenate original image + watermark
    input_tensor = torch.cat([original_images, watermarks], dim=1)

    with torch.no_grad():
        # 1. Embed the watermark
        embedded = model(input_tensor)
        embedded_images, _ = torch.chunk(embedded, 2, dim=1)

        # 2. Apply corruption
        corrupted_images = apply_corruptions(embedded_images)

    # Stack all sets for visualization
    all_images = torch.cat([original_images, embedded_images, corrupted_images], dim=0)

    # Build grid
    grid = utils.make_grid(all_images, nrow=num_images, padding=2, normalize=False)

    plt.figure(figsize=(15, 6))
    np_grid = grid.cpu().numpy()
    plt.imshow(np.transpose(np_grid, (1, 2, 0)))
    plt.title('Top: Original | Middle: Watermarked | Bottom: Corrupted', fontsize=16)
    plt.axis('off')
    plt.show()


# ================== MAIN TRAINING ==================

# Get dataset (keeps original image dimensions now: 218 x 178)
train_loader, val_loader, test_loader = get_data_loaders(
    image_directory='./celebA/img_align_celeba/img_align_celeba',
    total_num=1000,
    train_per=0.8,
    val_per=0.1
)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model (3 channels RGB + 3 watermark channels)
model = RevNet3(channels=6).to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Loss + metrics
psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
criterion2 = nn.MSELoss()

# Training
training_fn_set_trnfrmd_wtmk_2_0(
    EPOCHS,
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    criterion1=psnr_metric,
    criterion2=criterion2,
    device=device
)

print("Training complete.")

# Evaluate
evaluate_model(model=model, device=device, val_loader=val_loader)

# Show results
show_watermarking_results(model, val_loader, device, num_images=8)
