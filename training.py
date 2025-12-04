'''
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
from show_watermark_comparison import show_watermark_comparison

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
        embedded_images, embedded_watermark_part= torch.chunk(embedded, 2, dim=1)

        # ----------------------------------------------------
        # 2. DEBUG: Perform CLEAN (Pre-Save) Extraction
        # ----------------------------------------------------
        # The inverse model needs the embedded image and the watermark part it expects
        
        # NOTE: For your training_fn_set_trnfrmd_wtmk_2_0, the inverse expects torch.zeros_like 
        # for the watermark part (the constant 'key'), so we must replicate that logic here:
        
        # Use the expected zero tensor as the inverse input 'key'
        embedded_watermark_part_key = torch.zeros_like(embedded_watermark_part).detach()
        
        corrupted_input_for_clean_inverse = torch.cat([embedded_images, embedded_watermark_part_key], dim=1)
        
        recovered_clean = model.inverse(corrupted_input_for_clean_inverse)
        _, recovered_watermark_clean = torch.chunk(recovered_clean, 2, dim=1)
        
        # Calculate MSE against the original watermarks
        loss_clean_extraction = criterion2(recovered_watermark_clean, watermarks)
        
        print(f"\n✨ DEBUG: Clean (Pre-Save) Watermark MSE: {loss_clean_extraction.item():.8f}")
        # -
        
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
psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
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

show_watermark_comparison(model, val_loader, device, num_images=5, criterion2=criterion2)

# Evaluate
evaluate_model(model=model, device=device, val_loader=val_loader)

# Show & save watermarked results
show_and_save_watermarking_results(model, val_loader, device, num_images=30)
'''



#gemini 1 st
'''
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
from show_watermark_comparison import show_watermark_comparison

# --- NEW CONFIGURATION CONSTANT ---
# Define the expected length of the binary message
MESSAGE_LENGTH = 256 
# ----------------------------------


def show_and_save_watermarking_results(model, loader, device, num_images=30, save_dir="outputs/watermarked"):
    """
    Visualizes and saves original, watermarked, and corrupted images side-by-side.
    MODIFIED to handle the new dual output from generate_watermark_matrix and model.inverse.
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # Take one batch
    data_iter = iter(loader)
    images = next(data_iter)

    num_images = min(num_images, images.size(0))
    original_images = images[:num_images].to(device)
    

    # Generate watermark
    # --- MODIFIED: Get BOTH W_map (watermarks) AND the binary message ---
    # We must explicitly call generate_watermark_matrix with the message_length argument now.
    watermarks, binary_message = generate_watermark_matrix(
        batch_size=original_images.size(0),
        height=original_images.size(2),
        width=original_images.size(3),
        message_length=MESSAGE_LENGTH 
    )
    watermarks, binary_message = watermarks.to(device), binary_message.to(device)
    # --------------------------------------------------------------------


    # Concatenate original + watermark
    input_tensor = torch.cat([original_images, watermarks], dim=1)

    with torch.no_grad():
        # Embed watermark
        embedded = model(input_tensor)
        embedded_images, embedded_watermark_part = torch.chunk(embedded, 2, dim=1)

        # ----------------------------------------------------
        # 2. DEBUG: Perform CLEAN (Pre-Save) Extraction
        # ----------------------------------------------------
        
        # Use the expected zero tensor as the inverse input 'key'
        embedded_watermark_part_key = torch.zeros_like(embedded_watermark_part).detach()
        
        corrupted_input_for_clean_inverse = torch.cat([embedded_images, embedded_watermark_part_key], dim=1)
        
        # --- MODIFIED: model.inverse returns two things: recovered_features (ignored) and the message ---
        _, recovered_message_clean = model.inverse(corrupted_input_for_clean_inverse)
        
        # Calculate MSE against the ORIGINAL BINARY MESSAGE (the true target)
        loss_clean_extraction = criterion2(recovered_message_clean, binary_message)
        
        print(f"\n✨ DEBUG: Clean (Pre-Save) Watermark MSE (Message): {loss_clean_extraction.item():.8f}")
        # -
        
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
# --- MODIFIED: Pass MESSAGE_LENGTH to the RevNet3 constructor ---
model = RevNet3(channels=6, message_length=MESSAGE_LENGTH).to(device)
# ----------------------------------------------------------------

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Loss + metric
psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
criterion2 = nn.MSELoss()

# Train
# NOTE: This relies on training_fn_set_trnfrmd_wtmk_2_0 being correctly updated 
# to handle the dual output of generate_watermark_matrix and model.inverse.
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
#model.load_state_dict(torch.load("model_weights_1.pth", map_location=device))
#model.eval()


print("Training complete.")

show_watermark_comparison(model, val_loader, device, num_images=5, criterion2=criterion2)

# Evaluate
evaluate_model(model=model, device=device, val_loader=val_loader)

# Show & save watermarked results
show_and_save_watermarking_results(model, val_loader, device, num_images=30)
'''


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
from show_watermark_comparison import show_watermark_comparison

# --- NEW CONFIGURATION CONSTANT ---
# Define the expected length of the binary message
MESSAGE_LENGTH = 256 
# ----------------------------------


def show_and_save_watermarking_results(model, loader, device, num_images=30, save_dir="outputs/watermarked"):
    """
    Visualizes and saves original, watermarked, and corrupted images side-by-side.
    MODIFIED for message extraction and corrected device placement.
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # Take one batch
    data_iter = iter(loader)
    images = next(data_iter)

    num_images = min(num_images, images.size(0))
    original_images = images[:num_images].to(device)
    

    # Generate watermark
    # --- MODIFIED: Get BOTH W_map (watermarks) AND the binary message ---
    watermarks, binary_message = generate_watermark_matrix(
        batch_size=original_images.size(0),
        height=original_images.size(2),
        width=original_images.size(3),
        message_length=MESSAGE_LENGTH 
    )
    # Correct device placement for both tensors
    watermarks, binary_message = watermarks.to(device), binary_message.to(device)
    # --------------------------------------------------------------------


    # Concatenate original + watermark
    input_tensor = torch.cat([original_images, watermarks], dim=1)

    with torch.no_grad():
        # Embed watermark
        embedded = model(input_tensor)
        embedded_images, embedded_watermark_part = torch.chunk(embedded, 2, dim=1)

        # ----------------------------------------------------
        # 2. DEBUG: Perform CLEAN (Pre-Save) Extraction
        # ----------------------------------------------------
        
        # Use the expected zero tensor as the inverse input 'key'
        embedded_watermark_part_key = torch.zeros_like(embedded_watermark_part).detach()
        
        corrupted_input_for_clean_inverse = torch.cat([embedded_images, embedded_watermark_part_key], dim=1)
        
        # model.inverse returns two things: recovered_features (ignored) and the message logits
        _, recovered_message_logits = model.inverse(corrupted_input_for_clean_inverse)
        
        # To calculate MSE for display, we must apply Sigmoid to the logits
        recovered_message_clean = torch.sigmoid(recovered_message_logits)
        
        # Calculate MSE against the ORIGINAL BINARY MESSAGE (the true target)
        # NOTE: criterion2 is now nn.BCEWithLogitsLoss() but we use it here for quick MSE calculation
        # It's better to explicitly define a temporary MSE for consistency if criterion2 is BCE.
        temp_mse = nn.MSELoss()
        loss_clean_extraction = temp_mse(recovered_message_clean, binary_message)
        
        print(f"\n✨ DEBUG: Clean (Pre-Save) Watermark MSE (Message): {loss_clean_extraction.item():.8f}")
        # -
        
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
model = RevNet3(channels=6, message_length=MESSAGE_LENGTH).to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Loss + metric
psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
# --- CRITICAL FIX 1: Use BCEWithLogitsLoss for stable binary training (Fix 4) ---
criterion2 = nn.BCEWithLogitsLoss()
# -----------------------------------------------------------------------------

# Train
training_fn_set_trnfrmd_wtmk_2_0(
    EPOCHS,
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    criterion1=psnr_metric,
    criterion2=criterion2, # This is now BCEWithLogitsLoss
    device=device
)

# Load trained model weights
#model.load_state_dict(torch.load("model_weights_1.pth", map_location=device))
#model.eval()


print("Training complete.")

# NOTE: show_watermark_comparison and evaluate_model must also be updated 
# to handle BCEWithLogitsLoss and logits (as we did in the previous response).
show_watermark_comparison(model, val_loader, device, num_images=5, criterion2=criterion2)

# Evaluate
evaluate_model(model=model, device=device, val_loader=val_loader)

# Show & save watermarked results
show_and_save_watermarking_results(model, val_loader, device, num_images=30)