'''
import torch
from torchvision import utils
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn as nn # Need this for criterion2 if not globally available
from watermark_generation import generate_watermark_matrix
from noise import apply_corruptions

# NOTE: criterion2 (nn.MSELoss) needs to be passed in or defined.

def show_watermark_comparison(model, loader, device, num_images=3, criterion2=nn.MSELoss()):
    """
    Visualizes the Original Image, Watermarked Image, Corrupted Image, 
    Original Watermark, and Recovered Watermark for comparison.
    """
    model.eval()
    
    data_iter = iter(loader)
    images = next(data_iter)

    # Use only a few images for a clear comparison
    num_images = min(num_images, images.size(0))
    original_images = images[:num_images].to(device)

    # 1. Generate Watermark (same as in training)
    watermarks = generate_watermark_matrix(
        batch_size=original_images.size(0),
        height=original_images.size(2),
        width=original_images.size(3)
    ).to(device)

    # Concatenate original + watermark
    input_tensor = torch.cat([original_images, watermarks], dim=1)

    with torch.no_grad():
        # 2. Forward Pass (Embedding)
        embedded = model(input_tensor)
        embedded_images, embedded_watermark_part = torch.chunk(embedded, 2, dim=1)

        # 3. Corruption
        corrupted_images = apply_corruptions(embedded_images)

        # 4. Inverse Pass (Extraction)
        # Use the expected zero tensor (the 'key') as the watermark part for inverse pass
        embedded_watermark_key = torch.zeros_like(embedded_watermark_part).detach()
        corrupted_input_for_inverse = torch.cat([corrupted_images, embedded_watermark_key], dim=1)
        
        recovered = model.inverse(corrupted_input_for_inverse)
        _, recovered_watermark = torch.chunk(recovered, 2, dim=1)
        
        # Calculate Extraction MSE for this batch
        batch_mse = criterion2(recovered_watermark, watermarks).item()
        
        print(f"\nBatch Watermark MSE (Post-Corruption): {batch_mse:.6f}")

        # ----------------------------------------------------
        # 5. Visualization Setup
        # ----------------------------------------------------
        
        # We only need one channel of the watermark for visualization (they are all identical)
        original_w_1c = watermarks[:, 0:1, :, :]
        recovered_w_1c = recovered_watermark[:, 0:1, :, :]

        # Combine all tensors for the grid:
        # Row 1: Original Images
        # Row 2: Corrupted Images
        # Row 3: Original Watermarks (1 channel)
        # Row 4: Recovered Watermarks (1 channel)
        
        all_visuals = torch.cat([
            original_images, 
            corrupted_images,
            original_w_1c.repeat(1, 3, 1, 1), # Repeat 1C W to 3C for visibility
            recovered_w_1c.repeat(1, 3, 1, 1)
        ], dim=0)

        grid = utils.make_grid(all_visuals, nrow=num_images, padding=2, normalize=False)

        # 6. Display
        plt.figure(figsize=(15, 8))
        np_grid = grid.cpu().numpy()
        plt.imshow(np.transpose(np_grid, (1, 2, 0)))
        plt.title(
            f'Watermarking Comparison (MSE: {batch_mse:.4f})\n'
            'Top: Original | 2nd: Corrupted | 3rd: Original Watermark | Bottom: Recovered Watermark', 
            fontsize=12
        )
        plt.axis('off')
        plt.show()
        
'''
import torch
from torchvision import utils
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn as nn 
from watermark_generation import generate_watermark_matrix
from noise import apply_corruptions

# Define the MESSAGE_LENGTH used in your training (assuming 256)
MESSAGE_LENGTH = 256


def show_watermark_comparison(model, loader, device, num_images=3, criterion2=nn.MSELoss()):
    """
    MODIFIED: Correctly applies Sigmoid to the message logits before calculating 
    the final MSE metric for display.
    """
    model.eval()
    
    data_iter = iter(loader)
    images = next(data_iter)

    # Use only a few images for a clear comparison
    num_images = min(num_images, images.size(0))
    original_images = images[:num_images].to(device)

    # 1. Generate Watermark
    watermarks, message_tensor = generate_watermark_matrix(
        batch_size=original_images.size(0),
        height=original_images.size(2),
        width=original_images.size(3),
        message_length=MESSAGE_LENGTH
    )
    watermarks, message_tensor = watermarks.to(device), message_tensor.to(device)

    # Concatenate original + watermark
    input_tensor = torch.cat([original_images, watermarks], dim=1)

    with torch.no_grad():
        # 2. Forward Pass (Embedding)
        embedded = model(input_tensor)
        embedded_images, embedded_watermark_part = torch.chunk(embedded, 2, dim=1)

        # 3. Corruption
        corrupted_images = apply_corruptions(embedded_images)

        # 4. Inverse Pass (Extraction)
        embedded_watermark_key = torch.zeros_like(embedded_watermark_part).detach()
        corrupted_input_for_inverse = torch.cat([corrupted_images, embedded_watermark_key], dim=1)
        
        # recovered_message is the LOGITS output from model.inverse
        recovered_features, recovered_message_logits = model.inverse(corrupted_input_for_inverse)
        
        # NOTE: We use the recovered_features to get the W_map for VISUALIZATION ONLY.
        _, recovered_watermark = torch.chunk(recovered_features, 2, dim=1)
        
        # Calculate Extraction MSE (CRITICAL FIX)
        # --- Apply Sigmoid to the logits to get [0, 1] range for MSE comparison ---
        recovered_message = torch.sigmoid(recovered_message_logits)
        batch_mse = criterion2(recovered_message, message_tensor).item()
        # -------------------------------------------------------------------------
        
        print(f"\nBatch Watermark MSE (Post-Corruption): {batch_mse:.6f} (Message MSE)")

        # ----------------------------------------------------
        # 5. Visualization Setup (Uses W_map for display)
        # ----------------------------------------------------
        
        # We only need one channel of the watermark for visualization
        original_w_1c = watermarks[:, 0:1, :, :]
        recovered_w_1c = recovered_watermark[:, 0:1, :, :]

        # Combine all tensors for the grid:
        all_visuals = torch.cat([
            original_images, 
            corrupted_images,
            original_w_1c.repeat(1, 3, 1, 1), 
            recovered_w_1c.repeat(1, 3, 1, 1)
        ], dim=0)

        grid = utils.make_grid(all_visuals, nrow=num_images, padding=2, normalize=False)

        # 6. Display
        plt.figure(figsize=(15, 8))
        np_grid = grid.cpu().numpy()
        
        plt.imshow(np.transpose(np_grid, (1, 2, 0)).clip(0, 1)) 
        
        plt.title(
            f'Watermarking Comparison (Message MSE: {batch_mse:.4f})\n'
            'Top: Original | 2nd: Corrupted | 3rd: Original Watermark Map | Bottom: Recovered Watermark Map', 
            fontsize=12
        )
        plt.axis('off')
        plt.show()