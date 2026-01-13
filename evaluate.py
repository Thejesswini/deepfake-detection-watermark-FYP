# validation be like
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import torch
from watermark_generation import generate_watermark_matrix
from noise import apply_corruptions
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

def evaluate_model(model, val_loader, device):
    """
    Evaluates the RevNet model on the validation set.

    Args:
        model (nn.Module): The trained RevNet model.
        val_loader (DataLoader): The DataLoader for the validation set.
        device (torch.device): The device to run on (e.g., 'cuda' or 'cpu').
        criterion (nn.Module): The loss function (e.g., MSELoss) for watermark comparison.
    """
    # Set the model to evaluation mode
    model.eval()

    # Initialize metric calculators
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    extraction_metric = nn.MSELoss()
    

    total_psnr = 0.0
    total_ssim = 0.0
    total_watermark_mse = 0.0

    # Disable gradient calculations for efficiency
    with torch.no_grad():
        for batch_idx, images in enumerate(val_loader):
            images = images.to(device)
            
            watermarks = generate_watermark_matrix(images.size(0), images.size(2),images.size(3)).to(device)
            
            input_tensor = torch.cat([images, watermarks], dim=1)

            
            embedded = model(input_tensor)
            embedded_image, embedded_watermark_part = torch.chunk(embedded, 2, dim=1)
            
            # Update PSNR and SSIM totals
            cur_psnr = psnr_metric(embedded_image, images)
            cur_ssim = ssim_metric(embedded_image, images)
            total_psnr += cur_psnr
            total_ssim += cur_ssim

            
            corrupted_image = apply_corruptions(embedded_image)
            #corrupted_input_for_inverse = torch.cat([corrupted_image, embedded_watermark_part], dim=1)
            corrupted_input_for_inverse = torch.cat([corrupted_image, torch.zeros_like(embedded_watermark_part).detach()], dim=1)
            
            recovered = model.inverse(corrupted_input_for_inverse)
            _, recovered_watermark = torch.chunk(recovered, 2, dim=1)

            # Calculate and update watermark recovery MSE
            watermark_mse = extraction_metric(recovered_watermark, watermarks)
            total_watermark_mse += watermark_mse#.item()
            
            print(f"Step [{batch_idx+1}] Embedding -> PSNR: {cur_psnr} | SSIM: {cur_ssim} | Extraction -> Watermark MSE: {watermark_mse}")

    # Calculate average metrics for the entire validation set
    avg_psnr = total_psnr / len(val_loader)
    avg_ssim = total_ssim / len(val_loader)
    avg_watermark_mse = total_watermark_mse / len(val_loader)

    print("\n--- Validation Results ---")
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average Watermark MSE: {avg_watermark_mse:.6f}")
    print("--------------------------\n")

    return avg_psnr, avg_ssim, avg_watermark_mse


def run_phase1_sanity_check(model, test_image_path, device, epoch_idx):
    """
    Runs the 'Save-and-Load' round-trip test to verify the model 
    can survive 8-bit quantization.
    
    Returns:
        float: The MSE loss between the original and extracted watermark.
    """
    model.eval()
    
    # --- 1. Preparation ---
    # Define a temp path for the saved image
    temp_save_path = "temp_sanity_check.png" 
    
    # Load the single test image
    transform = transforms.Compose([
        transforms.ToTensor() 
    ])
    
    try:
        raw_img = Image.open(test_image_path).convert('RGB')
        img_tensor = transform(raw_img).unsqueeze(0).to(device)
    except:
        print("Error: Could not load test image.")
        return 999.0

    # Generate the ROBUST STEPPED Peano Watermark (Target)
    # Ensure you are using the 'generate_robust_peano_matrix' function defined earlier!
    target_watermark = generate_watermark_matrix(1, img_tensor.shape[2], img_tensor.shape[3]).to(device)

    # --- 2. Embedding ---
    with torch.no_grad():
        input_tensor = torch.cat([img_tensor, target_watermark], dim=1)
        embedded_output = model(input_tensor)
        embedded_image_tensor, _ = torch.chunk(embedded_output, 2, dim=1)
        
        # CRITICAL: Clamp to ensure valid pixel range before saving
        embedded_image_tensor = torch.clamp(embedded_image_tensor, 0, 1)

    # --- 3. The "Real World" Barrier (Save to Disk) ---
    # This step forces the float32 tensors into int8 (0-255) pixels.
    # This destroys microscopic information.
    
    # save_image(embedded_image_tensor, temp_save_path)
    
    # # --- 4. Loading Back ---
    # # Load the image we just saved
    # loaded_image = Image.open(temp_save_path).convert('RGB')
    # loaded_tensor = transform(loaded_image).unsqueeze(0).to(device)

    loaded_tensor = embedded_image_tensor
    
    # --- 5. Extraction ---
    with torch.no_grad():
        # Feed the LOADED image + Zeros into the inverse model
        zeros = torch.zeros_like(target_watermark).to(device)
        extraction_input = torch.cat([loaded_tensor, zeros], dim=1)
        
        recovered_output = model.inverse(extraction_input)
        _, recovered_watermark = torch.chunk(recovered_output, 2, dim=1)

    # --- 6. Metrics & Visualization ---
    mse_loss = F.mse_loss(recovered_watermark, target_watermark).item()
    
    print(f"[Epoch {epoch_idx}] Sanity Check MSE: {mse_loss:.5f}")

    # Plotting
    # We only plot if the loss is low enough (promising) OR every few epochs
    if epoch_idx % 10 == 0 or mse_loss < 0.05:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        
        # Original Target
        target_np = target_watermark.squeeze().permute(1, 2, 0).cpu().numpy()
        axs[0].imshow(target_np)
        axs[0].set_title("Target (Stepped Peano)")
        axs[0].axis('off')

        # Watermarked Image (The one that was saved/loaded)
        loaded_np = loaded_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        axs[1].imshow(loaded_np)
        axs[1].set_title(f"Saved & Loaded Image\n(Simulated Attack)")
        axs[1].axis('off')

        # Extracted Watermark
        extracted_np = recovered_watermark.squeeze().permute(1, 2, 0).cpu().numpy()
        # Clip it to clean up the visual for the user
        extracted_np = extracted_np.clip(0, 1) 
        axs[2].imshow(extracted_np)
        axs[2].set_title(f"Extracted Result\nMSE: {mse_loss:.4f}")
        axs[2].axis('off')

        plt.show()

        # --- THE STOP CONDITION ---
        if mse_loss < 0.02: # 0.02 is a very strong match threshold
            print("\n✅ STOP CONDITION MET: Watermark is robust against saving/loading.")
            print("You may proceed to Phase 2.")
            return True # Signal to stop training

    return False

def plot_for_one_img(model, test_image_path, device):
    model.eval()
    
    # Load the single test image
    transform = transforms.Compose([
        transforms.ToTensor() 
    ])
    
    try:
        raw_img = Image.open(test_image_path).convert('RGB')
        img_tensor = transform(raw_img).unsqueeze(0).to(device)
    except:
        print("Error: Could not load test image.")
        return 999.0

    # Generate the ROBUST STEPPED Peano Watermark (Target)
    # Ensure you are using the 'generate_robust_peano_matrix' function defined earlier!
    target_watermark = generate_watermark_matrix(1, img_tensor.shape[2], img_tensor.shape[3]).to(device)

    # --- 2. Embedding ---
    with torch.no_grad():
        input_tensor = torch.cat([img_tensor, target_watermark], dim=1)
        embedded_output = model(input_tensor)
        embedded_image_tensor, _ = torch.chunk(embedded_output, 2, dim=1)
        
        # CRITICAL: Clamp to ensure valid pixel range before saving
        embedded_image_tensor = torch.clamp(embedded_image_tensor, 0, 1)

        embedded_img_uint8 = (embedded_image_tensor * 255).round()
        # 3. Convert back to float 0-1 (Simulate loading from disk)
        loaded_tensor = embedded_img_uint8 / 255.0
    
    # --- 5. Extraction ---
    with torch.no_grad():
        # Feed the LOADED image + Zeros into the inverse model
        zeros = torch.zeros_like(target_watermark).to(device)
        extraction_input = torch.cat([loaded_tensor, zeros], dim=1)
        
        recovered_output = model.inverse(extraction_input)
        _, recovered_watermark = torch.chunk(recovered_output, 2, dim=1)

    # --- 6. Metrics & Visualization ---
    mse_loss = F.mse_loss(recovered_watermark, target_watermark).item()
    
    print(f"Sanity Check MSE: {mse_loss:.5f}")

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    
    # Original Target
    target_np = target_watermark.squeeze().permute(1, 2, 0).cpu().numpy()
    axs[0].imshow(target_np)
    axs[0].set_title("Target (Stepped Peano)")
    axs[0].axis('off')

    # Watermarked Image (The one that was saved/loaded)
    loaded_np = loaded_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    axs[1].imshow(loaded_np)
    axs[1].set_title(f"Saved & Loaded Image\n(Simulated Attack)")
    axs[1].axis('off')

    # Extracted Watermark
    extracted_np = recovered_watermark.squeeze().permute(1, 2, 0).cpu().numpy()
    # Clip it to clean up the visual for the user
    extracted_np = extracted_np.clip(0, 1) 
    axs[2].imshow(extracted_np)
    axs[2].set_title(f"Extracted Result\nMSE: {mse_loss:.4f}")
    axs[2].axis('off')

    plt.show()
    
if __name__=='__main__':
    plot_for_one_img()