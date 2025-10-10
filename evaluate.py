# validation be like
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import torch
from watermark_generation import generate_watermark_matrix
from noise import apply_corruptions
import torch.nn as nn

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
            corrupted_input_for_inverse = torch.cat([corrupted_image, embedded_watermark_part], dim=1)
            
            recovered = model.inverse(corrupted_input_for_inverse)
            _, recovered_watermark = torch.chunk(recovered, 2, dim=1)

            # Calculate and update watermark recovery MSE
            watermark_mse = extraction_metric(recovered_watermark, watermarks)
            total_watermark_mse += watermark_mse.item()
            
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