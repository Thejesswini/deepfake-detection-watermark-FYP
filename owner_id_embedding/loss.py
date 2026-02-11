import torch.nn.functional as F
import torch

def image_loss(original, watermarked):
    return F.mse_loss(watermarked, original)

def watermark_loss(original_wm, extracted_wm):
    return F.binary_cross_entropy(extracted_wm, original_wm)

def calculate_psnr(original, watermarked):
    """
    original, watermarked: tensors of shape [B, C, H, W] in range [0, 1]
    returns: average PSNR over batch
    """
    mse = F.mse_loss(watermarked, original, reduction='none')
    mse = mse.view(mse.size(0), -1).mean(dim=1)  # per-image MSE

    psnr = 10 * torch.log10(1.0 / mse)
    return psnr.mean().item()

def calculate_ber(original_wm, extracted_wm):
    """
    original_wm, extracted_wm: tensors of shape [B, 1, H, W]
    returns: average BER over batch
    """
    extracted_bits = (extracted_wm > 0.5).float()
    bit_errors = torch.abs(original_wm - extracted_bits)

    ber = bit_errors.mean()
    return ber.item()
