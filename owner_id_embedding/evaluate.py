import torch
import torch.nn.functional as F
from dataset_prep import get_data_loaders
from .models import WatermarkEmbedder, WatermarkExtractor
from .loss import image_loss, watermark_loss, calculate_ber, calculate_psnr
from .watermark import owner_id_to_watermark
# -------------------------------------------------
# Evaluation Function
# -------------------------------------------------

def evaluate(embedder, extractor, test_loader, watermark, device):
    embedder.eval()
    extractor.eval()

    total_loss = 0
    total_img_loss = 0
    total_wm_loss = 0
    total_psnr = 0
    total_ber = 0

    with torch.no_grad():
        for images in test_loader:

            images = images.to(device)

            # Expand watermark to batch size
            wm_batch = watermark.expand(images.size(0), -1, -1, -1).to(device)

            # Forward pass
            watermarked = embedder(images, wm_batch)
            extracted = extractor(watermarked)

            # Losses
            img_l = image_loss(images, watermarked)
            wm_l = watermark_loss(wm_batch, extracted)
            loss = img_l + wm_l

            # Metrics
            psnr = calculate_psnr(images, watermarked)
            ber = calculate_ber(wm_batch, extracted)

            # Accumulate
            total_loss += loss.item()
            total_img_loss += img_l.item()
            total_wm_loss += wm_l.item()
            total_psnr += psnr
            total_ber += ber

    num_batches = len(test_loader)

    print("\n--- Evaluation Results ---")
    print(f"Loss: {total_loss / num_batches:.4f}")
    print(f"Image Loss (MSE): {total_img_loss / num_batches:.4f}")
    print(f"PSNR: {total_psnr / num_batches:.4f} dB")
    print(f"Watermark Loss (BCE): {total_wm_loss / num_batches:.4f}")
    print(f"BER: {total_ber / num_batches:.4f}")

# -------------------------------------------------
# Main Execution
# -------------------------------------------------

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------
    # Load Models (replace with your actual classes)
    # -------------------------------------------------

    embedder = WatermarkEmbedder().to(device)
    extractor = WatermarkExtractor().to(device)

    embedder = torch.load(
        r".\owner_id_embedding\embedder_1.pth",
        map_location=device,
        weights_only=False
        )
    
    extractor = torch.load(
        r".\owner_id_embedding\extractor_1.pth",
        map_location=device,
        weights_only=False
        )
    #embedder.load_state_dict(torch.load(r".\owner_id_embedding\embedder_1.pth", map_location=device))
    #extractor.load_state_dict(torch.load(r".\owner_id_embedding\extractor_1.pth", map_location=device))

    # -------------------------------------------------
    # Load Test Data
    # -------------------------------------------------
    _, val_loader, _ = get_data_loaders(
    image_directory=r'D:\SSN\DEEPFAKE\code\celebA\img_align_celeba\img_align_celeba',
    total_num=1000,
    train_per=0.8,
    val_per=0.1
    )

    # -------------------------------------------------
    # Load Watermark
    # -------------------------------------------------

    watermark = owner_id_to_watermark("CREATOR_001", h=32, w=32, batch_size=1)

    # -------------------------------------------------
    # Run Evaluation
    # -------------------------------------------------
    evaluate(embedder, extractor, val_loader, watermark, device)


if __name__ == "__main__":
    main()
