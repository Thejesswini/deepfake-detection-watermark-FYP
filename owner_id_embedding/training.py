from .models import WatermarkEmbedder, WatermarkExtractor
from .loss import image_loss, watermark_loss, calculate_ber, calculate_psnr
import torch
from dataset_prep import get_data_loaders
from .watermark import owner_id_to_watermark
from config import BATCH_SIZE, EPOCHS

def train_step(embedder, extractor, optimizer, images, watermarks, alpha=1.0):
    optimizer.zero_grad()

    watermarked_images = embedder(images, watermarks)
    extracted_watermarks = extractor(watermarked_images)

    loss_img = image_loss(images, watermarked_images)
    loss_wm = watermark_loss(watermarks, extracted_watermarks)
    psnr = calculate_psnr(images, watermarked_images)
    ber = calculate_ber(watermarks, extracted_watermarks)
    
    total_loss = loss_img + alpha * loss_wm
    total_loss.backward()
    optimizer.step()

    return total_loss.item(), loss_img.item(), loss_wm.item(), psnr, ber

# Dummy data
if __name__=='__main__':
    embedder = WatermarkEmbedder(wm_channels=1)
    extractor = WatermarkExtractor(wm_channels=1)

    optimizer = torch.optim.Adam(
        list(embedder.parameters()) + list(extractor.parameters()),
        lr=1e-4
    )

    train_loader, val_loader, test_loader = get_data_loaders(
    image_directory=r'D:\SSN\DEEPFAKE\code\celebA\img_align_celeba\img_align_celeba',
    total_num=1000,
    train_per=0.8,
    val_per=0.1
    )
    
    watermarks = owner_id_to_watermark(
        owner_id="CREATOR_001",
        batch_size=BATCH_SIZE
    )

    print("starting training:")
    for epoch in range(EPOCHS):
        embedder.train()
        extractor.train()

        epoch_loss = 0.0
        img_loss = 0.0
        wmk_loss = 0.0
        psnr_tot = 0.0
        ber_tot = 0.0

        for images in train_loader:
            loss, i_l, w_l, psnr, ber = train_step(
                embedder,
                extractor,
                optimizer,
                images,
                watermarks)
            epoch_loss += loss
            img_loss += i_l
            wmk_loss += w_l
            psnr_tot += psnr
            ber_tot += ber

        epoch_loss /= len(train_loader)
        img_loss /= len(train_loader)
        wmk_loss /= len(train_loader)
        psnr_tot /= len(train_loader)
        ber_tot /= len(train_loader)
        print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Image loss = {img_loss:.4f}, PSNR = {psnr_tot:.4f}, Watermark loss = {wmk_loss:.4f}, BER = {ber_tot:.4f}")

    torch.save(embedder, r"D:\SSN\DEEPFAKE\code\owner_id_embedding\embedder_1.pth")
    torch.save(extractor, r"D:\SSN\DEEPFAKE\code\owner_id_embedding\extractor_1.pth")