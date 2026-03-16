import os
import torch
from revnet_model import RevNet3
from scripts.evaluation_metrics import extract_watermark, convert_img_path_to_tensor, compare_watermarks
from scripts.visualize_localization import overlay_heatmap_on_image
 
if __name__=='__main__':
    MODEL_PATH = "model_weights.pth"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMAGE_SIZE = (218, 178)
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    THRESHOLD = 0.05

    model = RevNet3(channels=6).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    wmk_path = r"code\only_watermarked\watermarked_0001.png"  # Example watermarked image path
    deepfake_path = r"code\only_deepfakes\deepfake_0001.png"  # Corresponding deepfake image path

    # Load images
    watermarked_img = convert_img_path_to_tensor(BASE_DIR, wmk_path, DEVICE)
    deepfake_img = convert_img_path_to_tensor(BASE_DIR, deepfake_path, DEVICE)

    # Extract watermark from deepfake
    extracted_wm = extract_watermark(model, deepfake_img, DEVICE)

    # Watermark from original watermarked image
    orig_wm = extract_watermark(model, watermarked_img, DEVICE)

    # Compare
    mse, psnr, ssim, _ = compare_watermarks(extracted_wm, orig_wm)
    print(f"MSE={mse:.6f}, PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")
    
    # plot original and deepfake images
    import matplotlib.pyplot as plt

    # watermarked_img_np = watermarked_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    # deepfake_img_np = deepfake_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # axs[0].imshow(watermarked_img_np)
    # axs[0].set_title("Watermarked Image")
    # axs[0].axis('off')
    # axs[1].imshow(deepfake_img_np)
    # axs[1].set_title("Deepfake Image")
    # axs[1].axis('off')
    
    # #set mse, psnr value as suptitle
    # plt.suptitle(f"MSE={mse:.6f}, PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")
    # plt.show()
    

    tamper_map = torch.abs(extracted_wm - orig_wm)
    tamper_map = tamper_map.mean(dim=1, keepdim=True)  # average over channels

    # Threshold small differences
    tamper_map = torch.where(tamper_map > THRESHOLD, tamper_map, torch.zeros_like(tamper_map))

    # Normalize
    if tamper_map.max() > 0:
        tamper_map = (tamper_map - tamper_map.min()) / (tamper_map.max() - tamper_map.min() + 1e-8)

    overlay_img = overlay_heatmap_on_image(deepfake_img, tamper_map, DEVICE, alpha=0.5)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(tamper_map[0][0], cmap='gray')
    axs[0].set_title("Watermarked Image")
    axs[0].axis('off')
    axs[1].imshow(overlay_img[0], cmap='gray')
    axs[1].set_title("Deepfake Image")
    axs[1].axis('off')
    
    #set mse, psnr value as suptitle
    plt.suptitle(f"MSE={mse:.6f}, PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")
    plt.show()