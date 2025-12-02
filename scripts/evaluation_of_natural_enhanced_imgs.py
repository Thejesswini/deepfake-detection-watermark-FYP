#import
import os
import cv2
import torch
from noise import apply_gaussian_noise, apply_jpeg_compression, apply_sharpening
from .evaluation_metrics import extract_watermark, compare_watermarks
from revnet_model import RevNet3
import pandas as pd
from watermark_generation import generate_watermark_matrix

#load images
def load_images(wmarked_dir="outputs/watermarked"):
    """
    takes a folder name, extracts all the images and makes it into [n, 3, h, w] tensor
    """
    tensors = []
    for fname in os.listdir(wmarked_dir):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            wmk_path = os.path.join(wmarked_dir, fname)
            
            wm_img = cv2.imread(wmk_path)
            
            # Convert BGR → RGB
            wm_img = cv2.cvtColor(wm_img, cv2.COLOR_BGR2RGB)

            # Convert to float32 and scale to [0,1]
            wm_img = wm_img.astype("float32") / 255.0

            # Convert to tensor (H, W, C) → (C, H, W)
            tensor = torch.from_numpy(wm_img).permute(2, 0, 1)

            tensors.append(tensor)
    tensors = torch.stack(tensors)
    return tensors


if __name__=='__main__':
    # Stack into one tensor: (N, 3, H, W)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "model_weights.pth"
    data = {'index':[], 'psnr':[], 'mse':[], 'ssim':[], 'ncc':[]}
    peano_watermark = generate_watermark_matrix(batch_size=1, height=218, width=178)
    
    model = RevNet3(channels=6).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    original_images = load_images()
    print(original_images.shape)

    #apply corruption
    corrupted_images = apply_jpeg_compression(original_images)

    #extract watermark from original and corrupted
    extracted_wm_from_original = extract_watermark(model=model, image_tensor=original_images, DEVICE=DEVICE)
    extracted_wm_from_corrupted = extract_watermark(model=model, image_tensor=corrupted_images, DEVICE=DEVICE)

    for image in range(len(extracted_wm_from_original)):
        orig = extracted_wm_from_original[image]
        corrupt = extracted_wm_from_corrupted[image]
        
        orig = orig.unsqueeze(0)
        corrupt = corrupt.unsqueeze(0)
        
        mse, psnr, ssim_score, ncc = compare_watermarks(corrupt, orig)
        
        data['index'].append(image+1)
        data['psnr'].append(psnr)
        data['mse'].append(mse)
        data['ssim'].append(ssim_score)
        data['ncc'].append(ncc)


    #compare using metrics
df = pd.DataFrame(data)
print(df.describe())
# df.to_csv('img_enhancements_66.csv')
# print('saved')