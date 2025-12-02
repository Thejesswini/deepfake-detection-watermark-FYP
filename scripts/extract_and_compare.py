import os
import torch
from torchvision.io import read_image
from torchvision.transforms import Resize
from torchvision.utils import save_image
from revnet_model import RevNet3
import pandas as pd
from .evaluation_metrics import extract_watermark, compare_watermarks
from watermark_generation import generate_watermark_matrix

# --------- CONFIG ----------
WMARKED_DIR = "outputs/watermarked"     # your saved watermarked images
DEEPFAKE_DIR = "deepfakes"             # deepfake images
OUT_DIR = "outputs/extracted_watermarks"
MODEL_PATH = "model_weights.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = (218, 178)

os.makedirs(OUT_DIR, exist_ok=True)

# --------- MODEL ----------
model = RevNet3(channels=6).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# --------- FUNCTIONS ----------
def preprocess_image(path):
    """Read and normalize image as tensor [1,3,H,W]"""
    img = read_image(path).float() / 255.0
    transform = Resize(IMAGE_SIZE)
    img = transform(img)
    img = img.unsqueeze(0)
    return img.to(DEVICE)

# --------- MAIN LOOP ----------
data = {'index':[], 'image':[], 'psnr':[], 'mse':[], 'ssim':[], 'ncc':[]}
ct = 1
deepfaked = [1, 4, 5, 8, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 24, 31,32,34,37,38,42,39,45,46,48,47,49,51,53,56,55,58,60,41,61,62,63,64,65,66]
deepfaked = ["_"+str(i)+"." for i in deepfaked]
peano_watermark = generate_watermark_matrix(batch_size=1, height=218, width=178)

# for fname in os.listdir(WMARKED_DIR):
#     if True:#any(s in fname for s in deepfaked):
#         wmk_path = os.path.join(WMARKED_DIR, fname)
#         deepfake_path = os.path.join(DEEPFAKE_DIR, fname)

#         if not os.path.exists(deepfake_path):
#             print(f"Skipping {fname} as deepfake not found")
#             continue

#         # Load images
#         watermarked_img = preprocess_image(wmk_path)
#         deepfake_img = preprocess_image(deepfake_path)

#         # Extract watermark from deepfake
#         extracted_wm = extract_watermark(model, deepfake_img, DEVICE)
        
#         # Watermark from original watermarked image
#         orig_wm = extract_watermark(model, watermarked_img, DEVICE)
        
#         # Compare
#         #mse, psnr, ssim_score, ncc = compare_watermarks(extracted_wm, orig_wm)
#         mse, psnr, ssim_score, ncc = compare_watermarks(orig_wm, peano_watermark)
#         #print(f"{fname}: MSE={mse:.6f}, PSNR={psnr:.2f} dB")
        
#         data['index'].append(ct)
#         data['image'].append(fname)
#         data['psnr'].append(psnr)
#         data['mse'].append(mse)
#         data['ssim'].append(ssim_score)
#         data['ncc'].append(ncc)

#         # Save extracted watermark
#         # save_path = os.path.join(OUT_DIR, f"{fname}_wm.png")
#         # save_image(extracted_wm, save_path)
#         # print(f"Saved extracted watermark: {save_path}")
#         ct+=1


#inswapper
# DEEPFAKE_DIR = "inswapper_results"
# for fname in os.listdir(DEEPFAKE_DIR):    
#     wmk_path = os.path.join(WMARKED_DIR, f"watermarked_{fname.split('_')[-1]}")
#     deepfake_path = os.path.join(DEEPFAKE_DIR, fname)

#     if not os.path.exists(deepfake_path):
#         print(f"Skipping {fname} as deepfake not found")
#         continue

#     # Load images
#     watermarked_img = preprocess_image(wmk_path)
#     deepfake_img = preprocess_image(deepfake_path)

#     # Extract watermark from deepfake
#     extracted_wm = extract_watermark(model, deepfake_img, DEVICE)
    
#     # Watermark from original watermarked image
#     orig_wm = extract_watermark(model, watermarked_img, DEVICE)
    
#     # Compare
#     mse, psnr, ssim_score, ncc = compare_watermarks(extracted_wm, orig_wm)
#     #print(f"{fname}: MSE={mse:.6f}, PSNR={psnr:.2f} dB")
    
#     data['index'].append(ct)
#     data['image'].append(fname)
#     data['psnr'].append(psnr)
#     data['mse'].append(mse)
#     data['ssim'].append(ssim_score)
#     data['ncc'].append(ncc)

#     ct+=1
        
df = pd.DataFrame(data)
print(df.describe())
# df.to_csv('remaker_66_s-wm_t-wm.csv')
# print('saved')