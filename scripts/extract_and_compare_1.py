import torch
import numpy as np
from torchvision.io import read_image
from torchvision.transforms import Resize
from torchvision.utils import save_image
from revnet_model import RevNet3
import torch.nn.functional as F

# ---------------- CONFIG ----------------
ORIG_IMG = "outputs/watermarked/watermarked_19.png"     # put your unwatermarked clean image here
DEEPFAKE_IMG = "unwatermarked/deepfake/uw2_deep.png" # put your deepfake version here
MODEL_PATH = "model_weights.pth"

OUT_ORIG_WM = "unwatermarked/orig_unwatermarked_wm.png"
OUT_DEEPFAKE_WM = "unwatermarked/deepfake_unwatermarked_wm.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = (218, 178)

# ---------------- HELPER FUNCTIONS ----------------
def preprocess_image(path):
    img = read_image(path).float() / 255.0
    img = Resize(IMAGE_SIZE)(img)
    return img.unsqueeze(0).to(DEVICE)

def extract_wm(model, image_tensor):
    zeros = torch.zeros_like(image_tensor)
    input_tensor = torch.cat([image_tensor, zeros], dim=1)
    with torch.no_grad():
        recovered = model.inverse(input_tensor)
    _, w = torch.chunk(recovered, 2, dim=1)
    return w

def compare(a, b):
    eps = 1e-10
    mse = F.mse_loss(a, b).item()
    psnr = 10 * np.log10(1.0 / (mse + eps))
    return mse, psnr

# ---------------- LOAD MODEL ----------------
model = RevNet3(channels=6).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ---------------- LOAD IMAGES ----------------
orig_img = preprocess_image(ORIG_IMG)
deepfake_img = preprocess_image(DEEPFAKE_IMG)

# ---------------- EXTRACT WATERMARKS ----------------
orig_wm = extract_wm(model, orig_img)
deepfake_wm = extract_wm(model, deepfake_img)

# ---------------- COMPARE ----------------
mse, psnr = compare(orig_wm, deepfake_wm)
print(f"\n--- UNWATERMARKED TEST RESULT ---")
print(f"MSE  = {mse:.6f}")
print(f"PSNR = {psnr:.2f} dB\n")

# ---------------- SAVE OUTPUTS ----------------
save_image(orig_wm, OUT_ORIG_WM)
save_image(deepfake_wm, OUT_DEEPFAKE_WM)

print(f"Saved original-extracted watermark   → {OUT_ORIG_WM}")
print(f"Saved deepfake-extracted watermark   → {OUT_DEEPFAKE_WM}")

'''
from PIL import Image, ImageChops
a = Image.open("unwatermarked/orig_unwatermarked_wm.png")
b = Image.open("unwatermarked/deepfake_unwatermarked_wm.png")
a.show(title="orig_wm")
b.show(title="deepfake_wm")
# show absolute diff image
diff = ImageChops.difference(a.convert("RGB"), b.convert("RGB"))
diff.show(title="diff (should be all black if identical)")

'''
'''
import numpy as np
from PIL import Image
a = np.array(Image.open("outputs/watermarked/watermarked_19.png").convert("RGB"), dtype=np.float32)/255.0
b = np.array(Image.open("unwatermarked/unw_2.jpg").convert("RGB"), dtype=np.float32)/255.0
mse = np.mean((a-b)**2)
print("Pixel MSE between original and watermarked image:", mse)
'''



#  **********************************code to check if embedding actually happened or not *************************
# ===== Replace these paths =====


'''
ORIG = "unwatermarked/unw_2.jpg"
WM   = "outputs/watermarked/watermarked_19.png"

# ---- Load as PIL images ----
a = Image.open(ORIG).convert("RGB")
b = Image.open(WM).convert("RGB")

print("Loaded images:", a.size, b.size)

# ---- Ensure same size ----
if a.size != b.size:
    print("Images differ in size! Resizing watermarked image…")
    b = b.resize(a.size)

# ---- Compute difference image ----
diff = ImageChops.difference(a, b)
diff.save("debug_diff_original_vs_watermarked.png")
print("Saved: debug_diff_original_vs_watermarked.png")

# ---- Check if any difference exists ----
bbox = diff.getbbox()
if bbox is None:
    print("RESULT: No difference detected. No visible embedding.")
else:
    print("RESULT: Differences detected in bounding box:", bbox)

# ---- Numeric MSE & PSNR ----
a_np = np.array(a, dtype=np.float32)
b_np = np.array(b, dtype=np.float32)

mse = ((a_np - b_np) ** 2).mean()
print(f"MSE = {mse:.10f}")

# PSNR (peak signal-to-noise ratio)
if mse == 0:
    psnr = 100  # treat as perfect
else:
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))

print(f"PSNR = {psnr:.2f} dB")


'''
