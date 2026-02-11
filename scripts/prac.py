import os
import re

deepfaked = [1,4, 5, 8, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 24, 31,32,34,37,38,42,39,45,46,48,47,49,51,53,56,55,58,60,41,61,62,63,64,65,66]
#deepfaked.sort()
deepfaked = ["_"+str(i)+"." for i in deepfaked]
print(len(deepfaked))
ct=0
# files_to_copy = []
# for fname in os.listdir("./outputs/watermarked"):
#     if any(s in fname for s in deepfaked):
#         files_to_copy.append(fname)
#         ct+=1
# print(ct)


from revnet_model import RevNet3
from watermark_generation import generate_watermark_matrix
import torch
import torch.nn.functional as F
from torchvision import utils
from torchvision.io import read_image

path1 = r"C:\Users\theju\Downloads\varied_bg.jpg"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img1 = read_image(path1).float() / 255.0
img1 = img1.unsqueeze(0)
img1.to(DEVICE)
    
model = RevNet3(channels=6).to(DEVICE)
model.load_state_dict(torch.load("model_weights.pth", map_location=DEVICE))
model.eval()

og_peano_curve = generate_watermark_matrix(batch_size=1, height=218, width=178)
input_tensor = torch.cat([img1, og_peano_curve], dim=1)
print(input_tensor.shape)

with torch.no_grad():
    embedded = model(input_tensor)
    embedded_image, embedded_watermark_part = torch.chunk(embedded, 2, dim=1)
utils.save_image(embedded_image, "varied_bg.jpg")
print('done')