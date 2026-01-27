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


folder_path = r"./only_watermarked"

# Get all png files
files = [f for f in os.listdir(folder_path) if f.lower().endswith(".png")]

# Function to extract number from filename
def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else float('inf')

# Sort files by extracted number
files.sort(key=extract_number)

# Rename files
for idx, filename in enumerate(files, start=1):
    new_name = f"watermarked_{idx:04d}.png"  # 0001.png format
    old_path = os.path.join(folder_path, filename)
    new_path = os.path.join(folder_path, new_name)
    os.rename(old_path, new_path)

print("Renaming completed.")
