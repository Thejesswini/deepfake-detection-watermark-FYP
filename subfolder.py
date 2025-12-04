import os
import random
import shutil
import zipfile

# --- CONFIGURATION (Match your existing paths and settings) ---

# The full, large directory where ALL CelebA images reside
FULL_IMAGE_DIR = '/Users/sem5/sem7/deepfake/celebA/img_align_celeba/img_align_celeba' 

# The name of the new, small folder to hold the 1000 images
SUBSET_FOLDER_NAME = 'celeba_subset_1000'

# The path where the new folder and the final ZIP file will be created
OUTPUT_BASE_PATH = '/Users/sem5/sem7/deepfake/celebA/' 

TOTAL_IMAGES_NEEDED = 1000
# ----------------------------------------------------------------

# 1. Prepare paths
full_source_path = FULL_IMAGE_DIR
new_subset_path = os.path.join(OUTPUT_BASE_PATH, SUBSET_FOLDER_NAME)
zip_output_path = os.path.join(OUTPUT_BASE_PATH, SUBSET_FOLDER_NAME + '.zip')

print(f"Starting subset creation for {TOTAL_IMAGES_NEEDED} images...")

# 2. List and shuffle all available image files
all_images = [f for f in os.listdir(full_source_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
# Use the same seed (42) as your DataLoader prep to ensure the subset is consistent
random.seed(42) 
random.shuffle(all_images)

# 3. Select the required number of files
files_to_copy = all_images[:TOTAL_IMAGES_NEEDED]

if len(files_to_copy) < TOTAL_IMAGES_NEEDED:
    print(f"Warning: Only found {len(files_to_copy)} images.")

# 4. Create the new subset directory
if os.path.exists(new_subset_path):
    print(f"Removing existing directory: {new_subset_path}")
    shutil.rmtree(new_subset_path)

os.makedirs(new_subset_path, exist_ok=True)
print(f"Created new subset directory: {new_subset_path}")

# 5. Copy the files
print("Copying images...")
for filename in files_to_copy:
    source_file = os.path.join(full_source_path, filename)
    destination_file = os.path.join(new_subset_path, filename)
    shutil.copy2(source_file, destination_file)

print(f"Successfully copied {len(files_to_copy)} images.")

# 6. Create the final ZIP archive
print(f"Creating ZIP file at: {zip_output_path}")

try:
    with zipfile.ZipFile(zip_output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(new_subset_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Ensure the ZIP structure starts inside the subset folder
                zipf.write(file_path, os.path.relpath(file_path, OUTPUT_BASE_PATH))
    print(f"✅ Subset creation and zipping complete! File is ready for upload: {zip_output_path}")

except Exception as e:
    print(f"An error occurred during zipping: {e}")