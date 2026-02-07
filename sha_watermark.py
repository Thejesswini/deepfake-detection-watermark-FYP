import hashlib
import numpy as np
import torch

def generate_sha_matrix(batch_size, height, width, image_id="default"):
    """
    Generate SHA-256 based watermark matrix.
    """

    # Step 1: Compute SHA-256 hash
    hash_obj = hashlib.sha256(image_id.encode())
    hash_hex = hash_obj.hexdigest()

    # Step 2: Convert hex → binary bits
    hash_bits = bin(int(hash_hex, 16))[2:].zfill(256)

    # Step 3: Resize bits to image size
    total_pixels = height * width
    bits_array = np.array(list(hash_bits), dtype=np.uint8)
    bits_array = np.resize(bits_array, total_pixels)

    # Step 4: Reshape into image
    sha_matrix = bits_array.reshape(height, width).astype(np.float32)

    # Normalize
    sha_matrix = sha_matrix / sha_matrix.max()

    # Convert to tensor [B,1,H,W] → [B,3,H,W]
    sha_tensor = torch.tensor(sha_matrix).unsqueeze(0).unsqueeze(0)
    sha_tensor = sha_tensor.repeat(batch_size, 3, 1, 1)

    return sha_tensor
