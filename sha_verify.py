import torch
import numpy as np
from sha_watermark import generate_sha_matrix

def verify_sha(recovered_watermark, batch_size, height, width, image_id="default"):
    expected_sha = generate_sha_matrix(batch_size, height, width, image_id).to(recovered_watermark.device)

    mse = torch.mean((recovered_watermark - expected_sha)**2).item()
    return mse
