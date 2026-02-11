import torch
import numpy as np
from config import BATCH_SIZE
import torch.nn.functional as F

def owner_id_to_watermark(owner_id: str, h=32, w=32, image_h=218, image_w=178, batch_size=BATCH_SIZE):
    """
    Converts owner ID string into a binary watermark.
    """
    bits_required = h * w

    byte_data = owner_id.encode("utf-8")
    bit_array = np.unpackbits(np.frombuffer(byte_data, dtype=np.uint8))

    # Pad or truncate
    if len(bit_array) < bits_required:
        bit_array = np.pad(bit_array, (0, bits_required - len(bit_array)))
    else:
        bit_array = bit_array[:bits_required]

    wm = torch.tensor(bit_array, dtype=torch.float32)
    wm = wm.view(1, 1, h, w)
    wm = F.interpolate(
        wm,
        size=(image_h, image_w),
        mode="bilinear",
        align_corners=False
    )
    
    wm = wm.repeat(batch_size, 1, 1, 1)
    
    return wm
