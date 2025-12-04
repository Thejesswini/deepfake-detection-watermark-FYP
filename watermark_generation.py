'''
import math
import numpy as np
import torch

def hilbert_curve_generator(order): #previously convert_indices_to_int

    """
    Converts a 2D coordinate (x, y) to its Hilbert curve distance 'd'.
    This is a self-contained implementation of the algorithm.

    Args:
        x (int): The x-coordinate.
        y (int): The y-coordinate.
        order (int): The order of the Hilbert curve.

    Returns:
        int: The Hilbert curve distance (index).
    """
    # apparently generating indices doesnt need the [x, y] coordinates generated in the prev cell. lol
    side_length = 2**order
    hilbert_matrix = np.zeros((side_length, side_length), dtype=int)
    
    def xy_to_d(x,y,order):
        d = 0
        # Iterate through each level of the curve's order
        for s_val in range(order - 1, -1, -1):
            side_len_sub_quad = 2**s_val
            # Determine the quadrant for the current level
            rx = (x >> s_val) & 1
            ry = (y >> s_val) & 1

            # Add the contribution of the quadrant to the total distance
            d += side_len_sub_quad * side_len_sub_quad * ((3 * rx) ^ ry)

            # Rotate/transform the coordinates for the next iteration
            if ry == 0:
                if rx == 1:
                    x = side_len_sub_quad - 1 - x
                    y = side_len_sub_quad - 1 - y
                x, y = y, x

        return d

        # Iterate through each coordinate (x, y) in the grid
    for y in range(side_length):
        for x in range(side_length):
            # Calculate the Hilbert index for the coordinate (x, y)
            distance = xy_to_d(x, y, order)
            # Assign this index to the matrix at the correct cell
            hilbert_matrix[y, x] = distance
    return hilbert_matrix


# Space-filling curve watermark generator placeholder
def generate_watermark_matrix(batch_size, side):
    hilbert_matrix = hilbert_curve_generator(int(math.log2(side)))

    # Normalize 
    hilbert_norm = hilbert_matrix.astype(np.float32) / (64*64 - 1)
    
    hilbert_tensor = torch.tensor(hilbert_norm).unsqueeze(0).repeat(batch_size, 1, 1, 1)  # shape: [batch, 1, 64, 64]
    # 3 channels (like RGB)
    hilbert_tensor_3c = hilbert_tensor.repeat(1, 3, 1, 1)  # shape: [batch, 3, 64, 64]
    return hilbert_tensor_3c


'''

import numpy as np
import torch
import random

# --- (Original peano_curve_generator function is RETAINED for localization) ---

def peano_curve_generator(height, width, order=3):
    mat = np.zeros((height, width), dtype=np.int64)
    counter = [0]

    def fill(x0, y0, h, w, depth, orientation=0):
        if depth == 0:
            for i in range(h):
                for j in range(w):
                    mat[y0 + i, x0 + j] = counter[0]
                    counter[0] += 1
            return

        if orientation == 0:  # vertical split
            step = max(1, h // 3)
            fill(x0, y0, step, w, depth - 1, 1)
            fill(x0, y0 + step, step, w, depth - 1, 1)
            fill(x0, y0 + 2*step, h - 2*step, w, depth - 1, 1)
        else:  # horizontal split
            step = max(1, w // 3)
            fill(x0, y0, h, step, depth - 1, 0)
            fill(x0 + step, y0, h, step, depth - 1, 0)
            fill(x0 + 2*step, y0, h, w - 2*step, depth - 1, 0)

    fill(0, 0, height, width, order)
    return mat

# --- MODIFIED generate_watermark_matrix ---

def generate_watermark_matrix(batch_size, height, width, order=6, message_length=256):
    """
    Generates a Peano-ordered Binary Watermark.
    Uses Peano curve for spatial arrangement (localization) but assigns 
    simple 0.0 or 1.0 values (binary encoding) for robustness.
    """
    
    # 1. Generate the Peano curve matrix (The Map)
    # This matrix gives the spatial ordering (0 to H*W-1)
    peano_matrix = peano_curve_generator(height, width, order=order)
    
    # Flatten the matrix to get the sequential Peano-ordered index list
    peano_order = peano_matrix.flatten()
    
    # 2. Identify the embedding locations based on the Peano curve's path
    # We find the original (h, w) coordinates for the first 'message_length'
    # indices in the Peano path.
    # np.argsort finds the indices that would sort the array (i.e., where each number is located)
    peano_locations_flat_index = np.argsort(peano_order)[:message_length]
    
    # Convert the 1D index back to 2D (h, w) coordinates
    h_coords = peano_locations_flat_index // width
    w_coords = peano_locations_flat_index % width

    # 3. Create the Random Binary Message Signal
    # The actual data we want to embed/extract. Values are 0.0 or 1.0.
    # Shape: [B, message_length]
    binary_message = torch.randint(0, 2, (batch_size, message_length)).float()
    
    # 4. Initialize the final Watermark Tensor as zeros
    # Shape: [B, 1, H, W]
    W_1c = torch.zeros(batch_size, 1, height, width, dtype=torch.float32)
    
    # 5. Embed the Binary Message using the Peano Locations
    # This places the robust binary signal at the complex Peano locations
    for b in range(batch_size):
        for i in range(message_length):
            # Assign the binary message bit (0.0 or 1.0) to the specific location
            W_1c[b, 0, h_coords[i], w_coords[i]] = binary_message[b, i]

    # 6. Repeat across 3 channels to match RGB input
    W_3c = W_1c.repeat(1, 3, 1, 1)  # [B, 3, H, W]
    
    return W_3c,binary_message