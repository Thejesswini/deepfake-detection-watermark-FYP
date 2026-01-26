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

def robust_peano_matrix(peano_matrix):
    # 2. Define how many distinct "levels" you want. 
    # 16 is a safe number (4 bits of info per pixel).
    # This creates a "staircase" effect.
    num_levels = 16 
    
    # 3. Quantize the Peano curve into these levels
    # Normalize to 0-1 first
    normalized = peano_matrix / peano_matrix.max()
    
    # Bucketize into discrete integer levels (0, 1, 2 ... 15)
    #stepped = np.floor(normalized * num_levels)
    
    # 4. Scale to 0-1 range for the Tensor
    # (The model likes 0.0 - 1.0 floats)
    # The actual values will be: 0.0, 0.066, 0.133... 1.0
    #final_matrix = stepped / (num_levels - 1)
    
    #return final_matrix
    return normalized

def generate_watermark_matrix(batch_size, height, width, order=3):
    peano_matrix = peano_curve_generator(height, width, order)
    #peano_norm = peano_matrix.astype(np.float32) / peano_matrix.max()
    peano_fin = robust_peano_matrix(peano_matrix)
    
    peano_tensor = torch.tensor(peano_fin, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [B,1,H,W]
    peano_tensor_3c = peano_tensor.repeat(1, 3, 1, 1)  # [B,3,H,W]
    return peano_tensor_3c

if __name__=='__main__':
    import matplotlib.pyplot as plt
    
    pc = robust_peano_matrix(peano_curve_generator(20,20,3))
    # pc=pc.astype(np.float32) / pc.max()
    # print(pc*255)
    
    plt.imshow(pc, cmap='gray')
    plt.show()