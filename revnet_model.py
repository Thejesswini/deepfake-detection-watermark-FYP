'''
from revblock import RevBlock
import torch.nn as nn

class RevNet(nn.Module):
    def __init__(self, channels=6):  # 3 for image + 3 for watermark matrix
        super(RevNet, self).__init__()
        self.revblock1 = RevBlock(channels)
        self.revblock2 = RevBlock(channels)

    def forward(self, x):
        x = self.revblock1(x)
        x = self.revblock2(x)
        return x 

    def inverse(self, y):
        y = self.revblock2.inverse(y)
        y = self.revblock1.inverse(y)
        return y
 
class RevNet3(nn.Module):
    def __init__(self, channels=6):  # 3 for image + 3 for watermark matrix
        super(RevNet3, self).__init__()
        self.revblock1 = RevBlock(channels)
        self.revblock2 = RevBlock(channels)
        self.revblock3 = RevBlock(channels)

    def forward(self, x):
        x = self.revblock1(x)
        x = self.revblock2(x)
        x = self.revblock3(x)
        return x 

    def inverse(self, y):
        y = self.revblock3.inverse(y)
        y = self.revblock2.inverse(y)
        y = self.revblock1.inverse(y)
        return y
'''


#new model - gemini 1 ( 64x64 images)
'''
from revblock import RevBlock
import torch.nn as nn
import torch

class RevNet3(nn.Module):
    def __init__(self, channels=6, message_length=256):
        super(RevNet3, self).__init__()
        self.revblock1 = RevBlock(channels)
        self.revblock2 = RevBlock(channels)
        self.revblock3 = RevBlock(channels)
        
        self.message_length = message_length
        # --- NEW: Extractor Head for Decoder ---
        # Assuming H=64, W=64, channels=6
        self.extractor_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(4), # Downsample features to 4x4
            nn.Flatten(),
            nn.Linear(channels * 4 * 4, message_length), # Output shape [B, 256]
            nn.Sigmoid() # Squashes output between 0 and 1 (matches binary message)
        )
        # ----------------------------------------

    def forward(self, x):
        # Forward pass (Encoder) is unchanged
        x = self.revblock1(x)
        x = self.revblock2(x)
        x = self.revblock3(x)
        return x 

    def inverse(self, y):
        # Inverse pass (Decoder)
        y = self.revblock3.inverse(y)
        y = self.revblock2.inverse(y)
        y = self.revblock1.inverse(y)
        
        # 1. Extract the message vector from the full feature tensor 'y'
        extracted_message = self.extractor_head(y)
        
        # 2. Reconstruct the output tensor for the training loop's chunking logic:
        y1, _ = torch.chunk(y, 2, dim=1) # The recovered image part
        zero_tensor = torch.zeros_like(y1).to(y.device) 
        
        # Return the feature map concatenation (for consistency) AND the extracted message
        return torch.cat([y1, zero_tensor], dim=1), extracted_message
'''


# revnet_model.py
from revblock import RevBlock
import torch
import torch.nn as nn

class RevNet3(nn.Module):
    """
    RevNet3 with a proper Extractor Head.
    - channels: total channels of the reversible tensor (e.g. 6 = 3 image + 3 watermark).
    - message_length: length of binary message to extract (e.g. 256).
    Notes:
    - The Extractor reads only the watermark-side channels (assumed to be the second half).
    - The extractor head uses global pooling (AdaptiveAvgPool2d(1)) for spatial-size independence.
    - The head DOES NOT apply sigmoid; use BCEWithLogitsLoss during training.
    - inverse(y) returns (reconstructed_full_tensor, extracted_message_logits).
    """
    def __init__(self, channels=6, message_length=256):
        super(RevNet3, self).__init__()
        if channels % 2 != 0:
            raise ValueError("channels must be even (split into two halves).")
        self.channels = channels
        self.half_channels = channels // 2
        self.message_length = message_length

        # Reversible blocks (same as yours)
        self.revblock1 = RevBlock(channels)
        self.revblock2 = RevBlock(channels)
        self.revblock3 = RevBlock(channels)

        # Extractor head: pool the watermark-side feature maps to 1x1, flatten, linear -> logits
        # We DO NOT apply Sigmoid here. Use BCEWithLogitsLoss outside.
        self.extractor_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                    # [B, half_channels, 1, 1]
            nn.Flatten(),                               # [B, half_channels]
            nn.Linear(self.half_channels, message_length)  # [B, message_length] (logits)
        )

    def forward(self, x):
        """Encoder / forward pass: apply reversible blocks in order."""
        x = self.revblock1(x)
        x = self.revblock2(x)
        x = self.revblock3(x)
        return x

    def inverse(self, y):
        """
        Decoder / inverse pass:
        - Run inverse revblocks
        - Extract message logits from the watermark half of the feature tensor
        - Return reconstructed full tensor (same shape as input) AND extracted message logits
        """
        # run inverse blocks in reverse order
        y = self.revblock3.inverse(y)
        y = self.revblock2.inverse(y)
        y = self.revblock1.inverse(y)

        # y shape: [B, channels, H, W]
        # watermark-side channels are assumed to be the *second half*.
        # Extract only the watermark-side features for the extractor head.
        watermark_features = y[:, self.half_channels:, :, :]  # [B, half_channels, H, W]

        # Extract message logits
        extracted_message_logits = self.extractor_head(watermark_features)  # [B, message_length]

        # Return the reconstructed full tensor AND the message logits
        return y, extracted_message_logits
