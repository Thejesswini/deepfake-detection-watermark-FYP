import torch.nn as nn
import torch
import torch.nn.functional as F

# # RevNet reversible block example
# class RevBlock(nn.Module):
#     def __init__(self, channels):
#         super(RevBlock, self).__init__()
#         self.F = nn.Sequential(
#             nn.Conv2d(channels // 2, channels // 2, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(channels // 2, channels // 2, kernel_size=3, padding=1),
#         )
#         self.G = nn.Sequential(
#             nn.Conv2d(channels // 2, channels // 2, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(channels // 2, channels // 2, kernel_size=3, padding=1),
#         )

#     def forward(self, x):
#         x1, x2 = torch.chunk(x, 2, dim=1)
#         y1 = x1 + self.F(x2)
#         y2 = x2 + self.G(y1)
#         y = torch.cat([y1, y2], dim=1)
#         return y

#     def inverse(self, y):
#         y1, y2 = torch.chunk(y, 2, dim=1)
#         x2 = y2 - self.G(y1)
#         x1 = y1 - self.F(x2)
#         x = torch.cat([x1, x2], dim=1)
#         return x

class Invertible1x1Conv(nn.Module):
    def __init__(self, channels):
        super(Invertible1x1Conv, self).__init__()
        # Initialize with a random orthogonal matrix for stability
        w_init = torch.linalg.qr(torch.randn(channels, channels))[0]
        self.weight = nn.Parameter(w_init)
    
    def forward(self, x):
        # x: [Batch, Channels, H, W]
        # Use conv2d with 1x1 kernel to perform matrix multiplication on every pixel
        return F.conv2d(x, self.weight.unsqueeze(2).unsqueeze(3))
    
    def inverse(self, y):
        # Calculate the mathematical inverse of the weight matrix
        inv_weight = torch.inverse(self.weight)
        return F.conv2d(y, inv_weight.unsqueeze(2).unsqueeze(3))

class RevBlock(nn.Module):
    def __init__(self, channels):
        super(RevBlock, self).__init__()
        in_channels = channels // 2 # 3
        out_channels = channels // 2 # 3
        
        # EXPANSION: Give the model working memory
        hidden_channels = 64

        self.F = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels), # Helps stability
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
        )
        self.G = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        
        # Residual connections ensure reversibility
        y1 = x1 + self.F(x2)
        y2 = x2 + self.G(y1)
        
        y = torch.cat([y1, y2], dim=1)
        return y

    def inverse(self, y):
        y1, y2 = torch.chunk(y, 2, dim=1)
        
        # Reverse the math exactly
        x2 = y2 - self.G(y1)
        x1 = y1 - self.F(x2)
        
        x = torch.cat([x1, x2], dim=1)
        return x
    
class RevBlockTanh(nn.Module):
    def __init__(self, channels):
        super(RevBlockTanh, self).__init__()
        in_channels = channels // 2 # 3
        out_channels = channels // 2 # 3
        
        # EXPANSION: Give the model working memory
        hidden_channels = 64
        self.scale = 10.0 / 255.0

        self.F = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels), # Helps stability
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        self.G = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        
        # Residual connections ensure reversibility
        y1 = x1 + (self.F(x2) * self.scale) 
        y2 = x2 + (self.G(y1) * self.scale)
        
        y = torch.cat([y1, y2], dim=1)
        return y

    def inverse(self, y):
        y1, y2 = torch.chunk(y, 2, dim=1)
        
        # Reverse the math exactly
        x2 = y2 - (self.G(y1) * self.scale)
        x1 = y1 - (self.F(x2) * self.scale)
        
        x = torch.cat([x1, x2], dim=1)
        return x