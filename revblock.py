import torch.nn as nn
import torch

# RevNet reversible block example
class RevBlock(nn.Module):
    def __init__(self, channels):
        super(RevBlock, self).__init__()
        self.F = nn.Sequential(
            nn.Conv2d(channels // 2, channels // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels // 2, channels // 2, kernel_size=3, padding=1),
        )
        self.G = nn.Sequential(
            nn.Conv2d(channels // 2, channels // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels // 2, channels // 2, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        y1 = x1 + self.F(x2)
        y2 = x2 + self.G(y1)
        y = torch.cat([y1, y2], dim=1)
        return y

    def inverse(self, y):
        y1, y2 = torch.chunk(y, 2, dim=1)
        x2 = y2 - self.G(y1)
        x1 = y1 - self.F(x2)
        x = torch.cat([x1, x2], dim=1)
        return x

