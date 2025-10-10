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