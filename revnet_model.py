from revblock import RevBlock, Invertible1x1Conv, RevBlockTanh
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
    

class RevNet3_with_ChannelMixing(nn.Module):
    def __init__(self, channels=6):  # 3 Image + 3 Watermark
        super().__init__()
        
        # 1. Mix channels immediately so image and watermark are inseparable
        self.mix1 = Invertible1x1Conv(channels)
        self.revblock1 = RevBlock(channels)
        
        # 2. Mix again between blocks (Optional but recommended)
        self.mix2 = Invertible1x1Conv(channels)
        self.revblock2 = RevBlock(channels)

    def forward(self, x):
        x = self.mix1(x)
        x = self.revblock1(x)
        
        x = self.mix2(x)
        x = self.revblock2(x)
        return x 

    def inverse(self, y):
        # Exact reverse order
        y = self.revblock2.inverse(y)
        y = self.mix2.inverse(y)
        
        y = self.revblock1.inverse(y)
        y = self.mix1.inverse(y)
        return y
    
class RevNet3_with_ChannelMixing_Tanh(nn.Module):
    def __init__(self, channels=6):  # 3 Image + 3 Watermark
        super().__init__()
        
        # 1. Mix channels immediately so image and watermark are inseparable
        self.mix1 = Invertible1x1Conv(channels)
        self.revblock1 = RevBlockTanh(channels)
        
        # 2. Mix again between blocks (Optional but recommended)
        self.mix2 = Invertible1x1Conv(channels)
        self.revblock2 = RevBlockTanh(channels)

    def forward(self, x):
        x = self.mix1(x)
        x = self.revblock1(x)
        
        x = self.mix2(x)
        x = self.revblock2(x)
        return x 

    def inverse(self, y):
        # Exact reverse order
        y = self.revblock2.inverse(y)
        y = self.mix2.inverse(y)
        
        y = self.revblock1.inverse(y)
        y = self.mix1.inverse(y)
        return y
    
class RevNet3_with_ChannelMixing_Tanh_Deep(nn.Module):
    def __init__(self, channels=6):  # 3 Image + 3 Watermark
        super().__init__()
        
        # 1. Mix channels immediately so image and watermark are inseparable
        self.mix1 = Invertible1x1Conv(channels)
        self.revblock1 = RevBlockTanh(channels)
        self.revblock2 = RevBlockTanh(channels)
        self.revblock3 = RevBlockTanh(channels)
        
        # 2. Mix again between blocks (Optional but recommended)
        self.mix2 = Invertible1x1Conv(channels)
        self.revblock4 = RevBlockTanh(channels)
        

    def forward(self, x):
        x = self.mix1(x)
        x = self.revblock1(x)
        x = self.revblock2(x)
        x = self.revblock3(x)
        
        x = self.mix2(x)
        x = self.revblock4(x)
        return x 

    def inverse(self, y):
        # Exact reverse order
        y = self.revblock4.inverse(y)
        y = self.mix2.inverse(y)
        
        y = self.revblock3.inverse(y)
        y = self.revblock2.inverse(y)
        y = self.revblock1.inverse(y)
        y = self.mix1.inverse(y)
        return y