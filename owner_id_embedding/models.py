import torch
import torch.nn as nn

class WatermarkEmbedder(nn.Module):
    def __init__(self, wm_channels=1):
        super().__init__()

        # Encode image
        self.img_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )

        # Encode watermark
        self.wm_encoder = nn.Sequential(
            nn.Conv2d(wm_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(96, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()  # output image
        )

    def forward(self, image, watermark):
        img_feat = self.img_encoder(image)
        wm_feat = self.wm_encoder(watermark)

        fused = torch.cat([img_feat, wm_feat], dim=1)
        watermarked_image = self.fusion(fused)
        return watermarked_image

class WatermarkExtractor(nn.Module):
    def __init__(self, wm_channels=1):
        super().__init__()

        self.extractor = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, wm_channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, image):
        return self.extractor(image)
