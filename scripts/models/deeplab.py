import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights
from pathlib import Path


class DeepLabMobileNet(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()

        weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self.model = deeplabv3_mobilenet_v3_large(weights=weights)

        self.model.classifier[4] = nn.Conv2d(
            in_channels=256,
            out_channels=num_classes,
            kernel_size=1
        )

        self.name = "DeepLab_MobileNetV3"

    def forward(self, x):
        return self.model(x)["out"]

    def save(self, directory: Path, suffix: str = "best"):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"deeplab_model_{suffix}.pth"
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")