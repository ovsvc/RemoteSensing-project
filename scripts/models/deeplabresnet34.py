import torch
import torch.nn as nn
from pathlib import Path
import segmentation_models_pytorch as smp


class DeepLabResNet34(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()
        
        self.model = smp.DeepLabV3(
            encoder_name="resnet34",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=num_classes
        )

        self.name = "DeepLabV3_ResNet34"

    def forward(self, x, return_water_prob=False):
        logits = self.model(x)
        
        if return_water_prob:
            probs = torch.softmax(logits, dim=1)
            water_probs = probs[:, 1, :, :]  
            return {"out": logits, "water_prob": water_probs}
        
        return {"out": logits}

    def save(self, directory: Path, suffix: str = "best"):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"deeplabv3_resnet34_model_{suffix}.pth"
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")
