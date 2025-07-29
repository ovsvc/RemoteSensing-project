import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from pathlib import Path

class UnetMobileNetV2(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()

        self.model = smp.Unet(
            encoder_name="mobilenet_v2",   
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=num_classes
        )

        self.name = "UNet_MobileNetV2"

    def forward(self, x, return_water_prob=False):
        logits = self.model(x)
        
        if return_water_prob:
            probs = torch.softmax(logits, dim=1)
            water_probs = probs[:, 1, :, :]  
            return {"out": logits, "water_prob": water_probs}
    
        return {"out": logits}


    def save(self, directory: Path, suffix: str = "best"):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"_unet_mobilenetv2_{suffix}.pth"
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")