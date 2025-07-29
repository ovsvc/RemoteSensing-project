import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
from pathlib import Path


class DeepLabResNet(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()

        weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        self.model = deeplabv3_resnet50(weights=weights)

        # Replace the classifier for custom number of classes
        self.model.classifier[4] = nn.Conv2d(
            in_channels=256,  # fixed in DeepLab classifier head
            out_channels=num_classes,
            kernel_size=1
        )

        self.name = "DeepLab_ResNet50"

    def forward(self, x, return_water_prob=False):
        output = self.model(x)  # OrderedDict with key "out"
        logits = output["out"]  # this is the actual Tensor
        
        if return_water_prob:
            probs = torch.softmax(logits, dim=1)
            water_probs = probs[:, 1, :, :]
            return {"out": logits, "water_prob": water_probs}
        
        return {"out": logits}

    def save(self, directory: Path, suffix: str = "best"):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"deeplab_resnet_model_{suffix}.pth"
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")
