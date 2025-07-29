import os, sys 
import re
import torch
import rasterio
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as tf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json
from typing import List, Tuple, Callable, Optional
import matplotlib.colors as mcolors


def create_rgb_scaled(rgb_bands):
    scale = rgb_bands.max() if rgb_bands.max() > 1 else 1.0
    rgb_scaled = np.clip(rgb_bands.astype(np.float32) / scale, 0, 1)
    return np.moveaxis(rgb_scaled, 0, -1)


def plot_sample(image_tensor, mask_tensor, class_colors=None):
    """
    Visualize a sample image and its segmentation mask.

    Args:
        image_tensor (torch.Tensor or np.ndarray): (3, H, W) if tensor or (H, W, 3) if ndarray
        mask_tensor (torch.Tensor or np.ndarray): (H, W)
        class_colors (list): Optional custom colormap
    """

    # Convert image to (H, W, 3) NumPy array
    if isinstance(image_tensor, torch.Tensor):
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    else:
        image_np = image_tensor

    # Convert mask to (H, W) NumPy array
    if isinstance(mask_tensor, torch.Tensor):
        mask_np = mask_tensor.cpu().numpy()
    else:
        mask_np = mask_tensor

    if class_colors is None:
        class_colors = [
            "#c8c8c8",  # 0 - Background
            "blue",     # 1 - Water
            "red",      # 2 - Cloud
            "green"     # 3 - Ice/Snow
        ]

    cmap = mcolors.ListedColormap(class_colors)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(image_np)
    ax[0].set_title("Image")
    ax[0].axis("off")

    ax[1].imshow(mask_np, cmap=cmap, vmin=0, vmax=len(class_colors)-1)
    ax[1].set_title("Segmentation Mask")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()


class FloodDataset(Dataset):

    def __init__(self,
                split_file: str,
                transform: Optional[Callable] = None,
                subset_size: Optional[int] = None):
      """
      Dataset for loading satellite images and segmentation masks.

      Args:
          split_file (str): Path to JSON file containing (image_path, mask_path) pairs.
          transform (callable, optional): Transformations to apply to image and mask.
          subset_size (int, optional): If set, limits dataset to first N samples.
      """
      with open(split_file, "r") as f:
          self.pairs = json.load(f)

      if subset_size:
          self.pairs = self.pairs[:subset_size]

      self.transform = transform
      self.class_names = {
        0: "Background",
        1: "Water",
        2: "Cloud",
        3: "Ice/Snow"}
      
      self.num_classes = 4

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        
        image_path, mask_path = self.pairs[idx]

        with rasterio.open(image_path) as src:
            image = src.read([1, 2, 3])  # (3, H, W)
            image = create_rgb_scaled(image)


        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype("int64")

        # Albumentations expects HWC, so make sure image is (H, W, C)
        image = image.astype(np.float32)  # Ensure float32 for Albumentations
    
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].long()

        return image, mask
     
    def get_num_classes(self):
        return self.num_classes

    def get_class_names(self):
        return self.class_names

    def calculate_water_pixel_percentage(self, water_class_id=1) -> float:
        """
        Calculates the percentage of water pixels (class_id=1) in the dataset.

        Args:
            water_class_id (int): Class index for water (default is 1)

        Returns:
            float: Percentage of water pixels across all masks.
        """
        total_pixels = 0
        water_pixels = 0

        print("Scanning masks for water pixels...")

        for i in range(len(self)):
            _, mask = self[i]

            if isinstance(mask, torch.Tensor):
                mask_np = mask.cpu().numpy()
            else:
                mask_np = mask

            mask_np = np.asarray(mask_np).squeeze()
            total_pixels += mask_np.size
            water_pixels += np.sum(mask_np == water_class_id)

        if total_pixels == 0:
            return 0.0

        percentage = (water_pixels / total_pixels) * 100
        return round(percentage, 3)

