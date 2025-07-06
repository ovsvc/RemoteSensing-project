import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.plot import show
from rasterio.enums import Resampling
import os
import tarfile
from math import sqrt
import math


def read_dataset_tar(tar_path):
    """
    Extracts a specific TIFF file from a TAR archive and opens it using rasterio.
    """

    bands = {}

    with tarfile.open(tar_path, 'r') as tar:


        for member in tar.getmembers():
            if member.name.endswith('.tif'):
              tar.extract(member, path="/tmp/")
              extracted_tif_path = os.path.join("/tmp/", member.name)
              with rasterio.open(extracted_tif_path) as dataset:
                # print(dataset.meta)
                if member.name == 'optical_bands.tif':
                  bands['rgb_bands'] = dataset.read([3,2,1])
                  bands['green_band'] = dataset.read(2)

                  bands['nir_band'] = dataset.read(4)
                  bands['rgb_nir']=dataset.read([3,2,1,4])
                  bands['rgb_nir_profile']=dataset.profile

                else:
                  bands[member.name[:-4]] = dataset.read(1)
        return bands
    

def create_rgb(rgb_bands):
  """
    Creates RGB image
  """
  rgb_scaled = np.clip(rgb_bands.astype(np.float32) / 4000, 0, 1)
  rgb_scaled = np.moveaxis(rgb_scaled, 0, -1)  # Rearrange to (H, W, C)
  return rgb_scaled


def create_ndwi(green_band, nir_band):
  """
    Creates NDWI index map
  """
  ndwi = (green_band - nir_band) / (green_band + nir_band +1e-10)
  return np.clip(ndwi, -1, 1)


def create_ndwi_mask(ndwi):
    """
    Create an RGB mask for NDWI values:
    0 - Dry/Other         -> light gray
    1 - Flooding/Humidity -> cyan
    2 - Water Surface      -> blue
    """

    # Step 1: Classification
    mask = np.zeros_like(ndwi, dtype=np.uint8)
    mask[(ndwi >= 0.1) & (ndwi < 0.2)] = 1  # Flooding / Humidity
    mask[ndwi >= 0.2] = 2                  # Water Surface

    # Step 2: Define RGB colors for each class
    NDWI_COLORS = {
        0: (200, 200, 200),  # Dry / Other (light gray)
        1: (0, 255, 255),    # Flooding / Humidity (cyan)
        2: (255, 105, 180),  # Water Surface (pink)
    }

    # Step 3: Create RGB mask
    mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for value, color in NDWI_COLORS.items():
        mask_rgb[mask == value] = color

    return mask_rgb


'''def create_scl_masks(scl_band):
    mask = np.zeros_like(scl_band, dtype=np.uint8)
    mask[np.isin(scl_band, WATER_VALUES)] = 2  # Water class

    mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for value, color in MASK_COLORS.items():
      mask_rgb[mask == value] = color

    return mask_rgb
'''


def create_scl_masks(scl_band, WATER_VALUES, ICE_VALUES, MASK_COLORS):
    mask = np.zeros_like(scl_band, dtype=np.uint8)

    # Class assignments
    mask[np.isin(scl_band, WATER_VALUES)] = 2     # Water
    mask[np.isin(scl_band, ICE_VALUES)] = 3       # Snow/Ice



    # Create RGB mask
    mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for value, color in MASK_COLORS.items():
        mask_rgb[mask == value] = color

    return mask_rgb


def create_cld_masks(cld_band, threshold,rgs_scaled): # Threshold choosen to be 20
  cloud_mask = (cld_band > threshold).astype(int)
  overlay = np.zeros_like(rgs_scaled)
  overlay[cloud_mask == 1] = [255, 0, 0]
  return overlay


def create_combined_mask(scl_band, cld_band, cld_threshold, rgb_scaled, WATER_VALUES, ICE_VALUES, MASK_COLORS):
    """
    Create a combined mask with clouds and water. Cloud masks take precedence over water masks.

    Args:
        scl_band (numpy.ndarray): Scene Classification Layer (SCL) band.
        cld_band (numpy.ndarray): Cloud probability band.
        cld_threshold (int): Threshold for cloud detection (e.g., 20).
        rgb_scaled (numpy.ndarray): Scaled RGB image for overlay.

    Returns:
        numpy.ndarray: Combined mask image with clouds (red) and water (blue).
    """
    # Create water mask from SCL band
    scl_mask = create_scl_masks(scl_band, WATER_VALUES, ICE_VALUES, MASK_COLORS)
    water_mask = scl_mask[..., 2] 
    ice_mask = scl_mask[..., 1]  

    # Create cloud mask from cloud probability band
    cloud_mask = create_cld_masks(cld_band, cld_threshold, rgb_scaled)
    cloud_mask = cloud_mask[..., 0] 

    # Create combined mask
    background_color=(200, 200, 200)
    combined_mask = np.full_like(rgb_scaled, background_color, dtype=np.uint8)

    # Apply water mask (blue) only where there is no cloud mask
    combined_mask[(water_mask == 255) & (cloud_mask != 255)] = [0, 0, 255]  # Blue for water

    combined_mask[
        (ice_mask == 255) & (cloud_mask != 255) & (water_mask != 255)
    ] = [0, 255, 0] # Green for ice/snow



    # Apply cloud mask (red) - takes precedence over water mask
    combined_mask[cloud_mask == 255] = [255, 0, 0]  # Red for clouds

    return combined_mask


def create_combined_mask_with_ndwi(scl_band, cld_band, cld_threshold, rgb_scaled, ndwi_mask_rgb, WATER_VALUES, ICE_VALUES, MASK_COLORS, background_color=(200, 200, 200)):
    """
    Create a combined mask using SCL (water), cloud mask, and NDWI.
    Cloud masks take precedence. NDWI is used for unclassified background areas.

    Args:
        scl_band (numpy.ndarray): Scene Classification Layer (SCL) band.
        cld_band (numpy.ndarray): Cloud probability band.
        cld_threshold (int): Threshold for cloud detection (e.g., 20).
        rgb_scaled (numpy.ndarray): Scaled RGB image for overlay.
        ndwi_mask_rgb (numpy.ndarray): NDWI RGB mask (with water/flood colors).
        background_color (tuple): RGB background color.

    Returns:
        numpy.ndarray: Combined RGB mask.
    """
    # Create water mask from SCL band
    scl_mask = create_scl_masks(scl_band, WATER_VALUES, ICE_VALUES, MASK_COLORS)
    water_mask = scl_mask[..., 2]  # Extract water class (blue channel)
    ice_mask = scl_mask[..., 1]  # Extract ice class (green channel)


    # Create cloud mask
    cloud_mask = create_cld_masks(cld_band, cld_threshold, rgb_scaled)
    cloud_mask = cloud_mask[..., 0]  # Red channel

    # Start with background color
    combined_mask = np.full_like(rgb_scaled, background_color, dtype=np.uint8)

    # Apply water (blue) where there’s no cloud
    combined_mask[(water_mask == 255) & (cloud_mask != 255)] = [0, 0, 255]  # Blue

    combined_mask[
        (ice_mask == 255) & (cloud_mask != 255) & (water_mask != 255)
    ] = [0, 255, 0] # Green for ice/snow

    # Apply clouds (red) – take precedence
    combined_mask[cloud_mask == 255] = [255, 0, 0]  # Red

    # Apply NDWI mask where current pixel is still background
    bg_mask = (combined_mask == background_color).all(axis=-1)  # Boolean mask where it's still background
    combined_mask[bg_mask] = ndwi_mask_rgb[bg_mask]  # Use NDWI color at those positions

    return combined_mask


def final_mask(scl_band, cld_band, cld_threshold, rgb_scaled, ndwi_mask_rgb, WATER_VALUES, ICE_VALUES, MASK_COLORS, background_color=(200, 200, 200)):
    """
    Create a combined mask using SCL (water), cloud mask, and NDWI.
    Cloud masks take precedence. NDWI is used for unclassified background areas.
    NDWI regions are colored blue, and green areas (ice/snow) are removed.

    Args:
        scl_band (numpy.ndarray): Scene Classification Layer (SCL) band.
        cld_band (numpy.ndarray): Cloud probability band.
        cld_threshold (int): Threshold for cloud detection (e.g., 20).
        rgb_scaled (numpy.ndarray): Scaled RGB image for overlay.
        ndwi_mask_rgb (numpy.ndarray): NDWI RGB mask (with water/flood colors).
        background_color (tuple): RGB background color.

    Returns:
        numpy.ndarray: Combined RGB mask.
    """
    # Create water mask from SCL band
    scl_mask = create_scl_masks(scl_band, WATER_VALUES, ICE_VALUES, MASK_COLORS)
    water_mask = scl_mask[..., 2]  # Extract water class (blue channel)
    ice_mask = scl_mask[..., 1]  # Extract ice class (green channel)


    # Create cloud mask
    cloud_mask = create_cld_masks(cld_band, cld_threshold, rgb_scaled)
    cloud_mask = cloud_mask[..., 0]  # Red channel

    # Start with background color
    combined_mask = np.full_like(rgb_scaled, background_color, dtype=np.uint8)

    # Apply water (blue) where there’s no cloud
    combined_mask[(water_mask == 255) & (cloud_mask != 255)] = [0, 0, 255]  # Blue

    combined_mask[
        (ice_mask == 255) & (cloud_mask != 255) & (water_mask != 255)
    ] = [0, 255, 0] # Green for ice/snow

    # Apply clouds (red) – take precedence
    combined_mask[cloud_mask == 255] = [255, 0, 0]  # Red

    # Apply NDWI mask where current pixel is still background
    bg_mask = (combined_mask == background_color).all(axis=-1)  # Boolean mask where it's still background
    combined_mask[bg_mask] = ndwi_mask_rgb[bg_mask]  # Use NDWI color at those positions

    # Recolor green areas
    green_mask = (combined_mask == [0, 255, 0]).all(axis=-1)
    combined_mask[green_mask] = background_color

    # Recolor ndwi areas into blue
    pink_mask = (combined_mask == [255, 105, 180]).all(axis=-1)
    cyan_mask = (combined_mask == [0, 255, 255]).all(axis=-1)
    
    combined_mask[pink_mask] = [0, 0, 255]
    combined_mask[cyan_mask] = background_color


    return combined_mask


def final_label_mask(scl_band: np.ndarray,
                     cld_band: np.ndarray,
                     cld_threshold: int,
                     ndwi_bool: np.ndarray, WATER_VALUES, ICE_VALUES, MASK_COLORS) -> np.ndarray:
    """
    Build a 2D uint8 label mask with these classes:
      0 = background
      1 = water  (SCL water & not cloud)
      2 = ice/snow (SCL ice & not cloud or water)
      3 = cloud   (cld_band > threshold)
      4 = NDWI-only water (ndwi_bool & not any of the above)
    """
    # 1) SCL-derived booleans
    scl_rgb = create_scl_masks(scl_band, WATER_VALUES, ICE_VALUES, MASK_COLORS)
    water    = (scl_rgb[...,2] == 255)        # blue channel == water
    ice      = (scl_rgb[...,1] == 255)        # green channel == ice

    # 2) cloud boolean
    cloud    = (cld_band > cld_threshold)

    # 3) ndwi_only: from the 2D boolean mask you’ll extract below
    ndwi_only = ndwi_bool & ~water & ~ice & ~cloud

    # 4) assemble label image
    lbl = np.zeros(scl_band.shape, dtype=np.uint8)
    lbl[ water & ~cloud     ] = 1
    lbl[ ice   & ~cloud & ~water ] = 2
    lbl[ cloud               ] = 3
    lbl[ ndwi_only           ] = 4

    return lbl


def visualize_images(img1, img2, img3, img4, img5, img6):
    fig, axes = plt.subplots(1, 6, figsize=(20, 15))

    images = [img1, img2, img3, img4, img5, img6]
    titles = ['RGB', 'NDWI','water_cloud_mask', 'water_cloud_mask_plus_ndwi', 'ndwi_mask', 'final_mask']

    for i, ax in enumerate(axes.flat):
        if i < len(images):  # Ensure we don't try to access beyond the image list
            ax.imshow(images[i])
            ax.set_title(titles[i])
            ax.axis('off')  # Hide axis ticks and labels

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()


def count_scene_pixels(image_array):
    """Counts the number of blue pixels [0, 0, 255] in a NumPy array.

    Args:
      image_array: A NumPy array representing an image, where each pixel is a
                  3-element array representing RGB values.

    Returns:
      The number of blue pixels in the image.
    """
    water_pixel_count = np.sum(np.all(image_array == [0, 0, 255], axis=-1))
    red_pixel_count = np.sum(np.all(image_array == [255, 0, 0], axis=-1))

    return water_pixel_count, red_pixel_count


def visualize_tiled_image_and_mask(output_dir, tile_size=256):
    rgb_tiles = sorted([f for f in os.listdir(output_dir) if f.startswith("rgb_nir_") and f.endswith(".tif")])
    mask_tiles = sorted([f for f in os.listdir(output_dir) if f.startswith("mask_") and f.endswith(".tif")])

    if not rgb_tiles or not mask_tiles:
        print("No tiles found.")
        return

    # Get all tile positions from filenames
    positions = []
    for file in rgb_tiles:
        parts = file.replace("rgb_nir_", "").replace(".tif", "").split("_")
        if len(parts) == 2:
            y, x = int(parts[0]), int(parts[1])
            positions.append((y, x))

    if not positions:
        print("No valid RGB tile positions found.")
        return

    max_y = max(y for y, _ in positions)
    max_x = max(x for _, x in positions)

    n_rows = max_y + 1
    n_cols = max_x + 1

    print(f"Reconstructing grid of size {n_rows} rows x {n_cols} columns")

    full_img = np.zeros((n_rows * tile_size, n_cols * tile_size, 3), dtype=np.uint8)
    full_mask = np.zeros((n_rows * tile_size, n_cols * tile_size), dtype=np.uint8)

    for file in rgb_tiles:
        parts = file.replace("rgb_nir_", "").replace(".tif", "").split("_")
        y_tile, x_tile = int(parts[0]), int(parts[1])

        with rasterio.open(os.path.join(output_dir, file)) as src:
            rgb = src.read([1, 2, 3])  # (3, H, W)
            rgb = np.transpose(rgb, (1, 2, 0))  # (H, W, 3)

            # Stretch normalization for display
            rgb_min = np.percentile(rgb, 2)
            rgb_max = np.percentile(rgb, 98)
            rgb_vis = np.clip((rgb - rgb_min) / (rgb_max - rgb_min), 0, 1)
            rgb_uint8 = (rgb_vis * 255).astype(np.uint8)

        full_img[
            y_tile * tile_size:(y_tile + 1) * tile_size,
            x_tile * tile_size:(x_tile + 1) * tile_size
        ] = rgb_uint8

    for file in mask_tiles:
        parts = file.replace("mask_", "").replace(".tif", "").split("_")
        y_tile, x_tile = int(parts[0]), int(parts[1])

        with rasterio.open(os.path.join(output_dir, file)) as src:
            mask = src.read(1)

        full_mask[
            y_tile * tile_size:(y_tile + 1) * tile_size,
            x_tile * tile_size:(x_tile + 1) * tile_size
        ] = mask

    plt.figure(figsize=(7, 3))

    plt.subplot(1, 2, 1)
    plt.imshow(full_img)
    plt.title("Reconstructed RGB Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    cmap = plt.cm.get_cmap('viridis', 3)
    plt.imshow(full_mask, cmap=cmap, vmin=0, vmax=2)
    plt.title("Reconstructed Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()