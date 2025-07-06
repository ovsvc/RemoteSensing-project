import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.plot import show
from rasterio.enums import Resampling
from rasterio.windows import Window
import os
import tarfile
from PIL import Image
from pathlib import Path



def rgb_to_class_id(rgb_mask):
    """Convert RGB-encoded mask to class ID mask."""
    class_id_mask = np.zeros((rgb_mask.shape[0], rgb_mask.shape[1]), dtype=np.uint8)
    class_id_mask[(rgb_mask == [0, 0, 255]).all(axis=-1)] = 1  # Water
    class_id_mask[(rgb_mask == [255, 0, 0]).all(axis=-1)] = 2  # Cloud
    class_id_mask[(rgb_mask == [0, 255, 0]).all(axis=-1)] = 3  # Ice/snow (if present)
    class_id_mask[(rgb_mask == [200, 200, 200]).all(axis=-1)] = 0  # Background
    return class_id_mask

def save_tiled_chunks(data, mask_rgb, output_dir, tile_size=256):
    os.makedirs(output_dir, exist_ok=True)

    n_tiles = 2048 // tile_size
    tile_coords = [(y, x) for y in range(n_tiles) for x in range(n_tiles)]

    for y_tile, x_tile in tile_coords:
        y_start = y_tile * tile_size
        x_start = x_tile * tile_size
        window = Window(x_start, y_start, tile_size, tile_size)

        datamask = data['data_mask'][y_start:y_start+tile_size, x_start:x_start+tile_size]
        tile_data = data['rgb_nir'][:, y_start:y_start+tile_size, x_start:x_start+tile_size]
        mask_tile_rgb = mask_rgb[y_start:y_start+tile_size, x_start:x_start+tile_size]

        cloud_pixels = np.sum(mask_tile_rgb[:, :, 0] == 255)  # red
        total_pixels = tile_size * tile_size
        red_ratio = cloud_pixels / total_pixels

        if np.all(datamask == 1) and red_ratio < 0.51:
            tile_profile = data['rgb_nir_profile'].copy()
            tile_profile.update({
                'driver': 'GTiff',
                'dtype': 'int16',
                'compress': 'DEFLATE' if 'compress' not in tile_profile else tile_profile['compress'],
                'tiled': True,
                'blockxsize': 256,
                'blockysize': 256,
                'height': tile_size,
                'width': tile_size,
                'transform': rasterio.windows.transform(window, data['rgb_nir_profile']['transform'])
            })

            # Save image tile
            img_tile_path = os.path.join(output_dir, f"rgb_nir_{y_tile:02d}_{x_tile:02d}.tif")
            with rasterio.open(img_tile_path, 'w', **tile_profile) as dst:
                dst.write(tile_data)

            # Create class ID mask from RGB
            class_id_mask = rgb_to_class_id(mask_tile_rgb)

            # Save mask as GeoTIFF (same spatial reference)
            mask_profile = tile_profile.copy()
            mask_profile.update({
                'count': 1,
                'dtype': 'uint8'
            })
            mask_tile_path = os.path.join(output_dir, f"mask_{y_tile:02d}_{x_tile:02d}.tif")
            with rasterio.open(mask_tile_path, 'w', **mask_profile) as dst:
                dst.write(class_id_mask, 1)

            print(f"Saved tile {y_tile},{x_tile} with bounds {dst.bounds}")






def rgb_to_class_id_npy(rgb_mask):
    class_id_mask = np.zeros((rgb_mask.shape[0], rgb_mask.shape[1]), dtype=np.uint8)
    class_id_mask[(rgb_mask == [0, 0, 255]).all(axis=-1)] = 1  # Water
    class_id_mask[(rgb_mask == [255, 0, 0]).all(axis=-1)] = 2  # Cloud
    class_id_mask[(rgb_mask == [0, 255, 0]).all(axis=-1)] = 3  # Ice/snow (if present)
    class_id_mask[(rgb_mask == [200, 200, 200]).all(axis=-1)] = 0  # Background
    return class_id_mask

def save_tiled_chunks_npy(npy_path, response_tar_path, output_dir, tile_size=256):
    os.makedirs(output_dir, exist_ok=True)

    labels = np.load(npy_path)
    H, W = labels.shape

    # Prepare class ID mask
    mask_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    mask_rgb[labels == 0] = (200, 200, 200)
    mask_rgb[labels == 1] = (0, 0, 255)
    mask_rgb[labels == 2] = (0, 255, 0)
    mask_rgb[labels == 3] = (255, 0, 0)
    mask_rgb[labels == 4] = (0, 0, 255)
    class_id_mask = rgb_to_class_id_npy(mask_rgb)

    bands = read_dataset_tar(response_tar_path)
    rgb_nir = bands['rgb_nir']  # (4, H, W)
    profile = bands['rgb_nir_profile']
    transform = profile['transform']
    crs = profile['crs']
    n_tiles = H // tile_size

    # Use your scaling function and convert to uint8
    rgb_scaled = create_rgb(rgb_nir[:3])  # (H, W, 3), float32 in [0,1]
    rgb_scaled = (rgb_scaled * 255).astype(np.uint8)  # (H, W, 3) → uint8
    rgb_scaled = np.transpose(rgb_scaled, (2, 0, 1))  # (3, H, W)

    nir_band = rgb_nir[3]  # Still int16

    for y_tile in range(n_tiles):
        for x_tile in range(n_tiles):
            y_start = y_tile * tile_size
            x_start = x_tile * tile_size
            window = Window(x_start, y_start, tile_size, tile_size)

            rgb_tile = rgb_scaled[:, y_start:y_start+tile_size, x_start:x_start+tile_size]
            nir_tile = nir_band[y_start:y_start+tile_size, x_start:x_start+tile_size].astype(np.uint8)
            mask_tile = class_id_mask[y_start:y_start+tile_size, x_start:x_start+tile_size]

            cloud_pixels = np.sum(mask_tile == 2)
            red_ratio = cloud_pixels / (tile_size * tile_size)
            if red_ratio <= 0.51:

                tile_transform = rasterio.windows.transform(window, transform)

                common_profile = profile.copy()
                common_profile.update({
                    'height': tile_size,
                    'width': tile_size,
                    'transform': tile_transform,
                    'crs': crs,
                    'tiled': True,
                    'compress': 'DEFLATE',
                    'blockxsize': tile_size,
                    'blockysize': tile_size,
                    'driver': 'GTiff',
                })

                # Combine and save
                img_tile = np.concatenate([rgb_tile, nir_tile[np.newaxis, :, :]], axis=0)
                img_profile = common_profile.copy()
                img_profile.update({'count': 4, 'dtype': 'uint8'})
                img_path = os.path.join(output_dir, f"rgb_nir_{y_tile:02d}_{x_tile:02d}.tif")
                with rasterio.open(img_path, 'w', **img_profile) as dst:
                    dst.write(img_tile)
                print(f"Saved image tile: {img_path}")

                # Save mask
                mask_profile = common_profile.copy()
                mask_profile.update({'count': 1, 'dtype': 'uint8'})
                mask_path = os.path.join(output_dir, f"mask_{y_tile:02d}_{x_tile:02d}.tif")
                with rasterio.open(mask_path, 'w', **mask_profile) as dst:
                    dst.write(mask_tile, 1)
                print(f"Saved mask tile: {mask_path}")






def process_all_npy_and_tar_pairs(modify_folder, output_root, tile_size=256):
    modify_path = Path(modify_folder)
    output_root = Path(output_root)

    npy_files = list(modify_path.glob("*.npy"))
    total = len(npy_files)

    for npy_path in modify_path.glob("*.npy"):
        tile_id = npy_path.stem
        response_tar_path = modify_path / tile_id / "response.tar"
        output_dir = output_root / tile_id

        if response_tar_path.exists():
            print(f"\nProcessing {tile_id} out of {total}")
            save_tiled_chunks_npy(npy_path, response_tar_path, output_dir, tile_size)
        else:
            print(f"response.tar not found for {tile_id}")
Ü