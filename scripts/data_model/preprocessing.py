import os
import json
import random
from typing import List, Tuple, Dict
import numpy as np
import rasterio
from sklearn.model_selection import train_test_split
from tqdm import tqdm

random.seed(42)


#Image, mask pairs

def gather_image_mask_pairs(data_dir: str) -> List[Tuple[str, str]]:
    """
    Collect all (image, mask) pairs inside OK/ and Modify/ directories.
    """
    pairs = []
    for status_folder in ["OK", "Modify"]:
        status_path = os.path.join(data_dir, status_folder)
        if not os.path.exists(status_path):
            continue
        for image_folder in os.listdir(status_path):
            full_image_folder = os.path.join(status_path, image_folder)
            if not os.path.isdir(full_image_folder):
                continue

            all_files = os.listdir(full_image_folder)
            image_files = [f for f in all_files if f.startswith("rgb_nir_") and f.endswith(".tif")]

            for image_file in image_files:
                image_path = os.path.join(full_image_folder, image_file)
                mask_file = image_file.replace("rgb_nir_", "mask_")
                mask_path = os.path.join(full_image_folder, mask_file)
                if os.path.exists(mask_path):
                    pairs.append((image_path, mask_path))
    return pairs

def gather_all_years(root_dir: str) -> List[Tuple[str, str]]:
    """
    Traverse all year folders and gather image-mask pairs.
    """
    all_pairs = []
    for year_folder in os.listdir(root_dir):
        year_path = os.path.join(root_dir, year_folder)
        if os.path.isdir(year_path):
            print(f"Processing year: {year_folder}")
            pairs = gather_image_mask_pairs(year_path)
            all_pairs.extend(pairs)
    return all_pairs

def split_dataset(pairs: List[Tuple[str, str]], val_size=0.15, test_size=0.15):
    train_val, test = train_test_split(pairs, test_size=test_size, random_state=42)
    train, val = train_test_split(train_val, test_size=val_size / (1 - test_size), random_state=42)
    return train, val, test

def save_splits(train, val, test, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    for name, split in zip(["train", "val", "test"], [train, val, test]):
        with open(os.path.join(save_dir, f"{name}_split.json"), "w") as f:
            json.dump(split, f)


#Oversampling water

def get_sample_water_ratios(pairs: List[Tuple[str, str]]) -> List[Tuple[float, Tuple[str, str]]]:
    """
    Returns list of (water_ratio, (image_path, mask_path)) tuples.
    """
    result = []
    for img_path, mask_path in tqdm(pairs, desc="Scanning masks for water"):
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
        water_pixels = np.sum(mask == 1)
        total_pixels = mask.size
        ratio = water_pixels / total_pixels
        result.append((ratio, (img_path, mask_path)))
    return result

def save_water_ratios_to_json(ratios_with_pairs, filepath):
    """
    Save list of (ratio, (image_path, mask_path)) to JSON.
    """
    serializable = [
        {"ratio": ratio, "image": pair[0], "mask": pair[1]}
        for ratio, pair in ratios_with_pairs
    ]
    with open(filepath, "w") as f:
        json.dump(serializable, f, indent=2)

def bin_samples_by_water_ratio(ratios_with_pairs):
    bins = {
        "0": [],
        "0-10": [],
        "10-20": [],
        "20-30": [],
        "30+": [],
    }
    for ratio, pair in ratios_with_pairs:
        if ratio == 0:
            bins["0"].append(pair)
        if 0 < ratio <= 0.10:
            bins["0-10"].append(pair)
        elif ratio <= 0.20:
            bins["10-20"].append(pair)
        elif ratio <= 0.30:
            bins["20-30"].append(pair)
        else:
            bins["30+"].append(pair)
    return bins

def count_bins(ratios_with_pairs):
    """
    Count number of samples in each water ratio bin.
    """
    bin_counts = {
        "0": 0,
        "0-10": 0,
        "10-20": 0,
        "20-30": 0,
        "30+": 0,
    }

    for ratio, _ in ratios_with_pairs:
        if ratio == 0:
            bin_counts["0"] += 1
        if 0 < ratio <= 0.10:
            bin_counts["0-10"] += 1
        elif ratio <= 0.20:
            bin_counts["10-20"] += 1
        elif ratio <= 0.30:
            bin_counts["20-30"] += 1
        else:
            bin_counts["30+"] += 1

    return bin_counts

def sample_from_bins(bins, sizes: Dict[str, int], max_multiplier=2):
    sampled = []
    for bin_name, bin_samples in bins.items():
        count = sizes.get(bin_name, 0)
        available = len(bin_samples)

        if count <= available:
            # Sample without replacement
            sampled += random.sample(bin_samples, count)
        elif count <= available * max_multiplier:
            # Sample with limited duplication
            full_copies = count // available
            remainder = count % available
            sampled += bin_samples * full_copies
            sampled += random.sample(bin_samples, remainder)
        else:
            print(f"Requested {count} from bin '{bin_name}', "
                  f"but only {available} available (limit is {available * max_multiplier}). "
                  f"Using all available instead.")
            sampled += bin_samples
    return sampled



# Fuction for combining all steps

def load_water_ratios_from_json(filepath):
    """
    Load list of (ratio, (image_path, mask_path)) from JSON.
    """
    with open(filepath, "r") as f:
        data = json.load(f)
    return [(item["ratio"], (item["image"], item["mask"])) for item in data]

def collect_patch_paths_and_save(data_dir: str, save_path: str, mask_prefix="mask_"):
    """
    Traverse OK/ and Modify/ folders to collect tile-level (image, mask) patch pairs.
    Save them in the same JSON format used in preprocessing.
    """
    collected_pairs = []
    missing = 0

    for status_folder in ["OK", "Modify"]:
        status_path = os.path.join(data_dir, status_folder)
        if not os.path.exists(status_path):
            print(f"Missing: {status_path}")
            continue

        for tile_folder in os.listdir(status_path):
            full_tile_path = os.path.join(status_path, tile_folder)
            if not os.path.isdir(full_tile_path):
                continue

            all_files = os.listdir(full_tile_path)
            image_files = [f for f in all_files if f.startswith("rgb_nir_") and f.endswith(".tif")]

            for img_file in image_files:
                img_path = os.path.join(full_tile_path, img_file)
                mask_file = img_file.replace("rgb_nir_", mask_prefix)
                mask_path = os.path.join(full_tile_path, mask_file)

                if os.path.exists(mask_path):
                    collected_pairs.append((img_path, mask_path))
                else:
                    missing += 1
                    print(f"Missing mask for {img_file} → expected {mask_file}")

    if not collected_pairs:
        print("No valid pairs found.")
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(collected_pairs, f, indent=2)
        print(f"Saved {len(collected_pairs)} image-mask patch pairs to {save_path}")
        print(f"Skipped {missing} image files due to missing masks.")

def run_preprocessing(data_dir: str,
                      save_dir: str,
                      desired_sizes: Dict[str, int],
                      test_mode: bool = False,
                      test_fraction: float = 0.1,
                      precomputed_ratios_path: str = None):
    """
    Gathers all data, performs smart sampling from water ratio bins using desired_sizes,
    splits the sampled dataset into train/val/test (70/15/15), and saves the splits as JSON.
    """
    print("Gathering image-mask pairs...")
    all_pairs = gather_all_years(data_dir)
    print(f"Found {len(all_pairs)} total samples.")

    if test_mode:
        count = max(1, int(len(all_pairs) * test_fraction))
        all_pairs = random.sample(all_pairs, count)
        print(f"Test mode: Using only {count} samples.")

    
    if precomputed_ratios_path and os.path.exists(precomputed_ratios_path):
        print(f"Loading precomputed ratios from {precomputed_ratios_path}...")
        ratios_with_pairs = load_water_ratios_from_json(precomputed_ratios_path)
    else:
        print("Calculating water ratios from scratch...")
        ratios_with_pairs = get_sample_water_ratios(all_pairs)
        save_water_ratios_to_json(ratios_with_pairs, "./Data/Split/water_ratios.json")
    

    print("Binning samples by water ratio...")
    binned = bin_samples_by_water_ratio(ratios_with_pairs)
    bin_counts = count_bins(ratios_with_pairs)

    print("Original bin distribution:")
    for bin_name, count in bin_counts.items():
        print(f"  Bin {bin_name}: {count} samples")

    print("Sampling according to desired sizes...")
    sampled_pairs = sample_from_bins(binned, desired_sizes)
    print(f"Sampled {len(sampled_pairs)} pairs in total.")

    print("Splitting sampled dataset into train/val/test (70/15/15)...")
    train, val, test = split_dataset(sampled_pairs)

    print("Saving splits to disk...")
    save_splits(train, val, test, save_dir)

    print("Processing complete.")
