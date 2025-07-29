import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    jaccard_score
)

import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch.nn.functional as F
import os
import re
import rasterio
from rasterio.windows import from_bounds
from skimage.transform import resize
import glob


def plot_confusion_matrix(all_labels, all_predictions, classes, normalize=False, fontsize=8):
    """
    Plots a tidier confusion matrix.

    Args:
        all_labels (np.ndarray): Ground-truth labels.
        all_predictions (np.ndarray): Model predictions.
        classes (list): Non-numerical names of the classes.
        normalize (bool): Whether to normalize the confusion matrix.
        fontsize (int): Font size for annotations.
    """
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = "Normalized Confusion Matrix"
    else:
        title = "Confusion Matrix"

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 12))  # Adjust size for clarity
    cax = ax.matshow(cm, cmap=plt.cm.Blues)  # Heatmap display
    plt.colorbar(cax)

    # Add labels
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation = 90, ha="right", fontsize=fontsize)
    ax.set_yticklabels(classes, fontsize=fontsize)
    
    # Annotate cells with values, omitting zeros for tidiness
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            if value != 0:  # Skip zero values
                ax.text(j, i, f"{value}", ha="center", va="center",
                        color="white" if value > cm.max() / 2 else "black", fontsize=fontsize)

    plt.title(title, fontsize=fontsize + 2)
    plt.xlabel("Predicted Labels", fontsize=fontsize + 2)
    plt.ylabel("True Labels", fontsize=fontsize + 2)
    plt.tight_layout()
    plt.show()

    return cm

def analyze_test_results(test_loss, test_accuracy, test_per_class_accuracy, all_labels, all_predictions, classes):
    """
    Analyzes the test results and calculates metrics such as confusion matrix, precision, recall, F1-score, 
    and accuracy. Plots the confusion matrix and prints a classification report.

    Args:
        test_loss (float): The test loss value.
        test_accuracy (float): The overall test accuracy.
        test_per_class_accuracy (list or np.ndarray): Per-class accuracy values.
        all_labels (np.ndarray): Ground-truth labels.
        all_predictions (np.ndarray): Model predictions.
        classes (list): Non-numerical names of the classes.

    Returns:
        dict: A dictionary containing precision, recall, F1-score, and accuracy.
    """

    # Calculate confusion matrix
    cm = plot_confusion_matrix(all_labels, all_predictions, classes, normalize=True)

    # Calculate additional metrics
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    accuracy = accuracy_score(all_labels, all_predictions)

    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=classes))

    # Package metrics in a dictionary
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "confusion_matrix": cm
    }

    # Print summary
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Overall Precision: {precision:.4f}")
    print(f"Overall Recall: {recall:.4f}")
    print(f"Overall F1-Score: {f1:.4f}")
    print(f"Overall Accuracy: {accuracy:.4f}")

    return metrics




###############################################
#MCDROPOUT

def enable_mc_dropout(model: torch.nn.Module):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout) or isinstance(m, torch.nn.Dropout2d):
            m.train()


@torch.no_grad()
def get_mc_dropout_predictions(model, input_tensor, n_iter=30, class_idx=1):
    model.eval()
    enable_mc_dropout(model)
    preds = []
    for _ in range(n_iter):
        out_dict = model(input_tensor, return_water_prob=False)
        logits = out_dict["out"]
        probs = F.softmax(logits, dim=1)
        preds.append(probs)
    preds = torch.stack(preds)         # (T, B, C, H, W)
    mean = preds.mean(dim=0)           # (B, C, H, W)
    var = preds.var(dim=0)             # (B, C, H, W)
    return mean, var[:, class_idx, :, :]  # (B, C, H, W), (B, H, W)


def detect_flood_with_uncertainty(
    image_path, model, perm_water_path, threshold=0.5, device='cpu', n_iter=30
):
    model.eval()
    model.to(device)

    with rasterio.open(image_path) as src:
        rgb_img = src.read([1, 2, 3])
        bounds = src.bounds
        crs = src.crs
        shape = src.shape

    rgb_img = rgb_img.astype(np.float32)
    rgb_img = rgb_img / rgb_img.max()
    img_tensor = torch.tensor(rgb_img, dtype=torch.float32).to(device).unsqueeze(0)

    with torch.no_grad():
        out = model(img_tensor, return_water_prob=True)
        water_prob = out["water_prob"].squeeze(0).cpu().numpy()
        pred_mask = torch.argmax(out["out"].squeeze(0), dim=0).cpu().numpy()

    mean_probs, var_map = get_mc_dropout_predictions(model, img_tensor, n_iter=n_iter)
    mc_water_prob = mean_probs[0, 1].cpu().numpy()
    uncertainty_map = var_map[0].cpu().numpy()

    with rasterio.open(perm_water_path) as pw_src:
        if pw_src.crs != crs:
            raise ValueError("CRS mismatch between image and permanent water map.")
        window = from_bounds(*bounds, transform=pw_src.transform)
        perm_crop = pw_src.read(1, window=window)

    perm_resized = resize(
        perm_crop, shape, order=0, preserve_range=True, anti_aliasing=False
    ).astype(np.uint8)
    permanent_water_mask = (perm_resized == 1).astype(float)

    predicted_water = (water_prob >= threshold).astype(float)
    flood_mask = np.clip(predicted_water - permanent_water_mask, 0, 1)

    return rgb_img, pred_mask, water_prob, flood_mask, permanent_water_mask, mc_water_prob, uncertainty_map

def get_patch_grid(patch_paths, prefix="rgb_nir"):
    grid = []
    pattern = re.compile(rf"{prefix}_(\d+)_(\d+)(?:_[A-Z])?\.tif")
    for path in patch_paths:
        filename = os.path.basename(path)
        match = pattern.match(filename)
        if match:
            row, col = map(int, match.groups()[:2])
            grid.append((row, col, path))
    return sorted(grid, key=lambda x: (x[0], x[1]))


def stitch_with_uncertainty(grid, model, perm_water_path, mask_prefix="mask", threshold=0.5, device="cuda", n_iter=30):
    rows = [p[0] for p in grid]
    cols = [p[1] for p in grid]
    max_row = max(rows) + 1
    max_col = max(cols) + 1

    sample_patch = grid[0][2]
    with rasterio.open(sample_patch) as src:
        h, w = src.height, src.width

    rgb_stitched = np.zeros((3, max_row * h, max_col * w), dtype=np.float32)
    gt_stitched = np.zeros((max_row * h, max_col * w), dtype=np.uint8)
    pred_stitched = np.zeros_like(gt_stitched)
    mc_mean_stitched = np.zeros_like(rgb_stitched[0], dtype=np.float32)
    mc_var_stitched = np.zeros_like(rgb_stitched[0], dtype=np.float32)

    for row, col, rgb_path in grid:
        mask_name = os.path.basename(rgb_path).replace("rgb_nir", mask_prefix)
        mask_path = os.path.join(os.path.dirname(rgb_path), mask_name)

        with rasterio.open(mask_path) as src:
            gt_patch = src.read(1)

        rgb_img, pred_mask, _, _, _, mc_mean, mc_var = detect_flood_with_uncertainty(
            image_path=rgb_path,
            model=model,
            perm_water_path=perm_water_path,
            threshold=threshold,
            device=device,
            n_iter=n_iter
        )

        y0, y1 = row * h, (row + 1) * h
        x0, x1 = col * w, (col + 1) * w

        rgb_stitched[:, y0:y1, x0:x1] = rgb_img
        gt_stitched[y0:y1, x0:x1] = gt_patch
        pred_stitched[y0:y1, x0:x1] = pred_mask
        mc_mean_stitched[y0:y1, x0:x1] = mc_mean
        mc_var_stitched[y0:y1, x0:x1] = mc_var

    return rgb_stitched, gt_stitched, pred_stitched, mc_mean_stitched, mc_var_stitched


def visualize_uncertainty_minimal(
    rgb_img,
    gt_mask,
    pred_mask,
    mc_water_prob,
    uncertainty_map,
    class_colors=None,
    title=None
):
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.colors as mcolors

    if class_colors is None:
        class_colors = ["#c8c8c8", "blue", "red"]
    cmap = mcolors.ListedColormap(class_colors)

    rgb_vis = np.moveaxis(rgb_img, 0, -1)

    fig, axs = plt.subplots(2, 3, figsize=(20, 10))  # 2 rows, 3 columns
    axs = axs.ravel()  # Flatten to 1D array for easier indexing

    axs[0].imshow(rgb_vis)
    axs[0].set_title("RGB Image", fontsize=16)
    axs[0].axis("off")

    axs[1].imshow(gt_mask, cmap=cmap, vmin=0, vmax=len(class_colors) - 1)
    axs[1].set_title("Ground Truth Mask", fontsize=16)
    axs[1].axis("off")

    axs[2].imshow(pred_mask, cmap=cmap, vmin=0, vmax=len(class_colors) - 1)
    axs[2].set_title("Predicted Mask", fontsize=16)
    axs[2].axis("off")

    im3 = axs[3].imshow(mc_water_prob, cmap='jet', vmin=0, vmax=1)
    axs[3].set_title("MC Dropout\nMean Probability", fontsize=16)
    axs[3].axis("off")
    fig.colorbar(im3, ax=axs[3], fraction=0.046, pad=0.04)

    im4 = axs[4].imshow(uncertainty_map, cmap='hot')
    axs[4].set_title("MC Dropout\nUncertainty (Variance)", fontsize=16)
    axs[4].axis("off")
    fig.colorbar(im4, ax=axs[4], fraction=0.046, pad=0.04)

    axs[5].axis("off")

    if title:
        fig.suptitle(title, fontsize=22)

    plt.tight_layout()
    plt.show()




###############################################
#MCDROPOUT WITH DIFFERENT TRESHOLDS


def evaluate_fixed_gt_with_confidence_thresholds(
    gt_mask,
    mean_prob_map,
    class_prob_map,
    thresholds=np.linspace(0.1, 1.0, 5),
    class_of_interest=1,
):
    """
    Evaluate metrics using a fixed ground truth mask.
    Model predictions below the confidence threshold are considered 'abstained'
    and ignored in metric computation.
    """
    results = []
    all_filtered_preds = []

    H, W = gt_mask.shape
    gt_flat = gt_mask.flatten()
    class_flat = class_prob_map.flatten()
    prob_flat = mean_prob_map.flatten()



    print("mean_prob_map min/max:", mean_prob_map.min(), mean_prob_map.max())
    print("mean_prob_map dtype:", mean_prob_map.dtype)



    for thresh in thresholds:
        # Create mask of predictions above threshold
        confident_mask = prob_flat >= thresh

        confident_mask = prob_flat >= thresh
        print(f"Threshold {thresh:.2f} -> confident pixels: {np.sum(confident_mask)} / {len(prob_flat)}")

        # Build prediction array: -1 means abstain
        pred_with_abstain = np.full_like(class_flat, fill_value=-1)
        pred_with_abstain[confident_mask] = class_flat[confident_mask]

        # Masked comparison (only where prediction is not -1)
        valid = pred_with_abstain != -1
        pred_bin = (pred_with_abstain[valid] == class_of_interest).astype(int)
        gt_bin = (gt_flat[valid] == class_of_interest).astype(int)

        if len(pred_bin) == 0:
            precision = recall = f1 = iou = 0.0
        else:
            precision = precision_score(gt_bin, pred_bin, zero_division=0)
            recall = recall_score(gt_bin, pred_bin, zero_division=0)
            f1 = f1_score(gt_bin, pred_bin, zero_division=0)
            iou = jaccard_score(gt_bin, pred_bin, zero_division=0)

        coverage = np.sum(mean_prob_map >= thresh) / mean_prob_map.size

        results.append({
            "threshold": thresh,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "iou": iou,
            "coverage": coverage,
        })

        # Store the 2D mask
        mask_filtered = pred_with_abstain.reshape(H, W)
        all_filtered_preds.append(mask_filtered)

    return results, all_filtered_preds


def plot_metrics_vs_threshold(results):
    thresholds = [r["threshold"] for r in results]
    f1s = [r["f1_score"] for r in results]
    ious = [r["iou"] for r in results]
    precisions = [r["precision"] for r in results]
    recalls = [r["recall"] for r in results]
    coverage = [r["coverage"] for r in results]

    plt.figure(figsize=(7,3))
    plt.plot(thresholds, precisions, label="Precision")
    plt.plot(thresholds, recalls, label="Recall")
    plt.plot(thresholds, f1s, label="F1 Score")
    plt.plot(thresholds, ious, label="IoU")
    plt.xlabel("Mean Probability Threshold")
    plt.ylabel("Score")
    plt.title("Metrics vs. Confidence Threshold")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_prediction_masks_by_threshold(filtered_masks, thresholds, class_of_interest=1):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import numpy as np

    n = len(thresholds)
    cols = min(n, 5)
    rows = int(np.ceil(n / cols))

    # Define custom colormap: 0 = background/abstain (gray), 1 = water (blue)
    cmap = ListedColormap(["lightgray", "blue"])

    plt.figure(figsize=(4 * cols, 4 * rows))
    for i, (mask, thresh) in enumerate(zip(filtered_masks, thresholds)):
        plt.subplot(rows, cols, i + 1)

        # Create binary mask: 1 = predicted water, 0 = everything else
        binary_mask = (mask == class_of_interest).astype(int)

        plt.imshow(binary_mask, cmap=cmap, vmin=0, vmax=1)
        plt.title(f"Threshold ≥ {thresh:.2f}", fontsize=12)
        plt.axis("off")

    plt.suptitle("Filtered Predictions by Confidence Threshold", fontsize=16)
    plt.tight_layout()
    plt.show()
