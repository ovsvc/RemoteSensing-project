import torch
from matplotlib import pyplot as plt
import seaborn as sns

class SegmentationMetrics:
    def __init__(self, num_classes: int, focus_class: int = 1):
        self.num_classes = num_classes
        self.focus_class = focus_class
        self.reset()

    def reset(self):
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)

    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        if prediction.shape != target.shape:
            raise ValueError("Shape mismatch between prediction and target.")
        prediction = prediction.flatten()
        target = target.flatten()
        for t, p in zip(target, prediction):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                self.confusion_matrix[t, p] += 1

    def get_stats(self, class_index):
        tp = self.confusion_matrix[class_index, class_index]
        fp = self.confusion_matrix[:, class_index].sum() - tp
        fn = self.confusion_matrix[class_index, :].sum() - tp
        tn = self.confusion_matrix.sum() - (tp + fp + fn)
        return tp, fp, fn, tn

    def precision(self, class_index):
        tp, fp, _, _ = self.get_stats(class_index)
        return tp / (tp + fp).float() if (tp + fp) > 0 else torch.tensor(0.0)

    def recall(self, class_index):
        tp, _, fn, _ = self.get_stats(class_index)
        return tp / (tp + fn).float() if (tp + fn) > 0 else torch.tensor(0.0)

    def accuracy(self, class_index):
        tp, fp, fn, tn = self.get_stats(class_index)
        return (tp + tn) / (tp + tn + fp + fn).float() if (tp + tn + fp + fn) > 0 else torch.tensor(0.0)

    def dice(self, class_index):
        tp, fp, fn, _ = self.get_stats(class_index)
        denom = 2 * tp + fp + fn
        return 2 * tp / denom.float() if denom > 0 else torch.tensor(0.0)

    def iou(self, class_index):
        tp, fp, fn, _ = self.get_stats(class_index)
        denom = tp + fp + fn
        return tp / denom.float() if denom > 0 else torch.tensor(0.0)

    def mean_iou(self):
        ious = [self.iou(i) for i in range(self.num_classes)]
        valid = [iou for iou in ious if iou > 0]
        return torch.mean(torch.tensor(valid)) if valid else torch.tensor(0.0)

    def overall_accuracy(self):
        correct = self.confusion_matrix.diag().sum()
        total = self.confusion_matrix.sum()
        return correct / total if total > 0 else torch.tensor(0.0)

    def overall_precision(self):
        tp = self.confusion_matrix.diag().sum()
        fp = self.confusion_matrix.sum(0) - self.confusion_matrix.diag()
        total_fp = fp.sum()
        return tp / (tp + total_fp) if (tp + total_fp) > 0 else torch.tensor(0.0)

    def overall_recall(self):
        tp = self.confusion_matrix.diag().sum()
        fn = self.confusion_matrix.sum(1) - self.confusion_matrix.diag()
        total_fn = fn.sum()
        return tp / (tp + total_fn) if (tp + total_fn) > 0 else torch.tensor(0.0)

    def plot_confusion_matrix(self, class_names=None, normalize=False, figsize=(6, 5)):
        cm = self.confusion_matrix.cpu().numpy()
        
        if normalize:
            cm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-6)
    
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
        plt.tight_layout()
        plt.show()

    def overall_dice(self):
        tp = self.confusion_matrix.diag().sum()
        fp = self.confusion_matrix.sum(0) - self.confusion_matrix.diag()
        fn = self.confusion_matrix.sum(1) - self.confusion_matrix.diag()
        denom = 2 * tp + fp.sum() + fn.sum()
        return 2 * tp / denom if denom > 0 else torch.tensor(0.0)

    def all_metrics(self):
        return {
            i: {
                "IoU": self.iou(i).item(),
                "Dice": self.dice(i).item(),
                "Precision": self.precision(i).item(),
                "Recall": self.recall(i).item()
            }
            for i in range(self.num_classes)
        }

    def focus_metrics(self):
        i = self.focus_class
        return {
            "IoU": self.iou(i).item(),
            "Dice": self.dice(i).item(),
            "Precision": self.precision(i).item(),
            "Recall": self.recall(i).item()
        }

    def global_metrics(self):
        return {
            "Overall Accuracy": self.overall_accuracy().item(),
            "Overall Precision": self.overall_precision().item(),
            "Overall Recall": self.overall_recall().item(),
            "Overall Dice": self.overall_dice().item(),
            "Mean IoU": self.mean_iou().item()
        }

    def __str__(self):
        f = self.focus_metrics()
        g = self.global_metrics()
        return (f"Focus (class {self.focus_class}) → "
                f"IoU: {f['IoU']:.3f}, Dice: {f['Dice']:.3f}, "
                f"Precision: {f['Precision']:.3f}, Recall: {f['Recall']:.3f}\n"
                f"Global → Acc: {g['Overall Accuracy']:.3f}, mIoU: {g['Mean IoU']:.3f}, "
                f"Dice: {g['Overall Dice']:.3f}, Prec: {g['Overall Precision']:.3f}, "
                f"Recall: {g['Overall Recall']:.3f}")
