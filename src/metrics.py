"""
Evaluation metrics for punctuation restoration.
Macro F1 excludes class 0 (O - no punctuation) per task specification.
"""

import numpy as np
from typing import Dict, Tuple


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 4,
    exclude_class_0_for_macro: bool = True,
) -> Dict[str, float]:
    """
    Compute accuracy, precision, recall, macro F1 (excluding class 0), and F1.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        num_classes: Number of classes (O=0, COMMA=1, PERIOD=2, QUESTION=3)
        exclude_class_0_for_macro: If True, macro F1 is computed over classes 1,2,3 only

    Returns:
        Dict with accuracy, precision, recall, macro_f1, f1 (micro)
    """
    # Mask for valid positions (exclude padding)
    mask = (y_true >= 0) & (y_true < num_classes)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    # Accuracy
    accuracy = np.mean(y_true == y_pred)

    # Per-class TP, FP, FN
    tp = np.zeros(num_classes, dtype=np.float64)
    fp = np.zeros(num_classes, dtype=np.float64)
    fn = np.zeros(num_classes, dtype=np.float64)

    for i in range(len(y_true)):
        cor = y_true[i]
        prd = y_pred[i]
        if cor == prd:
            tp[cor] += 1
        else:
            fn[cor] += 1
            fp[prd] += 1

    # Precision and recall per class
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    for c in range(num_classes):
        precision[c] = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        recall[c] = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0

    # Per-class F1
    f1_per_class = np.zeros(num_classes)
    for c in range(num_classes):
        if precision[c] + recall[c] > 0:
            f1_per_class[c] = 2 * precision[c] * recall[c] / (precision[c] + recall[c])
        else:
            f1_per_class[c] = 0.0

    # Macro F1: exclude class 0 (O) as specified
    if exclude_class_0_for_macro and num_classes > 1:
        classes_for_macro = list(range(1, num_classes))
        macro_f1 = np.mean([f1_per_class[c] for c in classes_for_macro])
    else:
        macro_f1 = np.mean(f1_per_class)

    # Micro F1 (overall F1): 2 * (micro_prec * micro_rec) / (micro_prec + micro_rec)
    total_tp = np.sum(tp)
    total_fp = np.sum(fp)
    total_fn = np.sum(fn)
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_micro = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1_per_class": f1_per_class.tolist(),
        "macro_f1": float(macro_f1),  # Excludes class 0
        "f1": float(f1_micro),  # Micro F1
    }


def compute_metrics_from_batches(
    all_y_true: np.ndarray,
    all_y_pred: np.ndarray,
    all_y_mask: np.ndarray,
    num_classes: int = 4,
) -> Dict[str, float]:
    """
    Compute metrics from batched predictions, using y_mask to select valid tokens.
    """
    # Flatten and apply mask
    y_true_flat = all_y_true.flatten()
    y_pred_flat = all_y_pred.flatten()
    y_mask_flat = all_y_mask.flatten()

    mask = y_mask_flat > 0
    y_true = y_true_flat[mask]
    y_pred = y_pred_flat[mask]

    return compute_metrics(y_true, y_pred, num_classes=num_classes, exclude_class_0_for_macro=True)
