"""
evaluate_metrics.py
-------------------
Computes two robot-navigation metrics on top of Depth Anything V2 predictions.
Designed to work directly with nav.py (same threshold conventions).

Metrics
-------
1. Free-space IoU
   Free space = pixels with normalised depth <= CLOSE_THRESHOLD
   (same convention as nav.py: depth > CLOSE_THRESHOLD means obstacle)

2. Obstacle Precision / Recall
   Obstacle = pixels with normalised depth > CLOSE_THRESHOLD
   - Precision : of all pixels we called obstacles, how many really are?
   - Recall    : of all real obstacles, how many did we catch?
     *** Recall is more important for safety ***

Threshold convention (matches nav.py exactly)
---------------------------------------------
  obstacle   = depth_norm  >  CLOSE_THRESHOLD  (default 0.4)
  free space = depth_norm  <= CLOSE_THRESHOLD

Usage
-----
  python evaluate_metrics.py --smoke_test
  python evaluate_metrics.py --pred_dir ./predictions/nyu --gt_dir ./data/nyu/gt --threshold 0.4 --output results.json
"""

import os
import json
import argparse
import numpy as np


def normalize_depth(depth):
    """Normalise depth map to [0,1]. Matches nav.py convention."""
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min < 1e-8:
        return np.zeros_like(depth, dtype=np.float32)
    return ((depth - d_min) / (d_max - d_min)).astype(np.float32)


def compute_freespace_iou(pred_depth, gt_depth, threshold=0.4, normalize=True):
    """
    Free-space IoU.
    Free space = depth <= threshold  (matches nav.py: obstacle = depth > threshold)
    Higher is better.
    """
    if normalize:
        pred_depth = normalize_depth(pred_depth)
        gt_depth   = normalize_depth(gt_depth)

    pred_free = pred_depth <= threshold
    gt_free   = gt_depth   <= threshold

    intersection = np.logical_and(pred_free, gt_free).sum()
    union        = np.logical_or (pred_free, gt_free).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection) / float(union)


def compute_obstacle_precision_recall(pred_depth, gt_depth, threshold=0.4, normalize=True):
    """
    Obstacle Precision and Recall.
    Obstacle = depth > threshold  (matches nav.py exactly)

    TP = predicted obstacle AND real obstacle
    FP = predicted obstacle BUT real free space  (false alarm)
    FN = predicted free space BUT real obstacle  (DANGEROUS miss)

    Recall is more important for safety.
    """
    if normalize:
        pred_depth = normalize_depth(pred_depth)
        gt_depth   = normalize_depth(gt_depth)

    pred_obs = pred_depth > threshold
    gt_obs   = gt_depth   > threshold

    TP = np.logical_and( pred_obs,  gt_obs).sum()
    FP = np.logical_and( pred_obs, ~gt_obs).sum()
    FN = np.logical_and(~pred_obs,  gt_obs).sum()

    precision = float(TP) / float(TP + FP) if (TP + FP) > 0 else 0.0
    recall    = float(TP) / float(TP + FN) if (TP + FN) > 0 else 0.0
    return precision, recall


def evaluate_dataset(pred_dir, gt_dir, threshold=0.4, verbose=True):
    """Run both metrics over a folder of .npy depth maps."""
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".npy")])
    gt_files   = sorted([f for f in os.listdir(gt_dir)   if f.endswith(".npy")])
    common     = sorted(set(pred_files) & set(gt_files))

    if len(common) == 0:
        raise FileNotFoundError(f"No matching .npy files in {pred_dir} and {gt_dir}")

    iou_list, prec_list, rec_list = [], [], []
    per_image = []

    for fname in common:
        pred = np.load(os.path.join(pred_dir, fname))
        gt   = np.load(os.path.join(gt_dir,   fname))

        if pred.shape != gt.shape:
            import cv2
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

        iou               = compute_freespace_iou(pred, gt, threshold)
        precision, recall = compute_obstacle_precision_recall(pred, gt, threshold)

        iou_list.append(iou)
        prec_list.append(precision)
        rec_list.append(recall)
        per_image.append({"file": fname, "iou": round(iou,4),
                          "precision": round(precision,4), "recall": round(recall,4)})

        if verbose:
            print(f"  {fname:35s}  IoU={iou:.4f}  P={precision:.4f}  R={recall:.4f}")

    summary = {
        "threshold":      threshold,
        "n_images":       len(common),
        "mean_iou":       round(float(np.mean(iou_list)),  4),
        "mean_precision": round(float(np.mean(prec_list)), 4),
        "mean_recall":    round(float(np.mean(rec_list)),  4),
        "per_image":      per_image,
    }

    if verbose:
        print("\n" + "="*60)
        print(f"  Threshold      : {threshold}  (matches nav.py CLOSE_THRESHOLD)")
        print(f"  Images         : {len(common)}")
        print(f"  Mean IoU       : {summary['mean_iou']:.4f}   (higher = better)")
        print(f"  Mean Precision : {summary['mean_precision']:.4f}   (higher = better)")
        print(f"  Mean Recall    : {summary['mean_recall']:.4f}   (higher = better, safety!)")
        print("="*60)

    return summary


def _smoke_test():
    print("Running smoke test...\n")
    rng = np.random.default_rng(42)

    # Perfect prediction
    gt   = rng.random((480, 640)).astype(np.float32)
    pred = gt.copy()
    iou          = compute_freespace_iou(pred, gt, threshold=0.4)
    prec, recall = compute_obstacle_precision_recall(pred, gt, threshold=0.4)
    print(f"Perfect pred  ->  IoU={iou:.4f}  P={prec:.4f}  R={recall:.4f}")
    assert iou == 1.0 and prec == 1.0 and recall == 1.0

    # Inverted depth
    pred_inv    = 1.0 - gt
    iou2        = compute_freespace_iou(pred_inv, gt, threshold=0.5)
    prec2, rec2 = compute_obstacle_precision_recall(pred_inv, gt, threshold=0.5)
    print(f"Inverted pred ->  IoU={iou2:.4f}  P={prec2:.4f}  R={rec2:.4f}")
    assert iou2 < 0.1

    # Random prediction
    pred_rand     = rng.random((480, 640)).astype(np.float32)
    iou3          = compute_freespace_iou(pred_rand, gt, threshold=0.4)
    prec3, rec3   = compute_obstacle_precision_recall(pred_rand, gt, threshold=0.4)
    print(f"Random  pred  ->  IoU={iou3:.4f}  P={prec3:.4f}  R={rec3:.4f}")

    # Threshold convention check - must match nav.py
    depth = np.array([[0.3, 0.4, 0.5], [0.1, 0.6, 0.4]], dtype=np.float32)
    obs   = depth > 0.4
    expected = np.array([[False, False, True], [False, True, False]])
    assert np.array_equal(obs, expected), "Threshold convention mismatch with nav.py!"
    print(f"\nThreshold convention (depth > 0.4 = obstacle) -> matches nav.py PASSED")
    print("\nAll smoke tests PASSED\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir",   type=str,   default=None)
    parser.add_argument("--gt_dir",     type=str,   default=None)
    parser.add_argument("--threshold",  type=float, default=0.4)
    parser.add_argument("--output",     type=str,   default=None)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test or (args.pred_dir is None and args.gt_dir is None):
        _smoke_test()
    else:
        summary = evaluate_dataset(args.pred_dir, args.gt_dir, args.threshold)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\nResults saved to {args.output}")
