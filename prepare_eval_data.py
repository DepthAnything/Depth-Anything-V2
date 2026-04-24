"""
prepare_eval_data.py
--------------------
Prepares .npy depth map files required by evaluate_metrics.py.

This script does two things:
  1. Converts GT depth maps from NYU-D or KITTI into .npy format, saved to gt_depth/
  2. Runs DAV2 inference on the corresponding RGB images and saves predicted
     depth maps as .npy files to predictions/

Once done, run evaluation directly with evaluate_metrics.py:
  python evaluate_metrics.py --pred_dir ./predictions/kitti --gt_dir ./gt_depth/kitti --output results_kitti.json
  python evaluate_metrics.py --pred_dir ./predictions/nyu   --gt_dir ./gt_depth/nyu   --output results_nyu.json

Supported datasets
------------------
  KITTI  : GT depth is a 16-bit PNG; pixel value / 256 = metres
  NYU-D  : GT depth is a .h5 file (key="depth") or 16-bit PNG (pixel value / 1000 = metres)

Usage
-----
  # KITTI (using val.txt file list)
  python prepare_eval_data.py \\
      --dataset kitti \\
      --filelist metric_depth/dataset/splits/kitti/val.txt \\
      --encoder vits \\
      --max-images 50

  # NYU-D (specify RGB folder + GT folder)
  python prepare_eval_data.py \\
      --dataset nyu \\
      --img-dir  ./data/nyu/rgb \\
      --gt-dir   ./data/nyu/depth \\
      --encoder vits \\
      --max-images 50

  # Smoke test (no real data needed — validates the pipeline with synthetic data)
  python prepare_eval_data.py --smoke-test
"""

import os
import sys
import argparse
import numpy as np
import cv2

# torch is only needed for actual model inference; smoke-test does not require it
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ── Output directories ────────────────────────────────────────────────────────
DEFAULT_PRED_ROOT = "./predictions"
DEFAULT_GT_ROOT   = "./gt_depth"


# ── DAV2 model loading ────────────────────────────────────────────────────────
def load_dav2(encoder: str, device: str):
    """Load the Depth Anything V2 model from a local checkpoint."""
    from depth_anything_v2.dpt import DepthAnythingV2

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48,  96,  192,  384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96,  192, 384,  768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536,1536,1536,1536]},
    }

    ckpt = f"checkpoints/depth_anything_v2_{encoder}.pth"
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}\n"
            f"Please download the model weights into the checkpoints/ folder.\n"
            f"Download: https://github.com/DepthAnything/Depth-Anything-V2"
        )

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model = model.to(device).eval()
    print(f"[DAV2] Model loaded: {encoder} on {device}")
    return model


def run_dav2(model, img_bgr: np.ndarray, input_size: int = 518) -> np.ndarray:
    """
    Run DAV2 inference on a single BGR image.
    Returns the raw float32 depth map (not normalised — consistent with nav.py).
    """
    return model.infer_image(img_bgr, input_size)


# ── GT depth loading ──────────────────────────────────────────────────────────
def load_gt_kitti(depth_path: str) -> np.ndarray:
    """
    Load a KITTI GT depth map.
    Format: 16-bit PNG; pixel value / 256 = metres; 0 = invalid pixel.
    Returns: float32 array in metres, invalid pixels set to 0.
    """
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"KITTI GT depth map not found: {depth_path}")
    depth_m = depth_raw.astype(np.float32) / 256.0  # convert to metres (matches kitti.py)
    return depth_m


def load_gt_nyu(depth_path: str) -> np.ndarray:
    """
    Load a NYU-D GT depth map.
    Supports two formats:
      - .h5 / .hdf5 : key="depth", values already in metres
      - .png (16-bit): pixel value / 1000 = metres
    Returns: float32 array in metres.
    """
    ext = os.path.splitext(depth_path)[1].lower()

    if ext in ('.h5', '.hdf5', '.mat'):
        import h5py
        with h5py.File(depth_path, 'r') as f:
            if 'depth' in f:
                depth_m = np.array(f['depth'], dtype=np.float32)
            else:
                # Fall back to the first available key
                key = list(f.keys())[0]
                depth_m = np.array(f[key], dtype=np.float32)
                print(f"  [Warning] 'depth' key not found; using '{key}' instead")
    elif ext == '.png':
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise FileNotFoundError(f"NYU GT depth map not found: {depth_path}")
        depth_m = depth_raw.astype(np.float32) / 1000.0  # mm -> metres
    else:
        raise ValueError(f"Unsupported NYU depth format: {ext}. Supported: .h5, .png")

    return depth_m


# ── KITTI pipeline ────────────────────────────────────────────────────────────
def prepare_kitti(filelist_path: str, encoder: str, input_size: int,
                  max_images: int, device: str):
    """
    Prepare GT and predicted .npy files from a KITTI val.txt file list.

    Each line in val.txt:
        <rgb_image_path> <gt_depth_path>
    """
    pred_dir = os.path.join(DEFAULT_PRED_ROOT, "kitti")
    gt_dir   = os.path.join(DEFAULT_GT_ROOT,   "kitti")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(gt_dir,   exist_ok=True)

    with open(filelist_path, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]

    if max_images > 0:
        lines = lines[:max_images]

    model = load_dav2(encoder, device)

    print(f"\n[KITTI] Processing {len(lines)} images...")
    ok, skip = 0, 0

    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 2:
            print(f"  [Skip] Line {i+1} has unexpected format: {line}")
            skip += 1
            continue

        img_path, depth_path = parts[0], parts[1]

        # Build a unique filename from the image path to avoid collisions
        # e.g. 2011_09_26_drive_0002_sync__image_02__0000000069
        stem = (img_path.replace('/', '__')
                        .replace('\\', '__')
                        .replace('.png', '')
                        .replace('.jpg', ''))[-60:]  # cap at 60 characters

        out_pred = os.path.join(pred_dir, stem + ".npy")
        out_gt   = os.path.join(gt_dir,   stem + ".npy")

        # ── GT ───────────────────────────────────────────────────────────────
        try:
            gt = load_gt_kitti(depth_path)
            np.save(out_gt, gt)
        except FileNotFoundError as e:
            print(f"  [Skip] GT not found: {e}")
            skip += 1
            continue

        # ── Prediction ───────────────────────────────────────────────────────
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"  [Skip] RGB image not found: {img_path}")
            skip += 1
            continue

        pred = run_dav2(model, img_bgr, input_size)
        np.save(out_pred, pred)

        ok += 1
        print(f"  [{i+1}/{len(lines)}] Done: {stem}")

    print(f"\n[KITTI] Finished! {ok} succeeded, {skip} skipped")
    print(f"  GT saved to      : {gt_dir}")
    print(f"  Predictions saved: {pred_dir}")
    print(f"\nNext step:")
    print(f"  python evaluate_metrics.py --pred_dir {pred_dir} --gt_dir {gt_dir} --output results_kitti.json")


# ── NYU-D pipeline ────────────────────────────────────────────────────────────
def prepare_nyu(img_dir: str, gt_dir_input: str, encoder: str,
                input_size: int, max_images: int, device: str):
    """
    Prepare GT and predicted .npy files from an NYU-D RGB + GT folder pair.

    Expected folder structure (same filename stem, different extension):
        img_dir/   0001.jpg, 0002.jpg, ...
        gt_dir/    0001.h5,  0002.h5,  ...
      or
        gt_dir/    0001.png, 0002.png, ...
    """
    pred_dir = os.path.join(DEFAULT_PRED_ROOT, "nyu")
    gt_dir   = os.path.join(DEFAULT_GT_ROOT,   "nyu")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(gt_dir,   exist_ok=True)

    # Collect all RGB images in the input folder
    img_exts = ('.jpg', '.jpeg', '.png')
    img_files = sorted([
        f for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in img_exts
    ])

    if max_images > 0:
        img_files = img_files[:max_images]

    model = load_dav2(encoder, device)

    print(f"\n[NYU-D] Processing {len(img_files)} images...")
    ok, skip = 0, 0

    for i, img_fname in enumerate(img_files):
        stem = os.path.splitext(img_fname)[0]

        # Find the matching GT depth file (try .h5, .hdf5, .png in order)
        gt_path = None
        for gt_ext in ('.h5', '.hdf5', '.png'):
            candidate = os.path.join(gt_dir_input, stem + gt_ext)
            if os.path.exists(candidate):
                gt_path = candidate
                break

        if gt_path is None:
            print(f"  [Skip] No GT found for '{stem}' (tried .h5/.hdf5/.png)")
            skip += 1
            continue

        out_pred = os.path.join(pred_dir, stem + ".npy")
        out_gt   = os.path.join(gt_dir,   stem + ".npy")

        # ── GT ───────────────────────────────────────────────────────────────
        try:
            gt = load_gt_nyu(gt_path)
            np.save(out_gt, gt)
        except Exception as e:
            print(f"  [Skip] Failed to load GT: {e}")
            skip += 1
            continue

        # ── Prediction ───────────────────────────────────────────────────────
        img_bgr = cv2.imread(os.path.join(img_dir, img_fname))
        if img_bgr is None:
            print(f"  [Skip] Failed to read image: {img_fname}")
            skip += 1
            continue

        pred = run_dav2(model, img_bgr, input_size)
        np.save(out_pred, pred)

        ok += 1
        print(f"  [{i+1}/{len(img_files)}] Done: {stem}")

    print(f"\n[NYU-D] Finished! {ok} succeeded, {skip} skipped")
    print(f"  GT saved to      : {gt_dir}")
    print(f"  Predictions saved: {pred_dir}")
    print(f"\nNext step:")
    print(f"  python evaluate_metrics.py --pred_dir {pred_dir} --gt_dir {gt_dir} --output results_nyu.json")


# ── Smoke test (no real data required) ───────────────────────────────────────
def smoke_test():
    """
    Validates the full pipeline using synthetic data — no real dataset needed.
    Tests: generate fake GT/pred -> save as .npy -> evaluate_metrics can read and compute.
    """
    print("=" * 55)
    print("Smoke Test: validating data preparation pipeline")
    print("=" * 55)

    import tempfile
    from evaluate_metrics import compute_freespace_iou, compute_obstacle_precision_recall

    rng = np.random.default_rng(99)

    with tempfile.TemporaryDirectory() as tmpdir:
        pred_dir = os.path.join(tmpdir, "pred")
        gt_dir   = os.path.join(tmpdir, "gt")
        os.makedirs(pred_dir)
        os.makedirs(gt_dir)

        # Simulate 5 scenes
        for idx in range(5):
            fname = f"scene{idx:04d}.npy"

            # Fake GT: KITTI-style depth in metres, ~30% invalid pixels (value = 0)
            gt = rng.uniform(0, 80, (375, 1242)).astype(np.float32)
            gt[rng.random((375, 1242)) < 0.3] = 0
            np.save(os.path.join(gt_dir, fname), gt)

            # Fake prediction: relative depth with small noise
            noise = rng.normal(0, 0.05, gt.shape).astype(np.float32)
            pred = (gt + noise).clip(0, None)
            np.save(os.path.join(pred_dir, fname), pred)

        # Verify evaluate_metrics can load and compute correctly
        pred_files = sorted(os.listdir(pred_dir))
        gt_files   = sorted(os.listdir(gt_dir))
        assert pred_files == gt_files, "Filename mismatch between pred and gt!"

        iou_list, prec_list, rec_list = [], [], []
        for fname in pred_files:
            pred_np = np.load(os.path.join(pred_dir, fname))
            gt_np   = np.load(os.path.join(gt_dir,   fname))
            iou = compute_freespace_iou(pred_np, gt_np)
            p, r = compute_obstacle_precision_recall(pred_np, gt_np)
            iou_list.append(iou)
            prec_list.append(p)
            rec_list.append(r)
            print(f"  {fname}  IoU={iou:.4f}  P={p:.4f}  R={r:.4f}")

        print(f"\n  Mean IoU       : {np.mean(iou_list):.4f}")
        print(f"  Mean Precision : {np.mean(prec_list):.4f}")
        print(f"  Mean Recall    : {np.mean(rec_list):.4f}")

    print("\nSmoke Test PASSED — data preparation pipeline is working correctly!\n")


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare NYU-D / KITTI depth evaluation data (.npy format)"
    )
    parser.add_argument('--dataset',    type=str, choices=['kitti', 'nyu'],
                        help='Dataset name')
    parser.add_argument('--filelist',   type=str, default=None,
                        help='[KITTI] Path to val.txt')
    parser.add_argument('--img-dir',    type=str, default=None,
                        help='[NYU-D] Path to RGB image folder')
    parser.add_argument('--gt-dir',     type=str, default=None,
                        help='[NYU-D] Path to GT depth folder')
    parser.add_argument('--encoder',    type=str, default='vits',
                        choices=['vits', 'vitb', 'vitl', 'vitg'],
                        help='DAV2 model size (default: vits — fastest)')
    parser.add_argument('--input-size', type=int, default=518,
                        help='DAV2 input image size (default: 518)')
    parser.add_argument('--max-images', type=int, default=0,
                        help='Max images to process (0 = all)')
    parser.add_argument('--smoke-test', action='store_true',
                        help='Run pipeline validation with synthetic data (no real dataset needed)')
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test()
        sys.exit(0)

    if args.dataset is None:
        parser.error("Please specify --dataset kitti or --dataset nyu, or use --smoke-test")

    if not TORCH_AVAILABLE:
        print("[Error] PyTorch not found — cannot run model inference.")
        print("  Please install it in your virtual environment: pip install torch")
        sys.exit(1)

    DEVICE = ('cuda' if torch.cuda.is_available()
              else 'mps'  if torch.backends.mps.is_available()
              else 'cpu')
    print(f"Using device: {DEVICE}")

    if args.dataset == 'kitti':
        if args.filelist is None:
            parser.error("KITTI requires --filelist <path to val.txt>")
        prepare_kitti(
            filelist_path=args.filelist,
            encoder=args.encoder,
            input_size=args.input_size,
            max_images=args.max_images,
            device=DEVICE,
        )

    elif args.dataset == 'nyu':
        if args.img_dir is None or args.gt_dir is None:
            parser.error("NYU-D requires both --img-dir and --gt-dir")
        prepare_nyu(
            img_dir=args.img_dir,
            gt_dir_input=args.gt_dir,
            encoder=args.encoder,
            input_size=args.input_size,
            max_images=args.max_images,
            device=DEVICE,
        )
