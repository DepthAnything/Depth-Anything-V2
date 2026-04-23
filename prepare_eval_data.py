"""
prepare_eval_data.py
--------------------
準備 evaluate_metrics.py 所需的 .npy 深度圖資料。

這個腳本做兩件事：
  1. 把 NYU-D 或 KITTI 的 GT 深度圖轉成 .npy 格式，存到 gt_depth/ 資料夾
  2. 用 DAV2 對對應的 RGB 圖片跑推論，把預測深度也存成 .npy，存到 predictions/ 資料夾

完成後直接用 evaluate_metrics.py 評估：
  python evaluate_metrics.py --pred_dir ./predictions/kitti --gt_dir ./gt_depth/kitti --output results_kitti.json
  python evaluate_metrics.py --pred_dir ./predictions/nyu   --gt_dir ./gt_depth/nyu   --output results_nyu.json

支援的資料集
-----------
  KITTI  : GT 深度圖是 16-bit PNG，像素值除以 256 = 公尺
  NYU-D  : GT 深度圖是 .h5 檔（key="depth"）或 16-bit PNG（像素值除以 1000 = 公尺）

使用方式
--------
  # KITTI（用 val.txt 檔案清單）
  python prepare_eval_data.py \\
      --dataset kitti \\
      --filelist metric_depth/dataset/splits/kitti/val.txt \\
      --encoder vits \\
      --max-images 50

  # NYU-D（指定圖片資料夾 + GT 資料夾）
  python prepare_eval_data.py \\
      --dataset nyu \\
      --img-dir  ./data/nyu/rgb \\
      --gt-dir   ./data/nyu/depth \\
      --encoder vits \\
      --max-images 50

  # Smoke test（不需要任何資料，用假資料驗證流程是否正確）
  python prepare_eval_data.py --smoke-test
"""

import os
import sys
import argparse
import numpy as np
import cv2

# torch 只在實際跑模型推論時才需要，smoke-test 不需要
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ── 輸出資料夾 ────────────────────────────────────────────────────────────────
DEFAULT_PRED_ROOT = "./predictions"
DEFAULT_GT_ROOT   = "./gt_depth"


# ── DAV2 模型載入 ─────────────────────────────────────────────────────────────
def load_dav2(encoder: str, device: str):
    """載入 Depth Anything V2 模型。"""
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
            f"找不到 checkpoint：{ckpt}\n"
            f"請先下載模型放到 checkpoints/ 資料夾。\n"
            f"下載連結：https://github.com/DepthAnything/Depth-Anything-V2"
        )

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model = model.to(device).eval()
    print(f"[DAV2] 載入模型成功：{encoder}，裝置：{device}")
    return model


def run_dav2(model, img_bgr: np.ndarray, input_size: int = 518) -> np.ndarray:
    """
    對單張 BGR 圖片跑 DAV2 推論。
    回傳：原始 float32 深度圖（未正規化，和 nav.py 一致）
    """
    return model.infer_image(img_bgr, input_size)


# ── GT 深度讀取 ───────────────────────────────────────────────────────────────
def load_gt_kitti(depth_path: str) -> np.ndarray:
    """
    讀取 KITTI GT 深度圖。
    格式：16-bit PNG，像素值 / 256 = 公尺，0 = 無效像素。
    回傳：float32，單位公尺，無效像素為 0。
    """
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"找不到 KITTI GT 深度圖：{depth_path}")
    depth_m = depth_raw.astype(np.float32) / 256.0  # 轉公尺（kitti.py 的做法）
    return depth_m


def load_gt_nyu(depth_path: str) -> np.ndarray:
    """
    讀取 NYU-D GT 深度圖。
    支援兩種格式：
      - .h5 / .hdf5：key="depth"，值直接是公尺
      - .png（16-bit）：像素值 / 1000 = 公尺
    回傳：float32，單位公尺。
    """
    ext = os.path.splitext(depth_path)[1].lower()

    if ext in ('.h5', '.hdf5', '.mat'):
        import h5py
        with h5py.File(depth_path, 'r') as f:
            # NYU-D HDF5 的 key 通常是 'depth'
            if 'depth' in f:
                depth_m = np.array(f['depth'], dtype=np.float32)
            else:
                # 嘗試第一個 key
                key = list(f.keys())[0]
                depth_m = np.array(f[key], dtype=np.float32)
                print(f"  [警告] 找不到 'depth' key，改用 '{key}'")
    elif ext == '.png':
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise FileNotFoundError(f"找不到 NYU GT 深度圖：{depth_path}")
        depth_m = depth_raw.astype(np.float32) / 1000.0  # mm → 公尺
    else:
        raise ValueError(f"不支援的 NYU 深度格式：{ext}，支援 .h5 / .png")

    return depth_m


# ── KITTI 主流程 ──────────────────────────────────────────────────────────────
def prepare_kitti(filelist_path: str, encoder: str, input_size: int,
                  max_images: int, device: str):
    """
    根據 KITTI val.txt 準備 GT 和預測的 .npy 檔案。

    val.txt 每行格式：
        <rgb_path> <gt_depth_path>
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

    print(f"\n[KITTI] 開始處理 {len(lines)} 張圖片...")
    ok, skip = 0, 0

    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 2:
            print(f"  [跳過] 第 {i+1} 行格式錯誤：{line}")
            skip += 1
            continue

        img_path, depth_path = parts[0], parts[1]

        # 用圖片的 stem 當檔名，避免同名衝突
        # 例：2011_09_26_drive_0002_sync__image_02__0000000069
        stem = (img_path.replace('/', '__')
                        .replace('\\', '__')
                        .replace('.png', '')
                        .replace('.jpg', ''))[-60:]  # 最多 60 字元

        out_pred = os.path.join(pred_dir, stem + ".npy")
        out_gt   = os.path.join(gt_dir,   stem + ".npy")

        # ── GT ──────────────────────────────────────────────────────────────
        try:
            gt = load_gt_kitti(depth_path)
            np.save(out_gt, gt)
        except FileNotFoundError as e:
            print(f"  [跳過] GT 不存在：{e}")
            skip += 1
            continue

        # ── 預測 ─────────────────────────────────────────────────────────────
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"  [跳過] RGB 圖片不存在：{img_path}")
            skip += 1
            continue

        pred = run_dav2(model, img_bgr, input_size)
        np.save(out_pred, pred)

        ok += 1
        print(f"  [{i+1}/{len(lines)}] 完成：{stem}")

    print(f"\n[KITTI] 完成！成功 {ok} 張，跳過 {skip} 張")
    print(f"  GT 儲存於：{gt_dir}")
    print(f"  預測儲存於：{pred_dir}")
    print(f"\n下一步：")
    print(f"  python evaluate_metrics.py --pred_dir {pred_dir} --gt_dir {gt_dir} --output results_kitti.json")


# ── NYU-D 主流程 ──────────────────────────────────────────────────────────────
def prepare_nyu(img_dir: str, gt_dir_input: str, encoder: str,
                input_size: int, max_images: int, device: str):
    """
    根據圖片資料夾 + GT 資料夾準備 NYU-D 的 .npy 檔案。

    預期資料夾結構（圖片和 GT 同名，副檔名不同）：
        img_dir/   0001.jpg, 0002.jpg ...
        gt_dir/    0001.h5,  0002.h5  ...
      或
        gt_dir/    0001.png, 0002.png ...
    """
    pred_dir = os.path.join(DEFAULT_PRED_ROOT, "nyu")
    gt_dir   = os.path.join(DEFAULT_GT_ROOT,   "nyu")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(gt_dir,   exist_ok=True)

    # 找出所有 RGB 圖片
    img_exts = ('.jpg', '.jpeg', '.png')
    img_files = sorted([
        f for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in img_exts
    ])

    if max_images > 0:
        img_files = img_files[:max_images]

    model = load_dav2(encoder, device)

    print(f"\n[NYU-D] 開始處理 {len(img_files)} 張圖片...")
    ok, skip = 0, 0

    for i, img_fname in enumerate(img_files):
        stem = os.path.splitext(img_fname)[0]

        # 尋找對應的 GT 深度圖（支援 .h5 / .hdf5 / .png）
        gt_path = None
        for gt_ext in ('.h5', '.hdf5', '.png'):
            candidate = os.path.join(gt_dir_input, stem + gt_ext)
            if os.path.exists(candidate):
                gt_path = candidate
                break

        if gt_path is None:
            print(f"  [跳過] 找不到 GT：{stem}（試過 .h5/.hdf5/.png）")
            skip += 1
            continue

        out_pred = os.path.join(pred_dir, stem + ".npy")
        out_gt   = os.path.join(gt_dir,   stem + ".npy")

        # ── GT ──────────────────────────────────────────────────────────────
        try:
            gt = load_gt_nyu(gt_path)
            np.save(out_gt, gt)
        except Exception as e:
            print(f"  [跳過] GT 讀取失敗：{e}")
            skip += 1
            continue

        # ── 預測 ─────────────────────────────────────────────────────────────
        img_bgr = cv2.imread(os.path.join(img_dir, img_fname))
        if img_bgr is None:
            print(f"  [跳過] 圖片讀取失敗：{img_fname}")
            skip += 1
            continue

        pred = run_dav2(model, img_bgr, input_size)
        np.save(out_pred, pred)

        ok += 1
        print(f"  [{i+1}/{len(img_files)}] 完成：{stem}")

    print(f"\n[NYU-D] 完成！成功 {ok} 張，跳過 {skip} 張")
    print(f"  GT 儲存於：{gt_dir}")
    print(f"  預測儲存於：{pred_dir}")
    print(f"\n下一步：")
    print(f"  python evaluate_metrics.py --pred_dir {pred_dir} --gt_dir {gt_dir} --output results_nyu.json")


# ── Smoke Test（不需要任何真實資料）─────────────────────────────────────────
def smoke_test():
    """
    用假資料驗證整個流程是否正確，不需要真實資料集。
    測試：GT 讀取 → 儲存 .npy → evaluate_metrics 可以讀取並計算。
    """
    print("=" * 55)
    print("Smoke Test：驗證資料準備流程")
    print("=" * 55)

    import tempfile
    from evaluate_metrics import compute_freespace_iou, compute_obstacle_precision_recall

    rng = np.random.default_rng(99)

    with tempfile.TemporaryDirectory() as tmpdir:
        pred_dir = os.path.join(tmpdir, "pred")
        gt_dir   = os.path.join(tmpdir, "gt")
        os.makedirs(pred_dir)
        os.makedirs(gt_dir)

        # 模擬 5 張圖
        for idx in range(5):
            fname = f"scene{idx:04d}.npy"

            # 模擬 GT（KITTI 風格：公尺，有些像素是 0 無效）
            gt = rng.uniform(0, 80, (375, 1242)).astype(np.float32)
            gt[rng.random((375, 1242)) < 0.3] = 0  # 30% 無效像素
            np.save(os.path.join(gt_dir, fname), gt)

            # 模擬 DAV2 預測（相對深度，帶一點誤差）
            noise = rng.normal(0, 0.05, gt.shape).astype(np.float32)
            pred = gt + noise
            pred = pred.clip(0, None)
            np.save(os.path.join(pred_dir, fname), pred)

        # 驗證 evaluate_metrics 可以讀取並計算
        pred_files = sorted(os.listdir(pred_dir))
        gt_files   = sorted(os.listdir(gt_dir))
        assert pred_files == gt_files, "檔名不對應！"

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

    print("\nSmoke Test PASSED — 資料準備流程正常！\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="準備 NYU-D / KITTI 深度評估資料（.npy 格式）"
    )
    parser.add_argument('--dataset',    type=str, choices=['kitti', 'nyu'],
                        help='資料集名稱')
    parser.add_argument('--filelist',   type=str, default=None,
                        help='[KITTI] val.txt 的路徑')
    parser.add_argument('--img-dir',    type=str, default=None,
                        help='[NYU-D] RGB 圖片資料夾')
    parser.add_argument('--gt-dir',     type=str, default=None,
                        help='[NYU-D] GT 深度圖資料夾')
    parser.add_argument('--encoder',    type=str, default='vits',
                        choices=['vits', 'vitb', 'vitl', 'vitg'],
                        help='DAV2 模型大小（預設 vits，最快）')
    parser.add_argument('--input-size', type=int, default=518,
                        help='DAV2 輸入圖片大小（預設 518）')
    parser.add_argument('--max-images', type=int, default=0,
                        help='最多處理幾張圖（0 = 全部）')
    parser.add_argument('--smoke-test', action='store_true',
                        help='用假資料跑流程驗證，不需要真實資料集')
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test()
        sys.exit(0)

    if args.dataset is None:
        parser.error("請指定 --dataset kitti 或 --dataset nyu，或用 --smoke-test 測試")

    if not TORCH_AVAILABLE:
        print("[錯誤] 找不到 PyTorch，無法跑模型推論。")
        print("  請在你的專案虛擬環境裡安裝：pip install torch")
        sys.exit(1)

    DEVICE = ('cuda' if torch.cuda.is_available()
              else 'mps'  if torch.backends.mps.is_available()
              else 'cpu')
    print(f"使用裝置：{DEVICE}")

    if args.dataset == 'kitti':
        if args.filelist is None:
            parser.error("KITTI 需要 --filelist <val.txt 路徑>")
        prepare_kitti(
            filelist_path=args.filelist,
            encoder=args.encoder,
            input_size=args.input_size,
            max_images=args.max_images,
            device=DEVICE,
        )

    elif args.dataset == 'nyu':
        if args.img_dir is None or args.gt_dir is None:
            parser.error("NYU-D 需要 --img-dir 和 --gt-dir")
        prepare_nyu(
            img_dir=args.img_dir,
            gt_dir_input=args.gt_dir,
            encoder=args.encoder,
            input_size=args.input_size,
            max_images=args.max_images,
            device=DEVICE,
        )
