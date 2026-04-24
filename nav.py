import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

from depth_anything_v2.dpt import DepthAnythingV2


# ── Navigation constants ─────────────────────────────────────────────────────
CLOSE_THRESHOLD  = 0.4   # normalised depth: pixels closer than this are obstacles
NOGO_THRESHOLD   = 0.6   # if all three region costs exceed this, STOP
# ─────────────────────────────────────────────────────────────────────────────


def compute_cost_map(binary_mask: np.ndarray) -> np.ndarray:
    """
    Given a binary obstacle mask (1 = obstacle, 0 = free),
    return a float32 cost map in [0, 1] where each pixel's cost
    is the local obstacle density within a 15-pixel radius kernel.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    cost = cv2.filter2D(binary_mask.astype(np.float32), -1, kernel.astype(np.float32))
    cost /= kernel.sum()          # normalise to [0, 1]
    return cost


def region_cost(cost_map: np.ndarray, col_start: int, col_end: int) -> float:
    """Mean obstacle cost for the horizontal slice col_start:col_end."""
    return float(cost_map[:, col_start:col_end].mean())


def steering_policy(depth_norm: np.ndarray) -> tuple[str, dict]:
    """
    Divide the depth map into three vertical thirds, build per-region cost maps,
    apply the state-machine steering policy and return the chosen action.

    Returns
    -------
    action : str  — one of 'FORWARD', 'LEFT', 'RIGHT', 'STOP'
    info   : dict — obstacle mask, cost map, per-region costs, annotated frame
    """
    h, w = depth_norm.shape

    # 1. Threshold → binary obstacle mask  (1 = close / dangerous)
    obstacle_mask = (depth_norm > CLOSE_THRESHOLD).astype(np.uint8)

    # 2. Dense cost map
    cost_map = compute_cost_map(obstacle_mask)

    # 3. Per-region scalar costs
    third = w // 3
    costs = {
        'LEFT':    region_cost(cost_map, 0,         third),
        'FORWARD': region_cost(cost_map, third,     2 * third),
        'RIGHT':   region_cost(cost_map, 2 * third, w),
    }

    # 4. State machine
    if all(v > NOGO_THRESHOLD for v in costs.values()):
        action = 'STOP'
    else:
        action = min(costs, key=costs.get)   # pick direction with lowest cost

    return action, {
        'obstacle_mask': obstacle_mask,
        'cost_map':      cost_map,
        'costs':         costs,
    }


def build_nav_visualisation(raw_image: np.ndarray,
                             depth_colour: np.ndarray,
                             info: dict,
                             action: str) -> np.ndarray:
    """
    Build a composite image with four panels:
      [original | depth map | obstacle mask | cost map]
    plus a large action banner at the bottom.
    """
    h, w = raw_image.shape[:2]
    third = w // 3

    # -- Obstacle mask (3-channel) --
    obs_vis = (info['obstacle_mask'] * 255).astype(np.uint8)
    obs_vis = cv2.cvtColor(obs_vis, cv2.COLOR_GRAY2BGR)
    # red tint for obstacles
    obs_vis[info['obstacle_mask'] == 1] = [0, 0, 220]

    # -- Cost map (heat-map) --
    cost_uint8 = (info['cost_map'] * 255).clip(0, 255).astype(np.uint8)
    cost_vis   = cv2.applyColorMap(cost_uint8, cv2.COLORMAP_JET)

    # -- Draw region dividers and cost labels on cost_vis --
    for panel in (obs_vis, cost_vis):
        cv2.line(panel, (third,     0), (third,     h), (255, 255, 255), 1)
        cv2.line(panel, (2 * third, 0), (2 * third, h), (255, 255, 255), 1)

    for label, (col, cost) in zip(
            ['L', 'C', 'R'],
            [(third // 2,             info['costs']['LEFT']),
             (third + third // 2,     info['costs']['FORWARD']),
             (2 * third + third // 2, info['costs']['RIGHT'])]):
        cv2.putText(cost_vis, f'{label}:{cost:.2f}',
                    (col - 22, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # -- Action banner --
    COLOURS = {
        'FORWARD': (0, 200, 80),
        'LEFT':    (30, 160, 255),
        'RIGHT':   (30, 160, 255),
        'STOP':    (0, 0, 230),
    }
    banner_h = 44
    banner   = np.zeros((banner_h, w * 4, 3), dtype=np.uint8)
    banner[:] = COLOURS[action]
    text = f'ACTION: {action}  |  L={info["costs"]["LEFT"]:.2f}  ' \
           f'C={info["costs"]["FORWARD"]:.2f}  R={info["costs"]["RIGHT"]:.2f}'
    cv2.putText(banner, text, (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        # -- Stitch panels horizontally --
    sep = np.ones((h, 8, 3), dtype=np.uint8) * 200
    row = cv2.hconcat([raw_image, sep, depth_colour, sep, obs_vis, sep, cost_vis])

    # -- Action banner (width matches row exactly) --
    banner_h = 44
    banner   = np.zeros((banner_h, row.shape[1], 3), dtype=np.uint8)  # ← fix here
    banner[:] = COLOURS[action]
    text = f'ACTION: {action}  |  L={info["costs"]["LEFT"]:.2f}  ' \
        f'C={info["costs"]["FORWARD"]:.2f}  R={info["costs"]["RIGHT"]:.2f}'
    cv2.putText(banner, text, (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    return cv2.vconcat([row, banner])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 + Navigation')

    parser.add_argument('--img-path',    type=str)
    parser.add_argument('--input-size',  type=int, default=518)
    parser.add_argument('--outdir',      type=str, default='./vis_depth')
    parser.add_argument('--encoder',     type=str, default='vitl',
                        choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--pred-only',   dest='pred_only',  action='store_true')
    parser.add_argument('--grayscale',   dest='grayscale',  action='store_true')
    parser.add_argument('--nav',         dest='nav',        action='store_true',
                        help='overlay navigation cost maps and steering decision')
    parser.add_argument('--close-thresh', type=float, default=CLOSE_THRESHOLD,
                        help='normalised depth below which a pixel is an obstacle')
    parser.add_argument('--nogo-thresh',  type=float, default=NOGO_THRESHOLD,
                        help='all-region cost above which the robot stops')

    args = parser.parse_args()

    # Allow CLI overrides of the module-level constants
    CLOSE_THRESHOLD = args.close_thresh
    NOGO_THRESHOLD  = args.nogo_thresh

    DEVICE = ('cuda' if torch.cuda.is_available()
              else 'mps' if torch.backends.mps.is_available()
              else 'cpu')

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48,  96,  192,  384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96,  192, 384,  768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536,1536,1536, 1536]},
    }

    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(
        torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth',
                   map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    if os.path.isfile(args.img_path):
        filenames = (open(args.img_path).read().splitlines()
                     if args.img_path.endswith('txt')
                     else [args.img_path])
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)

    os.makedirs(args.outdir, exist_ok=True)
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')

        raw_image = cv2.imread(filename)

        # ── Depth inference ──────────────────────────────────────────────────
        depth = depth_anything.infer_image(raw_image, args.input_size)

        depth_norm = (depth - depth.min()) / (depth.max() - depth.min())  # [0,1]
        depth_uint8 = (depth_norm * 255).astype(np.uint8)

        if args.grayscale:
            depth_colour = np.repeat(depth_uint8[..., np.newaxis], 3, axis=-1)
        else:
            depth_colour = (cmap(depth_uint8)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        stem    = os.path.splitext(os.path.basename(filename))[0]
        outpath = os.path.join(args.outdir, stem + '.png')

        # ── Navigation overlay ───────────────────────────────────────────────
        if args.nav:
            action, info = steering_policy(depth_norm)
            print(f'  Action: {action}  costs={info["costs"]}')

            nav_frame = build_nav_visualisation(raw_image, depth_colour, info, action)
            cv2.imwrite(outpath, nav_frame)

        # ── Original output modes ────────────────────────────────────────────
        elif args.pred_only:
            cv2.imwrite(outpath, depth_colour)
        else:
            sep      = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined = cv2.hconcat([raw_image, sep, depth_colour])
            cv2.imwrite(outpath, combined)