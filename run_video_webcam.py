import argparse
import cv2
import matplotlib
import numpy as np
import torch
from depth_anything_v2.dpt import DepthAnythingV2


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 - Live Feed')

    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--pred-only', dest='pred_only', action='store_true')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true')
    parser.add_argument('--camera-index', type=int, default=0)

    args = parser.parse_args()

    DEVICE = get_device()

    # Reverted to your original configuration
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    print(f"Initializing model on {DEVICE}...")
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    # Initialize webcam
    raw_video = cv2.VideoCapture(args.camera_index)
    if not raw_video.isOpened():
        raise RuntimeError(f'Unable to open webcam at index {args.camera_index}.')

    margin_width = 50
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    print("Starting live feed. Press 'q' to exit.")

    while raw_video.isOpened():
        ret, raw_frame = raw_video.read()
        if not ret:
            break

        # Inference
        depth = depth_anything.infer_image(raw_frame, args.input_size)

        # Normalize and colorize based on your original logic
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)

        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        # Combine or isolate frames
        if args.pred_only:
            combined_frame = depth
        else:
            frame_height = raw_frame.shape[0]
            split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
            combined_frame = cv2.hconcat([raw_frame, split_region, depth])

        # Display window
        cv2.imshow('Depth Anything V2 - Live', combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    raw_video.release()
    cv2.destroyAllWindows()