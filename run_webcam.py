import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

from depth_anything_v2.dpt import DepthAnythingV2


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


def load_model(encoder, target_device):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    return model.to(target_device).eval()


def render_depth(depth, colormap, grayscale):
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)

    if grayscale:
        return np.repeat(depth[..., np.newaxis], 3, axis=-1)

    return (colormap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)


def combine_frames(raw_frame, depth_frame, pred_only):
    if pred_only:
        return depth_frame

    split_region = np.ones((raw_frame.shape[0], 50, 3), dtype=np.uint8) * 255
    return cv2.hconcat([raw_frame, split_region, depth_frame])


def get_filenames(img_path):
    if os.path.isfile(img_path):
        if img_path.endswith('txt'):
            with open(img_path, 'r', encoding='utf-8') as handle:
                return handle.read().splitlines()

        return [img_path]

    return glob.glob(os.path.join(img_path, '**/*'), recursive=True)


def run_on_webcam(model, input_size, colormap, grayscale, pred_only, camera_index):
    raw_video = cv2.VideoCapture(camera_index)
    if not raw_video.isOpened():
        raise RuntimeError(f'Unable to open webcam at index {camera_index}.')

    print("Starting live feed. Press 'q' to exit.")

    try:
        while raw_video.isOpened():
            ret, raw_frame = raw_video.read()
            if not ret:
                break

            depth = model.infer_image(raw_frame, input_size)
            depth_frame = render_depth(depth, colormap, grayscale)
            combined_frame = combine_frames(raw_frame, depth_frame, pred_only)

            cv2.imshow('Depth Anything V2 - Live', combined_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        raw_video.release()
        cv2.destroyAllWindows()


def run_on_files(model, input_filenames, input_size, outdir, colormap, grayscale, pred_only):
    os.makedirs(outdir, exist_ok=True)

    for k, filename in enumerate(input_filenames):
        print(f'Progress {k+1}/{len(input_filenames)}: {filename}')

        raw_image = cv2.imread(filename)
        if raw_image is None:
            print(f'Skipping unreadable file: {filename}')
            continue

        depth = model.infer_image(raw_image, input_size)
        depth_frame = render_depth(depth, colormap, grayscale)
        output_frame = combine_frames(raw_image, depth_frame, pred_only)

        cv2.imwrite(os.path.join(outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), output_frame)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')

    parser.add_argument('--img-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    parser.add_argument('--webcam', action='store_true', help='run on a live webcam feed instead of image files')
    parser.add_argument('--camera-index', type=int, default=0, help='OpenCV camera index to use with --webcam')

    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])

    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')

    args = parser.parse_args()

    if not args.webcam and not args.img_path:
        parser.error('--img-path is required unless --webcam is set')

    device = get_device()
    print(f'Initializing model on {device}...')
    depth_anything = load_model(args.encoder, device)

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    if args.webcam:
        run_on_webcam(depth_anything, args.input_size, cmap, args.grayscale, args.pred_only, args.camera_index)
    else:
        filenames = get_filenames(args.img_path)
        run_on_files(depth_anything, filenames, args.input_size, args.outdir, cmap, args.grayscale, args.pred_only)