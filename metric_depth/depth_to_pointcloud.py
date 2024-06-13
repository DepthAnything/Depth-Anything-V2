# Born out of Depth Anything V1 Issue 36
# Make sure you have the necessary libraries
# Code by @1ssb

import argparse
import cv2
import glob
import numpy as np
import open3d as o3d
import os
from PIL import Image
import torch

from depth_anything_v2.dpt import DepthAnythingV2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', default='vitl', type=str, choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--load-from', default='', type=str)
    parser.add_argument('--max-depth', default=20, type=float)
    
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--outdir', type=str, default='./vis_pointcloud')
    
    args = parser.parse_args()
    
    # Global settings
    FL = 715.0873
    FY = 784 * 0.6
    FX = 784 * 0.6
    NYU_DATA = False
    FINAL_HEIGHT = 518
    FINAL_WIDTH = 518
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    depth_anything.load_state_dict(torch.load(args.load_from, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
    
    os.makedirs(args.outdir, exist_ok=True)
    
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        color_image = Image.open(filename).convert('RGB')
        
        image = cv2.imread(filename)
        pred = depth_anything.infer_image(image, FINAL_HEIGHT)
        
        # Resize color image and depth to final size
        resized_color_image = color_image.resize((FINAL_WIDTH, FINAL_HEIGHT), Image.LANCZOS)
        resized_pred = Image.fromarray(pred).resize((FINAL_WIDTH, FINAL_HEIGHT), Image.NEAREST)
        
        focal_length_x, focal_length_y = (FX, FY) if not NYU_DATA else (FL, FL)
        x, y = np.meshgrid(np.arange(FINAL_WIDTH), np.arange(FINAL_HEIGHT))
        x = (x - FINAL_WIDTH / 2) / focal_length_x
        y = (y - FINAL_HEIGHT / 2) / focal_length_y
        z = np.array(resized_pred)
        points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
        colors = np.array(resized_color_image).reshape(-1, 3) / 255.0
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + ".ply"), pcd)