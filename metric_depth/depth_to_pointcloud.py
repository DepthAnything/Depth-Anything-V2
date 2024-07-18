# Born out of Depth Anything V1 Issue 36: Code by @1ssb
# Make sure you have the necessary libraries
# Note that this code is meant for batch processing, to make individual predictions on different parameters, rewrite the loop execution
# Load the images you want to perform inference on in the input_images directory

import argparse
import cv2
import glob
import numpy as np
import open3d as o3d
import os
from PIL import Image
import torch

from depth_anything_v2.dpt import DepthAnythingV2

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    # Model Parameters
    parser.add_argument('--encoder', default='vitl', type=str, choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--load-from', default='checkpoints/depth_anything_v2_metric_hypersim_vitl.pth', type=str)
    parser.add_argument('--max-depth', default=10, type=float)
    
    # I/O Information
    parser.add_argument('--img-path', default='./input_images', type=str)
    parser.add_argument('--outdir', type=str, default='./vis_pointcloud')
    
    # Inference Parameters
    parser.add_argument('--focal-length-x', default=470.4, type=float, help='Focal length along the x-axis.')
    parser.add_argument('--focal-length-y', default=470.4, type=float, help='Focal length along the y-axis.')
    parser.add_argument('--final_width', default=360, type=float, help='Final Width of the images.')
    parser.add_argument('--final_height', default=640, type=float, help='Final Height of the images.')

    return parser.parse_args()

def initialize_model(args, DEVICE):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    depth_anything.load_state_dict(torch.load(args.load_from, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    return depth_anything

def get_filenames(img_path):
    if os.path.isfile(img_path):
        if img_path.endswith('txt'):
            with open(img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [img_path]
    else:
        filenames = glob.glob(os.path.join(img_path, '**/*'), recursive=True)
    return filenames

def process_images(filenames, depth_anything, args, DEVICE):
    FX, FY = args.focal_length_x, args.focal_length_y
    H, W = args.final_height, args.final_width
    
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        color_image = Image.open(filename).convert('RGB')
        image = cv2.imread(filename)
        pred = depth_anything.infer_image(image, H)
        
        resized_color_image = color_image.resize((W, H), Image.LANCZOS)
        resized_pred = Image.fromarray(pred).resize((W, H), Image.NEAREST)
        
        focal_length_x, focal_length_y = (FX, FY)
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        x = (x - W / 2) / focal_length_x
        y = (y - H / 2) / focal_length_y
        z = np.array(resized_pred)
        points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
        colors = np.array(resized_color_image).reshape(-1, 3) / 255.0
        
        save_point_cloud(points, colors, args.outdir, filename)

def save_point_cloud(points, colors, outdir, filename):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(os.path.join(outdir, os.path.splitext(os.path.basename(filename))[0] + ".ply"), pcd)

if __name__ == '__main__':
    args = parse_arguments()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    depth_anything = initialize_model(args, DEVICE)
    filenames = get_filenames(args.img_path)
    os.makedirs(args.outdir, exist_ok=True)
    process_images(filenames, depth_anything, args, DEVICE)
