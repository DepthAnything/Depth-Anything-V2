import numpy as np
import cv2
import torch
import open3d as o3d
import csv
from PIL import Image
import matplotlib.pyplot as plt
from depth_anything_v2.dpt import DepthAnythingV2
from fuse_depth import parse_realsense_depth, scale_depth_anything, fuse_depth_advanced
import os

# Set Open3D visualization backend
o3d.visualization.webrtc_server.enable_webrtc = False
o3d.visualization.gui.Application.instance.initialize()

def read_metadata(metadata_path):
    metadata = {}
    with open(metadata_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                key = row[0].strip()
                value = row[1].strip()
                metadata[key] = value
    return metadata

def depth_to_pointcloud(depth, rgb_image, focal_length_x, focal_length_y):
    """Convert depth map to point cloud"""
    height, width = depth.shape
    
    # Resize RGB image to match depth map dimensions
    rgb_image = cv2.resize(rgb_image, (width, height))
    
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x = (x - width / 2) / focal_length_x
    y = (y - height / 2) / focal_length_y
    z = depth
    points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
    colors = np.array(rgb_image).reshape(-1, 3) / 255.0
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def visualize_pointclouds(pcd1, pcd2, pcd3, titles):
    """Save point clouds to PLY files and provide viewing instructions"""
    # Create output directory if it doesn't exist
    os.makedirs('pointcloud_comparison', exist_ok=True)
    
    # Save point clouds to PLY files
    pcd1_path = 'pointcloud_comparison/realsense.ply'
    pcd2_path = 'pointcloud_comparison/depth_anything.ply'
    pcd3_path = 'pointcloud_comparison/fused.ply'
    
    o3d.io.write_point_cloud(pcd1_path, pcd1)
    o3d.io.write_point_cloud(pcd2_path, pcd2)
    o3d.io.write_point_cloud(pcd3_path, pcd3)
    
    print("\nPoint clouds have been saved to PLY files.")
    print("To view them, run the following commands in separate terminals:")
    print(f"python view_ply.py {pcd1_path}  # {titles[0]}")
    print(f"python view_ply.py {pcd2_path}  # {titles[1]}")
    print(f"python view_ply.py {pcd3_path}  # {titles[2]}")
    print("\nYou can open multiple terminals to compare the point clouds side by side.")

def main():
    # Paths
    depth_file_path = "assets/examples/DC-510/kitchen/kitchen_5_Depth.raw"
    metadata_path = "assets/examples/DC-510/kitchen/kitchen_5_Depth_metadata.csv"
    rgb_img_path = "assets/examples/DC-510/kitchen/kitchen_5_Color.png"
    encoder = 'vits'
    
    # Read metadata for focal lengths
    metadata = read_metadata(metadata_path)
    focal_length_x = float(metadata.get('Fx', 216.249817))  # Changed to match metadata file
    focal_length_y = float(metadata.get('Fy', 216.249817))  # Changed to match metadata file
    
    # 1. Read D455 depth
    d455_depth = parse_realsense_depth(depth_file_path, metadata_path)
    
    # 2. Read RGB image
    rgb_image = cv2.imread(rgb_img_path)
    if rgb_image is None:
        raise FileNotFoundError(f"Cannot find RGB image: {rgb_img_path}")
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    
    # 3. Depth-Anything-V2 inference
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    # Get Depth-Anything prediction
    da_depth = depth_anything.infer_image(rgb_image, d455_depth.shape[0])
    da_depth = da_depth.astype(np.float32)
    da_depth = cv2.resize(da_depth, (d455_depth.shape[1], d455_depth.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # Scale Depth-Anything depth
    da_depth_scaled = scale_depth_anything(d455_depth, da_depth)
    
    # Fuse depths
    fused_depth = fuse_depth_advanced(d455_depth, da_depth_scaled)
    
    # Generate point clouds
    pcd_d455 = depth_to_pointcloud(d455_depth, rgb_image, focal_length_x, focal_length_y)
    pcd_da = depth_to_pointcloud(da_depth_scaled, rgb_image, focal_length_x, focal_length_y)
    pcd_fused = depth_to_pointcloud(fused_depth, rgb_image, focal_length_x, focal_length_y)
    
    # Visualize point clouds
    titles = ["RealSense D455", "Depth-Anything-V2", "Fused Depth"]
    visualize_pointclouds(pcd_d455, pcd_da, pcd_fused, titles)

if __name__ == "__main__":
    main() 