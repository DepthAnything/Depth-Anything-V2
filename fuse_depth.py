import numpy as np
import matplotlib.pyplot as plt
import csv
import cv2
import torch
from datetime import datetime
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim
from depth_anything_v2.dpt import DepthAnythingV2

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

def parse_realsense_depth(depth_file_path, metadata_path):
    metadata = read_metadata(metadata_path)
    width = int(metadata['Resolution x'])
    height = int(metadata['Resolution y'])
    depth_data = np.fromfile(depth_file_path, dtype=np.uint16)
    depth_frame = depth_data.reshape((height, width))
    depth_frame_meters = depth_frame.astype(np.float32) / 1000.0
    return depth_frame_meters

def compute_depth_confidence(depth, valid_range=(0.2, 4.0)):
    """Compute confidence map for depth measurements"""
    # Basic confidence based on valid range
    valid_mask = (depth > valid_range[0]) & (depth < valid_range[1])
    
    # Confidence decreases with distance
    distance_confidence = np.exp(-depth / valid_range[1])
    
    # Edge confidence (depth discontinuities are less reliable)
    depth_grad_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
    depth_grad_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)
    edge_confidence = np.exp(-np.sqrt(depth_grad_x**2 + depth_grad_y**2))
    
    # Combine confidences
    confidence = valid_mask * distance_confidence * edge_confidence
    return confidence

def scale_depth_anything(d455_depth, da_depth, valid_range=(0.2, 4.0)):
    """Improved depth scaling with confidence weighting"""
    d455_valid = (d455_depth > valid_range[0]) & (d455_depth < valid_range[1])
    if np.sum(d455_valid) < 100:
        return da_depth
    
    # Compute confidence maps
    d455_confidence = compute_depth_confidence(d455_depth, valid_range)
    
    # Weighted linear regression
    d455_valid_depth = d455_depth[d455_valid]
    da_valid_depth = da_depth[d455_valid]
    weights = d455_confidence[d455_valid]
    
    # Weighted statistics
    weighted_mean_d455 = np.average(d455_valid_depth, weights=weights)
    weighted_mean_da = np.average(da_valid_depth, weights=weights)
    
    # Weighted covariance
    weighted_cov = np.average((d455_valid_depth - weighted_mean_d455) * 
                            (da_valid_depth - weighted_mean_da), weights=weights)
    weighted_var_da = np.average((da_valid_depth - weighted_mean_da)**2, weights=weights)
    
    # Scale parameters
    a = weighted_cov / (weighted_var_da + 1e-6)
    b = weighted_mean_d455 - a * weighted_mean_da
    
    return da_depth * a + b

def fuse_depth_advanced(d455_depth, da_depth, valid_range=(0.2, 4.0)):
    """Advanced fusion with confidence weighting and edge preservation"""
    # Compute confidence maps
    d455_confidence = compute_depth_confidence(d455_depth, valid_range)
    da_confidence = compute_depth_confidence(da_depth, valid_range)
    
    # Normalize confidences
    total_confidence = d455_confidence + da_confidence + 1e-6
    d455_weight = d455_confidence / total_confidence
    da_weight = da_confidence / total_confidence
    
    # Edge detection
    d455_edges = cv2.Canny((d455_depth * 255).astype(np.uint8), 50, 150)
    da_edges = cv2.Canny((da_depth * 255).astype(np.uint8), 50, 150)
    
    # Edge-aware fusion
    edge_mask = (d455_edges > 0) | (da_edges > 0)
    edge_mask = gaussian_filter(edge_mask.astype(float), sigma=1.0)
    
    # Combine weights
    final_d455_weight = d455_weight * (1 - edge_mask) + edge_mask * (d455_confidence > da_confidence)
    final_da_weight = 1 - final_d455_weight
    
    # Fuse depths
    fused_depth = d455_depth * final_d455_weight + da_depth * final_da_weight
    
    # Post-processing
    fused_depth = gaussian_filter(fused_depth, sigma=0.5)  # Slight smoothing
    
    return fused_depth

def evaluate_fusion(d455_depth, da_depth, fused_depth, valid_range=(0.2, 4.0)):
    """Evaluate fusion quality"""
    valid_mask = (d455_depth > valid_range[0]) & (d455_depth < valid_range[1])
    
    # Compute metrics
    mae_d455 = np.mean(np.abs(d455_depth[valid_mask] - fused_depth[valid_mask]))
    mae_da = np.mean(np.abs(da_depth[valid_mask] - fused_depth[valid_mask]))
    
    # Get data range for SSIM
    data_range = max(d455_depth[valid_mask].max(), da_depth[valid_mask].max(), fused_depth[valid_mask].max())
    
    # Structural similarity with data_range parameter
    ssim_d455 = ssim(d455_depth[valid_mask], fused_depth[valid_mask], data_range=data_range)
    ssim_da = ssim(da_depth[valid_mask], fused_depth[valid_mask], data_range=data_range)
    
    print(f"Fusion Evaluation:")
    print(f"MAE with D455: {mae_d455:.4f} meters")
    print(f"MAE with Depth-Anything: {mae_da:.4f} meters")
    print(f"SSIM with D455: {ssim_d455:.4f}")
    print(f"SSIM with Depth-Anything: {ssim_da:.4f}")

if __name__ == "__main__":
    # Paths
    depth_file_path = "assets/examples/realsense_Depth.raw"
    metadata_path = "assets/examples/realsense_Depth_metadata.csv"
    rgb_img_path = "assets/examples/realsense_Color.png"
    encoder = 'vits'  # Changed to match the checkpoint
    input_size = 518
    
    # 1. Read D455 depth
    d455_depth = parse_realsense_depth(depth_file_path, metadata_path)
    
    # 2. Read RGB image
    raw_image = cv2.imread(rgb_img_path)
    if raw_image is None:
        raise FileNotFoundError(f"Cannot find RGB image: {rgb_img_path}")
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    
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
    
    # Inference and normalization (matching run.py)
    da_depth = depth_anything.infer_image(raw_image, input_size)
    da_depth = da_depth.astype(np.float32)
    da_depth = cv2.resize(da_depth, (d455_depth.shape[1], d455_depth.shape[0]), interpolation=cv2.INTER_LINEAR)

    
    # Resize to match D455 resolution
    da_depth_scaled = scale_depth_anything(d455_depth, da_depth)
    
    # 5. Advanced fusion
    fused_depth = fuse_depth_advanced(d455_depth, da_depth_scaled)
    
    # 6. Evaluation
    evaluate_fusion(d455_depth, da_depth_scaled, fused_depth)
    
    # 7. Visualization
    vmin, vmax = 0.2, 4.0
    plt.figure(figsize=(18, 6))
    plt.subplot(131)
    plt.imshow(d455_depth, cmap='jet', vmin=vmin, vmax=vmax)
    plt.title("D455 Depth (m)")
    plt.colorbar()
    plt.subplot(132)
    plt.imshow(da_depth_scaled, cmap='jet', vmin=vmin, vmax=vmax)
    plt.title("Depth-Anything V2 (m, scaled)")
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(fused_depth, cmap='jet', vmin=vmin, vmax=vmax)
    plt.title("Fused Depth (m)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('fused_depth_advanced.png')
    plt.close()
    
    print("Fused depth saved as fused_depth_advanced.png") 