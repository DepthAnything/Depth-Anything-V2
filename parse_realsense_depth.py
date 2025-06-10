import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import cv2

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

def colorize_depth(depth_frame, clipping_distance=0.0):
    """
    Colorize depth frame using RealSense's standard colorization approach
    """
    # Convert to float32 for processing
    depth_frame = depth_frame.astype(np.float32)
    
    # Create a mask for invalid depth values
    invalid_mask = depth_frame == 0
    
    # Normalize depth values to 0-1 range
    if clipping_distance > 0:
        depth_frame = np.clip(depth_frame, 0, clipping_distance)
    
    # Normalize to 0-1 range
    depth_min = np.min(depth_frame[~invalid_mask])
    depth_max = np.max(depth_frame[~invalid_mask])
    depth_frame = (depth_frame - depth_min) / (depth_max - depth_min)
    
    # Apply colormap (using RealSense's default colormap)
    colored = cv2.applyColorMap((depth_frame * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Set invalid pixels to black
    colored[invalid_mask] = [0, 0, 0]
    
    return colored

def parse_realsense_depth(depth_file_path, metadata_path):
    # Read metadata
    metadata = read_metadata(metadata_path)
    
    # Get dimensions from metadata
    width = int(metadata['Resolution x'])
    height = int(metadata['Resolution y'])
    bytes_per_pixel = int(metadata['Bytes per pixel'])
    frame_number = metadata['Frame Number']
    timestamp = float(metadata['Timestamp (ms)'])
    
    # Convert timestamp to readable format
    timestamp_dt = datetime.fromtimestamp(timestamp / 1000.0)
    
    # Read the raw depth data
    depth_data = np.fromfile(depth_file_path, dtype=np.uint16)
    
    # Reshape the data to the correct resolution
    depth_frame = depth_data.reshape((height, width))
    
    # Convert to meters (RealSense depth is in millimeters)
    depth_frame_meters = depth_frame.astype(np.float32) / 1000.0
    
    # Calculate average distance (excluding zero values which are invalid measurements)
    valid_mask = depth_frame_meters > 0
    avg_distance = np.mean(depth_frame_meters[valid_mask])
    
    # Print frame information
    print(f"Frame Information:")
    print(f"Frame Number: {frame_number}")
    print(f"Timestamp: {timestamp_dt}")
    print(f"Resolution: {width}x{height}")
    print(f"Format: {metadata['Format']}")
    print(f"Bytes per pixel: {bytes_per_pixel}")
    print("\nDepth Statistics:")
    print(f"Average distance: {avg_distance:.3f} meters")
    print(f"Min distance: {np.min(depth_frame_meters[valid_mask]):.3f} meters")
    print(f"Max distance: {np.max(depth_frame_meters[valid_mask]):.3f} meters")
    
    # Print camera intrinsics
    print("\nCamera Intrinsics:")
    print(f"Focal Length (Fx, Fy): ({metadata['Fx']}, {metadata['Fy']})")
    print(f"Principal Point (PPx, PPy): ({metadata['PPx']}, {metadata['PPy']})")
    print(f"Distortion Model: {metadata['Distorsion']}")
    
    # Create multiple visualizations
    plt.figure(figsize=(15, 5))
    
    # 1. Raw depth visualization
    plt.subplot(131)
    plt.imshow(depth_frame_meters, cmap='jet')
    plt.colorbar(label='Distance (meters)')
    plt.title('Raw Depth Map')
    
    # 2. Colorized depth using RealSense's approach
    colored_depth = colorize_depth(depth_frame_meters)
    plt.subplot(132)
    plt.imshow(cv2.cvtColor(colored_depth, cv2.COLOR_BGR2RGB))
    plt.title('Colorized Depth (RealSense Style)')
    
    # 3. Histogram of depth values
    plt.subplot(133)
    plt.hist(depth_frame_meters[valid_mask].flatten(), bins=50)
    plt.title('Depth Distribution')
    plt.xlabel('Distance (meters)')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('depth_analysis.png')
    plt.close()
    
    # Save the colorized depth separately
    cv2.imwrite('depth_colorized.png', colored_depth)
    
    # Additional analysis
    print("\nDepth Distribution Analysis:")
    percentiles = [25, 50, 75, 90, 95, 99]
    for p in percentiles:
        value = np.percentile(depth_frame_meters[valid_mask], p)
        print(f"{p}th percentile: {value:.3f} meters")

if __name__ == "__main__":
    depth_file_path = "assets/examples/realsense_Depth.raw"
    metadata_path = "assets/examples/realsense_Depth_metadata.csv"
    parse_realsense_depth(depth_file_path, metadata_path) 