import cv2
import numpy as np
import os
import tempfile
import torch
import streamlit as st
from sklearn.mixture import GaussianMixture
from model import load_model
from config import MAX_VIDEO_SIZE_MB, SUPPORTED_FORMATS, MAX_VIDEO_DURATION, MORPH_KERNEL_SIZE, MIN_PIG_AREA
from state import reset_app_state

from scipy import ndimage

# Updated segmentation in processing.py
# Updated segmentation in processing.py
def segment_sow_enhanced(depth_map, rgb_frame=None, min_area=15000):
    """
    Enhanced segmentation specifically for overhead sow detection
    Combines depth and color information for better accuracy
    """
    # Normalize depth map
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Adaptive thresholding - works better for varying lighting
    thresh = cv2.adaptiveThreshold(
        depth_norm, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 51, 7  # Increased block size for larger animals
    )
    
    # Morphological operations with elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))  # Larger kernel for sow size
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    # Find contours and filter by size and solidity
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area and solidity (sows should be large, solid shapes)
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area)/hull_area if hull_area > 0 else 0
        
        if area > min_area and solidity > 0.85:  # Sows should be fairly solid shapes
            valid_contours.append(cnt)
    
    # Create initial mask
    mask = np.zeros_like(depth_norm, dtype=np.uint8)
    cv2.drawContours(mask, valid_contours, -1, 255, -1)
    
    # Refine with color information if available
    if rgb_frame is not None:
        mask = refine_with_color(mask, rgb_frame)
    
    return mask, len(valid_contours) > 0  # Return mask and detection status

def refine_with_color(mask, rgb_frame):
    """Color-based refinement for sows (typically pinkish/brownish)"""
    hsv = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2HSV)
    
    # Color ranges for typical sow colors
    lower_pink = np.array([140, 40, 40])
    upper_pink = np.array([180, 255, 255])
    lower_brown = np.array([5, 40, 40])
    upper_brown = np.array([30, 255, 255])
    
    # Create color masks
    pink_mask = cv2.inRange(hsv, lower_pink, upper_pink)
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    color_mask = cv2.bitwise_or(pink_mask, brown_mask)
    
    # Combine with depth mask
    refined_mask = cv2.bitwise_and(mask, color_mask)
    
    # Final cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
    
    return refined_mask

def analyze_sow_depth(depth_map, mask):
    """
    Robust depth analysis with validation checks
    Returns None if no valid sow detected
    """
    if mask.sum() == 0:  # Empty mask
        return None
    
    sow_pixels = depth_map[mask == 255]
    if len(sow_pixels) < 1000:  # Minimum pixels to consider valid
        return None
    
    # Calculate depth statistics
    analysis = {
        'mean_depth': float(np.mean(sow_pixels)),
        'median_depth': float(np.median(sow_pixels)),
        'std_depth': float(np.std(sow_pixels)),
        'min_depth': float(np.min(sow_pixels)),
        'max_depth': float(np.max(sow_pixels)),
        'area_pixels': int(len(sow_pixels)),
        'percentiles': {
            '5': float(np.percentile(sow_pixels, 5)),
            '25': float(np.percentile(sow_pixels, 25)),
            '50': float(np.percentile(sow_pixels, 50)),
            '75': float(np.percentile(sow_pixels, 75)),
            '95': float(np.percentile(sow_pixels, 95))
        }
    }
    
    # Calculate body region depths
    y_coords, x_coords = np.where(mask == 255)
    height = y_coords.max() - y_coords.min()
    
    for region, y_range in [('head', (0, 0.3)), 
                          ('middle', (0.3, 0.7)), 
                          ('rear', (0.7, 1.0))]:
        region_mask = (y_coords >= y_coords.min() + y_range[0]*height) & \
                     (y_coords <= y_coords.min() + y_range[1]*height)
        region_depths = sow_pixels[region_mask]
        
        if len(region_depths) > 0:
            analysis[f'{region}_mean'] = float(np.mean(region_depths))
            analysis[f'{region}_std'] = float(np.std(region_depths))
    
    return analysis

def handle_segmentation_errors(results):
    """Analyze and handle cases where segmentation fails"""
    if results['detection_success']:
        return results
    
    # Try fallback methods when primary segmentation fails
    if results['sow_mask'] is None:
        # Method 1: Try with relaxed parameters
        relaxed_mask, _ = segment_sow_enhanced(
            results['raw_depth'], 
            results['frame'],
            min_area=10000  # Lower minimum area
        )
        
        # Method 2: Try pure depth-based segmentation
        if relaxed_mask.sum() == 0:
            _, relaxed_mask = cv2.threshold(
                (results['raw_depth'] * 255).astype(np.uint8),
                0, 255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
        
        # If either method worked, re-analyze
        if relaxed_mask.sum() > 0:
            results['sow_mask'] = relaxed_mask
            results['depth_analysis'] = analyze_sow_depth(results['raw_depth'], relaxed_mask)
            results['detection_success'] = results['depth_analysis'] is not None
    
    return results


def validate_video_file(f):
    if f.size > MAX_VIDEO_SIZE_MB * 1024 * 1024:
        st.error(f"El vídeo no puede superar los {MAX_VIDEO_SIZE_MB} MB."); return False
    if f.name.split('.')[-1].lower() not in SUPPORTED_FORMATS:
        st.error(f"Formato no soportado. Sube {', '.join(SUPPORTED_FORMATS)}."); return False
    return True

def extract_frames(path):
    frames=[]; cap=cv2.VideoCapture(path); fps=cap.get(cv2.CAP_PROP_FPS); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or n <= 0: st.error("No se pudo leer la información del vídeo."); return []
    if n / fps > MAX_VIDEO_DURATION: st.error(f"El vídeo no puede durar más de {MAX_VIDEO_DURATION} segundos."); return []
    prog = st.progress(0., text="Extrayendo frames...")
    while True:
        ok, f = cap.read()
        if not ok: break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        prog.progress(len(frames) / n, text=f"Extrayendo frame {len(frames)}/{n}")
    cap.release(); return frames

def predict_depth(model, device, img_rgb):
    with torch.no_grad():
        raw = model.infer_image(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    norm = cv2.normalize(raw, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    colored = cv2.cvtColor(cv2.applyColorMap((norm*255).astype(np.uint8), cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
    metrics = dict(min=float(raw.min()), max=float(raw.max()), mean=float(raw.mean()), std=float(raw.std()), median=float(np.median(raw)))
    return raw, metrics, colored

def segment_pig_fast(depth_map):
    """Segmenta la región del cerdo con una segmentación simple basada en profundidad."""
    median_depth = np.median(depth_map)
    pig_mask = (depth_map < median_depth * 1.1).astype(np.uint8) * 255
    pig_mask = cv2.morphologyEx(pig_mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return pig_mask

def extract_bbox_and_centroid(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        M = cv2.moments(c)
        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
        cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
        return (x, y, w, h), (cx, cy)
    return None, (0, 0)

# Suggested improvements to analyze_depth_segmented() in processing.py
from scipy import stats

def analyze_sow_depth(depth_map, mask):
    """Robust depth analysis that ensures required fields exist"""
    if mask is None or mask.sum() == 0:
        return None
    
    sow_pixels = depth_map[mask == 255]
    if len(sow_pixels) < 1000:
        return None

    # Ensure all required fields are included
    analysis = {
        'mean_depth': float(np.mean(sow_pixels)),
        'std_depth': float(np.std(sow_pixels)),
        'min_depth': float(np.min(sow_pixels)),
        'max_depth': float(np.max(sow_pixels)),
        # Ensure percentiles exist even if empty
        'percentiles': {
            '5': float(np.percentile(sow_pixels, 5)),
            '25': float(np.percentile(sow_pixels, 25)),
            '50': float(np.percentile(sow_pixels, 50)),
            '75': float(np.percentile(sow_pixels, 75)),
            '95': float(np.percentile(sow_pixels, 95))
        },
        'anomaly_pixels': 0.0  # Initialize with default value
    }

    # Calculate anomalies (same as before)
    q1 = analysis['percentiles']['25']
    q3 = analysis['percentiles']['75']
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    anomaly_mask = np.zeros_like(mask, dtype=np.uint8)
    anomaly_mask[(depth_map < lower) & (mask == 255)] = 255
    anomaly_mask[(depth_map > upper) & (mask == 255)] = 255

    analysis['anomaly_pixels'] = float(np.sum(anomaly_mask == 255)) / np.sum(mask == 255)
    
    return analysis



def analyze_depth_segmented(masked_depth, pig_mask, frame_shape):
    """
    Enhanced depth analysis with regional and symmetry metrics
    
    Args:
        masked_depth: Depth values within pig mask
        pig_mask: Binary mask of pig
        frame_shape: Shape of original frame (height, width)
        
    Returns:
        tuple: (analysis_dict, anomaly_mask)
    """
    pig_pixels = masked_depth[pig_mask == 255]
    if pig_pixels.size == 0:
        return None, None
    
    # Get coordinates of all pig pixels
    y_coords, x_coords = np.where(pig_mask == 255)
    coords = np.column_stack((x_coords, y_coords))
    depth_values = masked_depth[pig_mask == 255]
    
    # Basic statistics
    analysis = {
        'depth_values': depth_values.tolist(),  # For visualization
        'mean_depth': float(np.mean(depth_values)),
        'std_depth': float(np.std(depth_values)),
        'min_depth': float(np.min(depth_values)),
        'max_depth': float(np.max(depth_values)),
        'skewness': float(stats.skew(depth_values)),
        'kurtosis': float(stats.kurtosis(depth_values)),
        'percentiles': {
            '5': float(np.percentile(depth_values, 5)),
            '25': float(np.percentile(depth_values, 25)),
            '50': float(np.percentile(depth_values, 50)),
            '75': float(np.percentile(depth_values, 75)),
            '95': float(np.percentile(depth_values, 95)),
        }
    }
    
    # Regional analysis
    height, width = frame_shape
    regions = {
        'head': (y_coords < height//3),
        'middle': ((y_coords >= height//3) & (y_coords < 2*height//3)),
        'rear': (y_coords >= 2*height//3),
        'left': (x_coords < width//2),
        'right': (x_coords >= width//2)
    }
    
    for region_name, region_mask in regions.items():
        region_depths = depth_values[region_mask]
        if len(region_depths) > 0:
            analysis[f'{region_name}_mean'] = float(np.mean(region_depths))
            analysis[f'{region_name}_std'] = float(np.std(region_depths))
            analysis[f'{region_name}_min'] = float(np.min(region_depths))
            analysis[f'{region_name}_max'] = float(np.max(region_depths))
    
    # Symmetry analysis
    if 'left_mean' in analysis and 'right_mean' in analysis:
        analysis['symmetry_score'] = float(
            np.abs(analysis['left_mean'] - analysis['right_mean']) / 
            (analysis['mean_depth'] + 1e-8)  # Avoid division by zero
        )
    
    # Anomaly detection
    q1 = analysis['percentiles']['25']
    q3 = analysis['percentiles']['75']
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    anomaly_mask = np.zeros_like(pig_mask, dtype=np.uint8)
    anomaly_mask[(masked_depth < lower_bound) & (pig_mask == 255)] = 255
    anomaly_mask[(masked_depth > upper_bound) & (pig_mask == 255)] = 255
    
    analysis['anomaly_pixels'] = float(np.sum(anomaly_mask == 255)) / np.sum(pig_mask == 255)
    
    return analysis, anomaly_mask

# Update in processing.py
def process_video_file(up_file):
    if not validate_video_file(up_file): 
        return

    reset_app_state()
    model, device = load_model()

    with st.status("Procesando vídeo mejorado...", expanded=True) as status:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(up_file.read())
            video_path = tmp_file.name

        original_frames = extract_frames(video_path)
        frames = [cv2.resize(f, (640, 360)) for f in original_frames]
        os.remove(video_path)

        if not frames:
            status.update(label="Error en extracción de frames", state="error")
            return

        st.session_state.total_frames = len(frames)
        progress_bar = st.progress(0, text="Procesando frames...")

        # Initialize buffers
        originals, raws, coloreds, masks, masked_depths = [], [], [], [], []
        bboxes, centroids, analyses, anomalies = [], [], [], []
        segmented_matrices, confidence_scores = [], []

        for i, frame in enumerate(frames):
            # Replace the existing processing code with:
            raw, metrics, colored = predict_depth(model, device, frame)
            
            # Enhanced segmentation
            sow_mask, detection_success = segment_sow_enhanced(raw, frame)
            masked_depth = np.where(sow_mask == 255, raw, 0) if detection_success else None
            
            # Enhanced analysis
            analysis = analyze_sow_depth(raw, sow_mask) if detection_success else None
            bbox, centroid = extract_bbox_and_centroid(sow_mask) if detection_success else (None, None)
            
            # Store results
            originals.append(original_frames[i])
            raws.append(raw)
            coloreds.append(colored)
            masks.append(sow_mask if detection_success else None)
            masked_depths.append(masked_depth)
            bboxes.append(bbox)
            centroids.append(centroid)
            analyses.append(analysis)
            
            progress_bar.progress((i + 1) / len(frames), 
                                text=f"Frame {i + 1}/{len(frames)} - {'Detected' if detection_success else 'No detection'}")
        # Save to session state
        st.session_state.update({
            'original_frames': originals,
            'depth_maps_raw': raws,
            'depth_maps_colored': coloreds,
            'pig_masks': masks,
            'masked_depths': masked_depths,
            'pig_bboxes': bboxes,
            'pig_centroids': centroids,
            'depth_analysis': analyses,
            'anomaly_maps': anomalies,
            'depth_segmented_matrices': segmented_matrices,
            'segmentation_confidence': confidence_scores
        })

        st.session_state.video_processed = True
        status.update(label="Procesamiento completado", state="complete")
        st.rerun()
