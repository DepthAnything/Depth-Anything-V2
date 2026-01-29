import cv2
import torch
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2

# Setup device
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# Model configurations
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

# Choose encoder: 'vits' (small/fast), 'vitb' (base/balanced), 'vitl' (large/accurate)
encoder = 'vitb'

# Load model
model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()
print(f"Loaded Depth Anything V2 - {encoder}")

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("Press 'q' to quit, 'c' to toggle colormap, 's' to save frame")

use_colormap = True
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Get depth prediction
    depth = model.infer_image(frame)

    # Normalize depth for visualization (0-255)
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min()) * 255
    depth_uint8 = depth_norm.astype(np.uint8)

    # Apply colormap or grayscale
    if use_colormap:
        depth_vis = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
    else:
        depth_vis = cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)

    # Resize depth to match frame size (in case they differ)
    depth_vis = cv2.resize(depth_vis, (frame.shape[1], frame.shape[0]))

    # Show side by side
    combined = np.hstack([frame, depth_vis])
    cv2.imshow('Depth Anything V2 | RGB (left) - Depth (right)', combined)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        use_colormap = not use_colormap
        print(f"Colormap: {'ON' if use_colormap else 'OFF'}")
    elif key == ord('s'):
        cv2.imwrite(f'output/webcam_rgb_{frame_count}.png', frame)
        cv2.imwrite(f'output/webcam_depth_{frame_count}.png', depth_vis)
        print(f"Saved frame {frame_count}")
        frame_count += 1

cap.release()
cv2.destroyAllWindows()
print("Done!")
