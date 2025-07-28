# pig_depth_tracker/config.py
MAX_VIDEO_DURATION = 300
MAX_VIDEO_SIZE_MB = 100
SUPPORTED_FORMATS = ["mp4", "mov", "avi"]
RECORDING_SERVER = "http://192.168.1.42:8000"
FRAME_POLL_INTERVAL = 2
MIN_PIG_AREA = 10000
DEPTH_CHANGE_THRESHOLD = 0.02

# Configuración de detección mejorada
BACKGROUND_UPDATE_ALPHA = 0.01
MIN_PIG_AREA = 8000
MORPH_KERNEL_SIZE = 15
DEPTH_CHANGE_THRESHOLD = 0.02

MIN_SOW_AREA = 15000  # Minimum pixel area for valid sow detection
SEGMENTATION_MORPH_KERNEL = 25  # Size of morphological operation kernel
COLOR_LOWER_PINK = [140, 40, 40]  # HSV lower bounds for pink tones
COLOR_UPPER_PINK = [180, 255, 255]  # HSV upper bounds for pink tones
COLOR_LOWER_BROWN = [5, 40, 40]  # HSV lower bounds for brown tones
COLOR_UPPER_BROWN = [30, 255, 255]  # HSV upper bounds for brown tones

# Validation parameters
MIN_SOW_PIXELS = 1000  # Minimum pixels to consider as valid detection
MIN_SOLIDITY = 0.85  # Minimum contour solidity for valid detection