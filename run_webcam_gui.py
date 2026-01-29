import sys
import cv2
import torch
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QFrame, QStatusBar, QCheckBox,
    QSlider, QGroupBox, QGridLayout, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from depth_anything_v2.dpt import DepthAnythingV2


class DepthWorker(QThread):
    """Worker thread for depth estimation to keep UI responsive."""
    frame_ready = pyqtSignal(np.ndarray, np.ndarray, float, float)  # RGB, depth, min_val, max_val
    fps_updated = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.running = False
        self.camera_index = 0
        self.model = None
        self.device = None
        self.input_size = 518
        self.use_fp16 = False

        # Visualization parameters
        self.use_colormap = True
        self.colormap = cv2.COLORMAP_INFERNO
        self.invert_depth = False
        self.clip_near = 0      # Percentile (0-100)
        self.clip_far = 100     # Percentile (0-100)
        self.gamma = 1.0        # Gamma correction
        self.rotation = 0       # Rotation: 0, 90, 180, 270 degrees

    def set_model(self, model, device):
        self.model = model
        self.device = device

    def set_camera(self, index):
        self.camera_index = index

    def run(self):
        self.running = True
        # Use V4L2 backend explicitly on Linux for better compatibility
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)

        if not cap.isOpened():
            # Fallback to default backend
            cap = cv2.VideoCapture(self.camera_index)
            if not cap.isOpened():
                self.running = False
                return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency

        frame_count = 0
        fps_timer = cv2.getTickCount()

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            # Apply rotation if needed
            if self.rotation == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif self.rotation == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif self.rotation == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            if self.model is not None:
                # Get depth prediction (relative depth - higher = farther)
                depth = self.model.infer_image(frame, self.input_size)

                # Store raw min/max for display
                raw_min, raw_max = depth.min(), depth.max()

                # Invert if requested (make close objects = high values)
                if self.invert_depth:
                    depth = depth.max() - depth

                # Apply percentile clipping to focus on specific depth range
                near_val = np.percentile(depth, self.clip_near)
                far_val = np.percentile(depth, self.clip_far)

                # Clip and normalize
                depth_clipped = np.clip(depth, near_val, far_val)
                depth_norm = (depth_clipped - near_val) / (far_val - near_val + 1e-8)

                # Apply gamma correction (< 1 emphasizes near, > 1 emphasizes far)
                depth_norm = np.power(depth_norm, self.gamma)

                # Convert to 8-bit
                depth_uint8 = (depth_norm * 255).astype(np.uint8)

                if self.use_colormap:
                    depth_vis = cv2.applyColorMap(depth_uint8, self.colormap)
                else:
                    depth_vis = cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)

                depth_vis = cv2.resize(depth_vis, (frame.shape[1], frame.shape[0]))

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                depth_rgb = cv2.cvtColor(depth_vis, cv2.COLOR_BGR2RGB)

                self.frame_ready.emit(frame_rgb, depth_rgb, raw_min, raw_max)

                frame_count += 1
                if frame_count >= 10:
                    current_time = cv2.getTickCount()
                    fps = frame_count / ((current_time - fps_timer) / cv2.getTickFrequency())
                    self.fps_updated.emit(fps)
                    fps_timer = current_time
                    frame_count = 0

        cap.release()

    def stop(self):
        self.running = False
        self.wait()


class DepthAnythingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Depth Anything V2 - Real-time Depth Estimation")
        self.setMinimumSize(1000, 600)
        self.resize(1400, 800)

        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

        self.model_configs = {
            'vits (Small - Fast)': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb (Base - Balanced)': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl (Large - Accurate)': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }

        self.colormaps = {
            'Inferno': cv2.COLORMAP_INFERNO,
            'Viridis': cv2.COLORMAP_VIRIDIS,
            'Plasma': cv2.COLORMAP_PLASMA,
            'Magma': cv2.COLORMAP_MAGMA,
            'Jet': cv2.COLORMAP_JET,
            'Turbo': cv2.COLORMAP_TURBO,
            'Hot': cv2.COLORMAP_HOT,
            'Bone': cv2.COLORMAP_BONE,
        }

        self.model = None
        self.worker = None
        self.is_running = False

        self.init_ui()
        self.detect_cameras()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(5)

        # === TOP: Video Display Area (Maximum Size) ===
        display_frame = QFrame()
        display_layout = QHBoxLayout(display_frame)
        display_layout.setSpacing(10)

        # RGB frame
        rgb_container = QVBoxLayout()
        rgb_label_title = QLabel("RGB Camera")
        rgb_label_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rgb_label_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        rgb_container.addWidget(rgb_label_title)

        self.rgb_label = QLabel()
        self.rgb_label.setMinimumSize(400, 300)
        self.rgb_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.rgb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rgb_label.setStyleSheet("background-color: #1a1a1a; border: 2px solid #444;")
        self.rgb_label.setText("Camera feed")
        self.rgb_label.setScaledContents(False)
        rgb_container.addWidget(self.rgb_label, stretch=1)
        display_layout.addLayout(rgb_container, stretch=1)

        # Depth frame
        depth_container = QVBoxLayout()
        depth_label_title = QLabel("Depth Estimation")
        depth_label_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        depth_label_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        depth_container.addWidget(depth_label_title)

        self.depth_label = QLabel()
        self.depth_label.setMinimumSize(400, 300)
        self.depth_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.depth_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.depth_label.setStyleSheet("background-color: #1a1a1a; border: 2px solid #444;")
        self.depth_label.setText("Depth map")
        self.depth_label.setScaledContents(False)
        depth_container.addWidget(self.depth_label, stretch=1)
        display_layout.addLayout(depth_container, stretch=1)

        main_layout.addWidget(display_frame, stretch=1)

        # === BOTTOM: All Controls ===
        controls_frame = QFrame()
        controls_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        controls_frame.setMaximumHeight(140)
        controls_layout = QHBoxLayout(controls_frame)

        # --- Left: Camera & Model Selection ---
        left_group = QGroupBox("Source")
        left_layout = QGridLayout(left_group)

        left_layout.addWidget(QLabel("Camera:"), 0, 0)
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(100)
        left_layout.addWidget(self.camera_combo, 0, 1)

        left_layout.addWidget(QLabel("Model:"), 1, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(self.model_configs.keys())
        self.model_combo.setCurrentIndex(1)
        self.model_combo.setMinimumWidth(140)
        left_layout.addWidget(self.model_combo, 1, 1)

        # Buttons
        btn_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load Model")
        self.load_btn.clicked.connect(self.load_model)
        btn_layout.addWidget(self.load_btn)

        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.toggle_stream)
        self.start_btn.setEnabled(False)
        self.start_btn.setStyleSheet("font-weight: bold;")
        btn_layout.addWidget(self.start_btn)
        left_layout.addLayout(btn_layout, 2, 0, 1, 2)

        controls_layout.addWidget(left_group)

        # --- Performance (Jetson Optimization) ---
        perf_group = QGroupBox("Performance")
        perf_layout = QGridLayout(perf_group)

        perf_layout.addWidget(QLabel("Input Size:"), 0, 0)
        self.input_size_combo = QComboBox()
        self.input_size_combo.addItems(["518 (Default)", "392 (Faster)", "294 (Fastest)", "616 (Quality)"])
        self.input_size_combo.currentIndexChanged.connect(self.update_input_size)
        perf_layout.addWidget(self.input_size_combo, 0, 1)

        self.fp16_checkbox = QCheckBox("FP16 (Half Precision)")
        self.fp16_checkbox.setChecked(False)
        self.fp16_checkbox.setToolTip("Use FP16 for faster inference and lower memory (recommended for Jetson)")
        perf_layout.addWidget(self.fp16_checkbox, 1, 0, 1, 2)

        controls_layout.addWidget(perf_group)

        # --- Middle: Visualization Options ---
        mid_group = QGroupBox("Visualization")
        mid_layout = QGridLayout(mid_group)

        mid_layout.addWidget(QLabel("Colormap:"), 0, 0)
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(self.colormaps.keys())
        self.colormap_combo.currentTextChanged.connect(self.update_colormap)
        mid_layout.addWidget(self.colormap_combo, 0, 1)

        mid_layout.addWidget(QLabel("Rotation:"), 0, 2)
        self.rotation_combo = QComboBox()
        self.rotation_combo.addItems(["0°", "90°", "180°", "270°"])
        self.rotation_combo.currentIndexChanged.connect(self.update_rotation)
        mid_layout.addWidget(self.rotation_combo, 0, 3)

        self.grayscale_checkbox = QCheckBox("Grayscale")
        self.grayscale_checkbox.stateChanged.connect(self.toggle_grayscale)
        mid_layout.addWidget(self.grayscale_checkbox, 1, 0)

        self.invert_checkbox = QCheckBox("Invert (near=bright)")
        self.invert_checkbox.setChecked(False)
        self.invert_checkbox.stateChanged.connect(self.toggle_invert)
        mid_layout.addWidget(self.invert_checkbox, 1, 1, 1, 3)

        controls_layout.addWidget(mid_group)

        # --- Right: Depth Range Controls ---
        right_group = QGroupBox("Depth Range")
        right_layout = QGridLayout(right_group)

        right_layout.addWidget(QLabel("Near:"), 0, 0)
        self.near_slider = QSlider(Qt.Orientation.Horizontal)
        self.near_slider.setRange(0, 95)
        self.near_slider.setValue(0)
        self.near_slider.valueChanged.connect(self.update_clip_near)
        right_layout.addWidget(self.near_slider, 0, 1)
        self.near_label = QLabel("0%")
        self.near_label.setMinimumWidth(35)
        right_layout.addWidget(self.near_label, 0, 2)

        right_layout.addWidget(QLabel("Far:"), 1, 0)
        self.far_slider = QSlider(Qt.Orientation.Horizontal)
        self.far_slider.setRange(5, 100)
        self.far_slider.setValue(100)
        self.far_slider.valueChanged.connect(self.update_clip_far)
        right_layout.addWidget(self.far_slider, 1, 1)
        self.far_label = QLabel("100%")
        self.far_label.setMinimumWidth(35)
        right_layout.addWidget(self.far_label, 1, 2)

        right_layout.addWidget(QLabel("Gamma:"), 2, 0)
        self.gamma_slider = QSlider(Qt.Orientation.Horizontal)
        self.gamma_slider.setRange(10, 300)
        self.gamma_slider.setValue(100)
        self.gamma_slider.valueChanged.connect(self.update_gamma)
        right_layout.addWidget(self.gamma_slider, 2, 1)
        self.gamma_label = QLabel("1.00")
        self.gamma_label.setMinimumWidth(35)
        right_layout.addWidget(self.gamma_label, 2, 2)

        controls_layout.addWidget(right_group)

        # --- Far Right: Presets ---
        preset_group = QGroupBox("Presets")
        preset_layout = QGridLayout(preset_group)

        self.preset_near_btn = QPushButton("Near (~2m)")
        self.preset_near_btn.clicked.connect(self.preset_near)
        preset_layout.addWidget(self.preset_near_btn, 0, 0)

        self.preset_mid_btn = QPushButton("Mid Range")
        self.preset_mid_btn.clicked.connect(self.preset_mid)
        preset_layout.addWidget(self.preset_mid_btn, 0, 1)

        self.preset_reset_btn = QPushButton("Reset")
        self.preset_reset_btn.clicked.connect(self.preset_reset)
        preset_layout.addWidget(self.preset_reset_btn, 1, 0)

        self.preset_jetson_btn = QPushButton("Jetson")
        self.preset_jetson_btn.clicked.connect(self.preset_jetson)
        self.preset_jetson_btn.setToolTip("Optimal settings for Jetson Orin Nano")
        self.preset_jetson_btn.setStyleSheet("background-color: #76b900; color: white; font-weight: bold;")
        preset_layout.addWidget(self.preset_jetson_btn, 1, 1)

        controls_layout.addWidget(preset_group)

        main_layout.addWidget(controls_frame)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(f"Device: {self.device} | Ready - Load a model to start")

        self.depth_info_label = QLabel("Depth: --")
        self.status_bar.addPermanentWidget(self.depth_info_label)

        self.fps_label = QLabel("FPS: --")
        self.status_bar.addPermanentWidget(self.fps_label)

    def detect_cameras(self):
        """Detect available cameras by checking /dev/video* devices."""
        self.camera_combo.clear()
        available_cameras = []

        # On Linux, check /dev/video* files instead of probing with OpenCV
        # This avoids hanging on non-existent cameras
        import os
        for i in range(10):
            if os.path.exists(f"/dev/video{i}"):
                available_cameras.append(f"Camera {i}")

        if available_cameras:
            self.camera_combo.addItems(available_cameras)
        else:
            # Fallback: just offer camera 0
            self.camera_combo.addItem("Camera 0")

        self.start_btn.setEnabled(True)

    def load_model(self):
        model_name = self.model_combo.currentText()
        config = self.model_configs[model_name]
        encoder = config['encoder']

        self.status_bar.showMessage(f"Loading model: {encoder}...")
        QApplication.processEvents()

        try:
            self.model = DepthAnythingV2(**config)
            checkpoint_path = f'checkpoints/depth_anything_v2_{encoder}.pth'
            self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
            self.model = self.model.to(self.device).eval()

            # Apply FP16 if selected
            precision = "FP32"
            if self.fp16_checkbox.isChecked() and self.device != 'cpu':
                self.model = self.model.half()
                precision = "FP16"

            self.status_bar.showMessage(f"Device: {self.device} | Model: {encoder} | Precision: {precision}")
            self.start_btn.setEnabled(True)
            self.load_btn.setText("Reload Model")

        except FileNotFoundError:
            self.status_bar.showMessage(f"Error: Checkpoint not found for {encoder}. Download it first.")
            self.model = None
        except Exception as e:
            self.status_bar.showMessage(f"Error loading model: {str(e)}")
            self.model = None

    def toggle_stream(self):
        if self.is_running:
            self.stop_stream()
        else:
            self.start_stream()

    def start_stream(self):
        if self.model is None:
            self.status_bar.showMessage("Please load a model first!")
            return

        camera_index = self.camera_combo.currentIndex()

        self.worker = DepthWorker()
        self.worker.set_model(self.model, self.device)
        self.worker.set_camera(camera_index)
        self.apply_viz_settings()
        self.worker.frame_ready.connect(self.update_frames)
        self.worker.fps_updated.connect(self.update_fps)
        self.worker.start()

        self.is_running = True
        self.start_btn.setText("Stop")
        self.load_btn.setEnabled(False)
        self.camera_combo.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.fp16_checkbox.setEnabled(False)
        self.status_bar.showMessage(f"Device: {self.device} | Streaming from Camera {camera_index}")

    def stop_stream(self):
        if self.worker:
            self.worker.stop()
            self.worker = None

        self.is_running = False
        self.start_btn.setText("Start")
        self.load_btn.setEnabled(True)
        self.camera_combo.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.fp16_checkbox.setEnabled(True)
        self.fps_label.setText("FPS: --")
        self.status_bar.showMessage(f"Device: {self.device} | Stopped")

    def apply_viz_settings(self):
        """Apply all visualization settings to worker."""
        if self.worker:
            self.worker.use_colormap = not self.grayscale_checkbox.isChecked()
            colormap_name = self.colormap_combo.currentText()
            self.worker.colormap = self.colormaps.get(colormap_name, cv2.COLORMAP_INFERNO)
            self.worker.invert_depth = self.invert_checkbox.isChecked()
            self.worker.clip_near = self.near_slider.value()
            self.worker.clip_far = self.far_slider.value()
            self.worker.gamma = self.gamma_slider.value() / 100.0
            rotations = [0, 90, 180, 270]
            self.worker.rotation = rotations[self.rotation_combo.currentIndex()]
            # Performance settings
            sizes = [518, 392, 294, 616]
            self.worker.input_size = sizes[self.input_size_combo.currentIndex()]

    def update_colormap(self, name):
        if self.worker:
            self.worker.colormap = self.colormaps.get(name, cv2.COLORMAP_INFERNO)

    def update_rotation(self, index):
        rotations = [0, 90, 180, 270]
        if self.worker:
            self.worker.rotation = rotations[index]

    def update_input_size(self, index):
        sizes = [518, 392, 294, 616]
        if self.worker:
            self.worker.input_size = sizes[index]

    def toggle_grayscale(self, state):
        if self.worker:
            self.worker.use_colormap = state != Qt.CheckState.Checked.value

    def toggle_invert(self, state):
        if self.worker:
            self.worker.invert_depth = state == Qt.CheckState.Checked.value

    def update_clip_near(self, value):
        self.near_label.setText(f"{value}%")
        # Ensure near < far
        if value >= self.far_slider.value():
            self.far_slider.setValue(value + 5)
        if self.worker:
            self.worker.clip_near = value

    def update_clip_far(self, value):
        self.far_label.setText(f"{value}%")
        # Ensure far > near
        if value <= self.near_slider.value():
            self.near_slider.setValue(value - 5)
        if self.worker:
            self.worker.clip_far = value

    def update_gamma(self, value):
        gamma = value / 100.0
        self.gamma_label.setText(f"{gamma:.2f}")
        if self.worker:
            self.worker.gamma = gamma

    # === Presets ===
    def preset_near(self):
        """Optimize for viewing objects within ~2m."""
        self.invert_checkbox.setChecked(True)  # Make close objects bright
        self.near_slider.setValue(0)
        self.far_slider.setValue(40)  # Clip far objects (focus on near 40%)
        self.gamma_slider.setValue(70)  # Gamma < 1 emphasizes near range
        self.apply_viz_settings()

    def preset_mid(self):
        """Balanced mid-range view."""
        self.invert_checkbox.setChecked(True)
        self.near_slider.setValue(10)
        self.far_slider.setValue(80)
        self.gamma_slider.setValue(100)
        self.apply_viz_settings()

    def preset_reset(self):
        """Reset to defaults."""
        self.invert_checkbox.setChecked(False)
        self.near_slider.setValue(0)
        self.far_slider.setValue(100)
        self.gamma_slider.setValue(100)
        self.grayscale_checkbox.setChecked(False)
        self.colormap_combo.setCurrentIndex(0)
        self.rotation_combo.setCurrentIndex(0)
        self.input_size_combo.setCurrentIndex(0)
        self.apply_viz_settings()

    def preset_jetson(self):
        """Optimal settings for Jetson Orin Nano."""
        # Performance settings
        self.input_size_combo.setCurrentIndex(1)  # 392 (Faster)
        self.fp16_checkbox.setChecked(True)
        # Select small model for best FPS
        self.model_combo.setCurrentIndex(0)  # vits (Small)
        # Visualization
        self.invert_checkbox.setChecked(True)
        self.colormap_combo.setCurrentText("Turbo")
        self.apply_viz_settings()
        self.status_bar.showMessage("Jetson preset applied - Reload model with FP16 for best performance")

    def update_frames(self, rgb_frame, depth_frame, min_val, max_val):
        rgb_pixmap = self.numpy_to_pixmap(rgb_frame)
        depth_pixmap = self.numpy_to_pixmap(depth_frame)

        self.rgb_label.setPixmap(rgb_pixmap.scaled(
            self.rgb_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))
        self.depth_label.setPixmap(depth_pixmap.scaled(
            self.depth_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

        # Update depth info
        self.depth_info_label.setText(f"Depth range: {min_val:.2f} - {max_val:.2f}")

    def numpy_to_pixmap(self, img):
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def update_fps(self, fps):
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = DepthAnythingGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
