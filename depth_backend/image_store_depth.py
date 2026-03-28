import os
import time
import threading
import cv2
import numpy as np
import pandas as pd
import torch
import sqlite3
from depth_anything_v2.dpt import DepthAnythingV2
from contextlib import contextmanager

class DepthStore:
    def __init__(self, encoder="vitl", min_size=518, bins=50):
        self._lock = threading.Lock()
        self._frame_raw = self._frame_depth = None
        self._last_update = 0
        self.recording = False
        self.raw_path = self.proc_path = None
        self.frame_count = 0
        self.hist_bins = bins
        self.hist_edges = np.linspace(0, 1, bins + 1)
        self.hist_counts = np.zeros(bins, dtype=np.int64)
        self.encoder, self.min_size = encoder, min_size
        self.model, self.device = self._load_model()
        self._init_db()

    def _init_db(self):
        self.db_conn = sqlite3.connect(':memory:', check_same_thread=False)
        self.db_conn.execute("""
            CREATE TABLE IF NOT EXISTS depth_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                min REAL,
                max REAL,
                mean REAL,
                std REAL,
                frame_count INTEGER
            )
        """)
        self.db_conn.commit()

    @contextmanager
    def _db_cursor(self):
        cursor = self.db_conn.cursor()
        try:
            yield cursor
            self.db_conn.commit()
        finally:
            cursor.close()

    def _load_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = {
            "encoder": self.encoder,
            "features": 256,
            "out_channels": [256, 512, 1024, 1024]
        }
        m = DepthAnythingV2(**cfg)
        m.load_state_dict(torch.load(
            f"checkpoints/depth_anything_v2_{self.encoder}.pth",
            map_location=device))
        m.to(device).eval()
        return m, device

    def _depth_inference(self, rgb):
        return self.model.infer_image(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            self.min_size
        )

    def _process_frame(self, frame_bgr):
        # Convert to RGB for processing
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Depth processing
        with torch.no_grad():
            depth = self._depth_inference(frame_rgb)
        
        dmin, dmax = float(depth.min()), float(depth.max())
        norm = ((depth - dmin) / (dmax - dmin + 1e-6)).clip(0, 1)
        vis = cv2.applyColorMap((norm * 255).astype("uint8"), cv2.COLORMAP_MAGMA)
        
        # Histogram
        idx, _ = np.histogram(norm.flatten(), self.hist_edges)
        
        # Stats
        stats = (time.time(), dmin, dmax, float(depth.mean()), float(depth.std()))
        
        return frame_rgb, vis, stats, idx

    def set_frame(self, img_bytes: bytes):
        try:
            frame_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if frame_bgr is None:
                raise ValueError("Empty frame")
            
            # Process frame
            frame_rgb, vis, stats, hist_idx = self._process_frame(frame_bgr)
            
            # Update state
            with self._lock:
                # Encode raw frame
                _, enc_raw = cv2.imencode(".jpg", frame_bgr)
                self._frame_raw = enc_raw.tobytes()
                
                # Encode depth frame
                _, enc_depth = cv2.imencode(".jpg", vis)
                self._frame_depth = enc_depth.tobytes()
                
                # Update histogram
                self.hist_counts += hist_idx
                self._last_update = time.time()
                
                if self.recording:
                    # Save to database
                    with self._db_cursor() as cur:
                        cur.execute(
                            "INSERT INTO depth_metrics (timestamp, min, max, mean, std, frame_count) VALUES (?, ?, ?, ?, ?, ?)",
                            (*stats, self.frame_count)
                        )
                    
                    # Save images
                    cv2.imwrite(
                        os.path.join(self.raw_path, f"frame_{self.frame_count:06d}.jpg"),
                        frame_bgr
                    )
                    cv2.imwrite(
                        os.path.join(self.proc_path, f"depth_{self.frame_count:06d}.png"),
                        (norm * 255).astype("uint8")
                    )
                    self.frame_count += 1
                    
        except Exception as e:
            print(f"Error processing frame: {e}")

    def get_frame_raw(self):
        with self._lock:
            return self._frame_raw if (time.time() - self._last_update) < 5 else None

    def get_frame_depth(self):
        with self._lock:
            return self._frame_depth if (time.time() - self._last_update) < 5 else None

    def last_stats(self):
        with self._db_cursor() as cur:
            row = cur.execute(
                "SELECT min, max, mean, std FROM depth_metrics ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            return row if row else (0, 0, 0, 0)

    def stats_timeseries(self):
        with self._db_cursor() as cur:
            return cur.execute(
                "SELECT timestamp, min, max, mean, std FROM depth_metrics ORDER BY timestamp"
            ).fetchall()

    def hist(self):
        with self._lock:
            return self.hist_edges.tolist(), self.hist_counts.tolist()

    def start(self):
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.raw_path = os.path.join("data/raw", ts)
        self.proc_path = os.path.join("data/depth", ts)
        os.makedirs(self.raw_path, exist_ok=True)
        os.makedirs(self.proc_path, exist_ok=True)
        
        with self._lock:
            self.recording = True
            self.frame_count = 0
            self.hist_counts[:] = 0

    def stop(self):
        with self._lock:
            self.recording = False

    def is_recording(self):
        with self._lock:
            return self.recording

    def stats_to_csv(self):
        with self._db_cursor() as cur:
            df = pd.DataFrame(
                cur.execute("SELECT * FROM depth_metrics").fetchall(),
                columns=["id", "timestamp", "min", "max", "mean", "std", "frame_count"]
            )
            path = os.path.join(self.proc_path or ".", "metrics.csv")
            df.to_csv(path, index=False)
            return path

depth_store = DepthStore()