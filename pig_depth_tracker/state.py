# pig_depth_tracker/state.py
import streamlit as st
from collections import defaultdict
from config import MIN_PIG_AREA, DEPTH_CHANGE_THRESHOLD
import requests

def init_session_state():
    if "initialized" in st.session_state:
        return
    st.session_state.update({
        "initialized": True,
        "video_processed": False,
        "playing": False,
        "current_frame_index": 0,
        "total_frames": 0,
        "original_frames": [],
        "depth_maps_raw": [],
        "depth_maps_colored": [],
        "pig_masks": [],
        "pig_contours": [],
        "pig_bboxes": [],
        "pig_centroids": [],
        "depth_changes": [],
        "metrics_cache": [],
        "selected_points": [],
        "volume_analysis_results": None,
        "point_analysis_results": None,
        "posture_analysis_results": [],
        "depth_analysis": [],  # <--- ¡AQUÍ!
        "noise_threshold": 0.01,
        "previewing": False,
        "recording": False,
        "recording_session_id": None,
        "recorded_frames": 0,
        "frame_urls": [],
        "live_frame": None,
        "processing_recorded": False,
        "min_pig_area": MIN_PIG_AREA,
        "depth_change_threshold": DEPTH_CHANGE_THRESHOLD,
        "pig_tracks": defaultdict(list),
        "movement_maps": [],
        "masked_depths": []  # ✅ Añade esta línea
    })

def reset_app_state():
    """Función independiente para resetear el estado"""
    from config import RECORDING_SERVER  # Import local para evitar circularidad
    sid = st.session_state.get("recording_session_id")
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    init_session_state()
    if sid:
        try:
            requests.post(f"{RECORDING_SERVER}/stop-recording/{sid}", timeout=2)
        except Exception:
            pass