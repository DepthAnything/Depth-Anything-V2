import cv2
import numpy as np
import streamlit_depth_client as st
import torch

from .config import DEVICE


@st.cache_data(show_spinner="Estimating depth …")
def infer_depth(
    _model,          # leading underscore prevents Streamlit hashing
    image_rgb: np.ndarray,
    input_size: int,
) -> np.ndarray:
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=DEVICE == "cuda"):
        depth = _model.infer_image(img_bgr, input_size=input_size)
    return depth.astype("float32")
