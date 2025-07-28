# pig_depth_tracker/model.py
import streamlit as st
import torch
import os
import requests
import cv2
import numpy as np
from config import RECORDING_SERVER
try:
    from depth_anything_v2.dpt import DepthAnythingV2
except ImportError:
    st.error("Dependencia no encontrada: 'depth_anything_v2'. Por favor, instálala para continuar.")
    st.code("pip install git+https://github.com/LiheYoung/Depth-Anything-V2.git")
    st.stop()
    
@torch.no_grad()
@st.cache_resource(show_spinner="Cargando modelo de IA…")
def load_model(encoder="vitl"):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        st.info("GPU con CUDA detectada. Usando GPU para el procesamiento.")
    else:
        device = torch.device("cpu")
        st.warning("No se detectó una GPU con CUDA. El modelo se ejecutará en la CPU, lo que será considerablemente más lento.")
    
    if not os.path.exists("checkpoints"):
        st.error("Directorio 'checkpoints' no encontrado. Por favor, descarga los modelos y colócalos en esa carpeta.")
        st.stop()
        
    ckpt = f"checkpoints/depth_anything_v2_{encoder}.pth"
    if not os.path.exists(ckpt):
        st.error(f"Modelo no encontrado: {ckpt}. Asegúrate de que el archivo del modelo está en el directorio 'checkpoints'.")
        st.stop()
        
    cfg = {"encoder": encoder, "features": 256, "out_channels": [256, 512, 1024, 1024]}
    net = DepthAnythingV2(**cfg)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.to(device).eval()
    return net, device

@torch.no_grad()
def predict_depth(model, device, image):
    # Convertir a RGB y redimensionar si no está en tamaño adecuado
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Preprocesamiento
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 360))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)

    # Inferencia
    depth = model(img_tensor)[0, 0].cpu().numpy()
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    # Generar mapa coloreado
    depth_colored = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)

    metrics = {
        "min": float(depth.min()),
        "max": float(depth.max()),
        "mean": float(depth.mean()),
        "std": float(depth.std()),
    }

    return depth_normalized, metrics, depth_colored