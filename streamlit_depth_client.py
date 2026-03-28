import streamlit as st
import cv2
import numpy as np
import pandas as pd
import torch
import os
import tempfile
import time
import requests
import io
from contextlib import contextmanager
from streamlit_image_coordinates import streamlit_image_coordinates
from streamlit_echarts import st_echarts
from PIL import Image
from scipy import ndimage
from sklearn.cluster import DBSCAN
from collections import defaultdict

# ───────────────────── MODELO DE PROFUNDIDAD ─────────────────────
try:
    from depth_anything_v2.dpt import DepthAnythingV2
except ImportError:
    st.error("Dependencia no encontrada: 'depth_anything_v2'. Por favor, instálala para continuar.")
    st.code("pip install git+https://github.com/LiheYoung/Depth-Anything-V2.git")
    st.stop()

# ───────────────────── CONFIG APP ─────────────────────
st.set_page_config(page_title="PigDepthTracker", layout="wide",
                   initial_sidebar_state="collapsed")

MAX_VIDEO_DURATION = 300
MAX_VIDEO_SIZE_MB = 100
SUPPORTED_FORMATS = ["mp4", "mov", "avi"]
RECORDING_SERVER = "http://192.168.1.42:8000"
FRAME_POLL_INTERVAL = 2
MIN_PIG_AREA = 10000
DEPTH_CHANGE_THRESHOLD = 0.02  # Umbral para cambios significativos de profundidad

# ───────────────────── UTILIDADES AVANZADAS ─────────────────────
def load_css():
    st.markdown("""
    <style>
        :root {
            --primary: #2c3e50;
            --secondary: #3498db;
            --accent: #1abc9c;
            --light: #ecf0f1;
            --dark: #2c3e50;
            --success: #27ae60;
            --warning: #f39c12;
            --danger: #e74c3c;
        }
        
        .stApp {
            background-color: #f8f9fa;
            color: var(--dark);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        h1, h2, h3 {
            color: var(--primary);
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        
        .stButton>button {
            border-radius: 4px;
            border: 1px solid var(--secondary);
            background-color: white;
            color: var(--secondary);
            font-weight: 500;
            transition: all 0.2s ease;
            padding: 0.5rem 1rem;
        }
        
        .stButton>button:hover {
            background-color: #eaf5ff;
            color: var(--secondary);
            border-color: var(--secondary);
        }
        
        .stButton>button[kind="primary"] {
            background-color: var(--secondary);
            color: white;
            border: none;
        }
        
        .stButton>button[kind="primary"]:hover {
            background-color: #2980b9;
        }
        
        .card {
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            margin-bottom: 1.5rem;
            border: 1px solid #e0e0e0;
        }
        
        .card-title {
            margin-top: 0;
            color: var(--primary);
            font-weight: 600;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid #eee;
            margin-bottom: 1rem;
        }
        
        .status-indicator {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 4px;
            font-size: 0.85rem;
            font-weight: 500;
            margin-bottom: 1rem;
        }
        
        .recording-active {
            background-color: #fdecea;
            color: var(--danger);
            border: 1px solid #fadbd8;
        }
        
        .preview-active {
            background-color: #ebf5fb;
            color: var(--secondary);
            border: 1px solid #d6eaf8;
        }
        
        .metric-card {
            background-color: #fff;
            border: 1px solid #e6e6e6;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .metric-title {
            font-size: 0.85rem;
            color: #888;
            margin-bottom: 0.4rem;
            font-weight: 500;
        }
        .metric-value {
            font-size: 1.5rem;
            font-weight: 600;
            color: #2c3e50;
        }
        .divider {
            height: 1px;
            background: #e0e0e0;
            margin: 1.5rem 0;
        }
        .pig-highlight {
            border: 2px solid #e74c3c;
            border-radius: 4px;
            padding: 4px;
        }
        .movement-map {
            border: 2px solid #3498db;
            border-radius: 4px;
            padding: 4px;
        }
    </style>
    """, unsafe_allow_html=True)

@contextmanager
def card(title: str | None = None):
    st.markdown(f"<div class='card'>{'<div class=\"card-title\">'+title+'</div>' if title else ''}", unsafe_allow_html=True)
    yield
    st.markdown("</div>", unsafe_allow_html=True)

# ───────────────────── GESTIÓN DE ESTADO ─────────────────────
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
        "movement_maps": []
    })

def reset_app_state():
    sid = st.session_state.get("recording_session_id")
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    init_session_state()
    if sid:
        try:
            requests.post(f"{RECORDING_SERVER}/stop-recording/{sid}", timeout=2)
        except Exception:
            pass

# ───────────────────── MODELO DE IA ─────────────────────
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

# ───────────────────── LÓGICA DE ANÁLISIS ─────────────────────
def predict_depth(model, device, img_rgb):
    with torch.no_grad():
        raw = model.infer_image(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    norm = cv2.normalize(raw, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    colored = cv2.cvtColor(cv2.applyColorMap((norm*255).astype(np.uint8), cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
    metrics = dict(min=float(raw.min()), max=float(raw.max()), mean=float(raw.mean()), std=float(raw.std()), median=float(np.median(raw)))
    return raw, metrics, colored

def calculate_depth_changes(depth_maps):
    """Calcula los cambios de profundidad entre frames consecutivos"""
    changes = []
    for i in range(1, len(depth_maps)):
        diff = np.abs(depth_maps[i] - depth_maps[i-1])
        changes.append(diff)
    return changes

def detect_moving_objects(depth_change, threshold=DEPTH_CHANGE_THRESHOLD):
    """Detecta objetos en movimiento basado en cambios de profundidad"""
    # Umbralizar para obtener regiones con cambios significativos
    _, thresh = cv2.threshold(depth_change, threshold, 1.0, cv2.THRESH_BINARY)
    movement_mask = (thresh * 255).astype(np.uint8)
    
    # Operaciones morfológicas para limpiar la máscara
    kernel = np.ones((5, 5), np.uint8)
    movement_mask = cv2.morphologyEx(movement_mask, cv2.MORPH_OPEN, kernel)
    movement_mask = cv2.morphologyEx(movement_mask, cv2.MORPH_CLOSE, kernel)
    
    return movement_mask

def cluster_movement_regions(movement_mask, min_area=MIN_PIG_AREA):
    """Agrupa regiones de movimiento para formar objetos completos"""
    # Encontrar contornos
    contours, _ = cv2.findContours(movement_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Crear máscara vacía
    object_mask = np.zeros_like(movement_mask)
    
    # Dibujar todos los contornos significativos
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            cv2.drawContours(object_mask, [contour], -1, 255, cv2.FILLED)
    
    # Conectar regiones cercanas
    object_mask = cv2.dilate(object_mask, np.ones((15, 15), np.uint8), iterations=1)
    object_mask = cv2.erode(object_mask, np.ones((15, 15), np.uint8), iterations=1)
    
    return object_mask

def refine_pig_mask(object_mask, depth_map):
    """Refina la máscara del cerdo usando información de profundidad"""
    # Encontrar contornos en la máscara de objeto
    contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None
    
    # Tomar el contorno más grande (presumiblemente el cerdo)
    main_contour = max(contours, key=cv2.contourArea)
    
    # Crear máscara del cerdo
    pig_mask = np.zeros_like(object_mask)
    cv2.drawContours(pig_mask, [main_contour], -1, 255, cv2.FILLED)
    
    # Obtener bounding box
    x, y, w, h = cv2.boundingRect(main_contour)
    
    # Calcular centroide
    M = cv2.moments(main_contour)
    cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
    cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
    
    return pig_mask, (x, y, w, h), (cx, cy)

def analyze_pig_posture(depth_map, mask):
    """Analiza la postura del cerdo basado en la distribución de profundidad"""
    if mask is None or depth_map is None:
        return None
    
    # Calcular centroide
    moments = cv2.moments(mask.astype(np.uint8))
    if moments["m00"] == 0:
        return None
    
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    
    # Dividir en regiones izquierda/derecha
    left_mask = mask.copy()
    left_mask[:, cx:] = 0
    right_mask = mask.copy()
    right_mask[:, :cx] = 0
    
    # Calcular profundidad media en cada región
    left_depth = np.mean(depth_map[left_mask > 0]) if np.any(left_mask) else 0
    right_depth = np.mean(depth_map[right_mask > 0]) if np.any(right_mask) else 0
    
    # Calcular asimetría
    asymmetry = abs(left_depth - right_depth)
    
    return {
        "centroid": (cx, cy),
        "left_depth": left_depth,
        "right_depth": right_depth,
        "asymmetry": asymmetry
    }

def track_pig(centroids, frame_idx):
    """Seguimiento del cerdo entre frames usando centroides"""
    if frame_idx == 0:
        return 0
    
    last_centroid = st.session_state.pig_centroids[frame_idx-1]
    current_centroid = centroids
    
    # Distancia euclidiana
    distance = np.sqrt((current_centroid[0] - last_centroid[0])**2 + 
                       (current_centroid[1] - last_centroid[1])**2)
    
    # Si la distancia es grande, podría ser un error de detección
    if distance > 100:  # Umbral de distancia máxima
        return last_centroid  # Mantener posición anterior
    
    return current_centroid

# ───────────────────── PROCESAMIENTO DE VÍDEO ─────────────────────
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

def _store_frame(model, device, img, frame_idx):
    """Procesa y almacena un frame con detección de cerdo"""
    raw, m, clr = predict_depth(model, device, img)
    
    # Guardar datos básicos
    st.session_state.original_frames.append(img)
    st.session_state.depth_maps_raw.append(raw)
    st.session_state.depth_maps_colored.append(clr)
    st.session_state.metrics_cache.append(m)
    
    # Para el primer frame, inicializar sin cerdo
    if frame_idx == 0:
        st.session_state.pig_masks.append(np.zeros_like(raw, dtype=np.uint8))
        st.session_state.pig_contours.append(None)
        st.session_state.pig_bboxes.append(None)
        st.session_state.pig_centroids.append((0, 0))
        return
    
    # Calcular cambio de profundidad desde el frame anterior
    depth_change = np.abs(raw - st.session_state.depth_maps_raw[frame_idx-1])
    st.session_state.depth_changes.append(depth_change)
    st.session_state.movement_maps.append(depth_change)
    
    # Detectar movimiento
    movement_mask = detect_moving_objects(depth_change, st.session_state.depth_change_threshold)
    
    # Agrupar regiones de movimiento
    object_mask = cluster_movement_regions(movement_mask, st.session_state.min_pig_area)
    
    # Refinar máscara del cerdo
    pig_mask, bbox, centroid = refine_pig_mask(object_mask, raw)
    
    # Seguimiento del cerdo
    if centroid:
        tracked_centroid = track_pig(centroid, frame_idx)
    else:
        tracked_centroid = st.session_state.pig_centroids[-1] if st.session_state.pig_centroids else (0, 0)
    
    # Guardar resultados
    st.session_state.pig_masks.append(pig_mask if pig_mask is not None else np.zeros_like(raw, dtype=np.uint8))
    st.session_state.pig_contours.append(None)  # Se calcularán solo cuando sea necesario
    st.session_state.pig_bboxes.append(bbox)
    st.session_state.pig_centroids.append(tracked_centroid)
    
    # Analizar postura
    posture_data = analyze_pig_posture(raw, pig_mask)
    st.session_state.posture_analysis_results.append(posture_data)

def process_video_file(up_file):
    if not validate_video_file(up_file): return
    reset_app_state()
    model, device = load_model()
    with st.status("Procesando vídeo…", expanded=True) as s:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as t:
            t.write(up_file.read()); path = t.name
        frames = extract_frames(path); os.remove(path)
        if not frames: s.update(label="Extracción de frames fallida.", state="error"); return
        h, w = frames[0].shape[:2]; 
        st.session_state.total_frames = len(frames)
        prog = st.progress(0., "Analizando profundidad y movimiento…")
        for i, f in enumerate(frames):
            _store_frame(model, device, f, i)
            prog.progress((i + 1) / len(frames), f"Analizando frame {i+1}/{len(frames)}")
        st.session_state.video_processed = True
        s.update(label="Análisis completo.", state="complete"); st.rerun()

# ───────────────────── VISUALIZACIÓN ─────────────────────
def visualize_pig_detection(frame, pig_mask, centroid, bbox):
    """Crea una visualización de la detección del cerdo"""
    # Crear overlay de máscara
    mask_rgb = cv2.cvtColor(pig_mask, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(frame, 0.7, mask_rgb, 0.3, 0)
    
    # Dibujar bounding box
    if bbox:
        x, y, w, h = bbox
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Dibujar centroide
    if centroid and centroid != (0, 0):
        cv2.circle(overlay, centroid, 8, (0, 0, 255), -1)
    
    return overlay

# ───────────────────── APLICACIÓN PRINCIPAL ─────────────────────
init_session_state()
load_css()

st.markdown("""
    <h1 style='text-align:center; margin-bottom:0.25rem;'>
        <span style='color:#2c3e50;'>Pig</span>
        <span style='color:#e74c3c;'>Depth</span>
        <span style='color:#2c3e50;'>Tracker</span>
    </h1>
    <p style='text-align:center; color:#7f8c8d; margin-bottom:2rem;'>
        Detección y Análisis de Cerdos mediante Cambios de Profundidad
    </p>
""", unsafe_allow_html=True)

# --- PANTALLA INICIAL ---
if not st.session_state.video_processed:
    cols = st.columns([1, 1.5, 1])
    with cols[1]:
        with card("Iniciar Análisis"):
            tab_up, tab_live = st.tabs(["Subir Vídeo", "Grabación en Directo"])
            
            with tab_up:
                up = st.file_uploader("Selecciona un vídeo de cámara cenital", 
                                      type=SUPPORTED_FORMATS, 
                                      label_visibility="collapsed")
                
                if up:
                    st.session_state.depth_change_threshold = st.slider(
                        "Umbral de Cambio de Profundidad",
                        min_value=0.001,
                        max_value=0.1,
                        value=DEPTH_CHANGE_THRESHOLD,
                        step=0.001,
                        format="%.3f",
                        help="Ajusta la sensibilidad para detectar cambios de profundidad"
                    )
                    
                    st.session_state.min_pig_area = st.slider(
                        "Área Mínima del Cerdo (píxeles)",
                        min_value=1000,
                        max_value=30000,
                        value=MIN_PIG_AREA,
                        step=500
                    )
                    
                    if st.button("Analizar Vídeo", use_container_width=True, type="primary"):
                        process_video_file(up)
            
            with tab_live:
                st.info("Funcionalidad de grabación en directo disponible en próximas versiones")
                st.image("https://via.placeholder.com/600x300?text=Cámara+No+Disponible", 
                         use_container_width=True)

        st.markdown(f"""
            <div style='background:#f1f8ff;padding:1rem;border-radius:8px;border-left:3px solid #3498db;'>
                <b>Requisitos del Sistema</b>
                <p style='margin-bottom:0;'>
                    • GPU con CUDA • Vídeo ≤{MAX_VIDEO_DURATION}s • 
                    Archivo ≤{MAX_VIDEO_SIZE_MB} MB • Visión cenital clara
                </p>
            </div>
        """, unsafe_allow_html=True)

# --- PANTALLA DE RESULTADOS ---
else:
    with st.container():
        h1, h2 = st.columns([1, 0.2])
        h1.markdown(f"""
            <h2>Resultados del Análisis</h2>
            <p style='color:#7f8c8d;'>
                {st.session_state.total_frames} frames procesados | 
                Umbral de cambio: {st.session_state.depth_change_threshold:.3f}
            </p>
        """, unsafe_allow_html=True)
        
        h2.button("Nuevo Análisis", on_click=reset_app_state, use_container_width=True)
        
        tab_view, tab_ana = st.tabs(["Visor", "Análisis"])
        
        with tab_view:
            if st.session_state.original_frames:
                with card("Visor de Secuencia"):
                    c1, c2, c3 = st.columns([0.15, 1, 0.2])
                    c1.button("▶️ Play" if not st.session_state.playing else "⏸️ Pausa", 
                             on_click=lambda: st.session_state.update(playing=not st.session_state.playing), 
                             use_container_width=True)
                    
                    val = c2.slider("Frame", 0, st.session_state.total_frames - 1, 
                                   st.session_state.current_frame_index, 
                                   label_visibility="collapsed")
                    
                    if val != st.session_state.current_frame_index: 
                        st.session_state.current_frame_index = val
                        st.session_state.playing = False
                    
                    c3.markdown(f"<p style='text-align:center; padding-top:10px; color:#7f8c8d;'>{val + 1} / {st.session_state.total_frames}</p>", 
                               unsafe_allow_html=True)
                    
                    if st.session_state.total_frames > 0:
                        frame_idx = st.session_state.current_frame_index
                        frame = st.session_state.original_frames[frame_idx]
                        depth = st.session_state.depth_maps_colored[frame_idx]
                        pig_mask = st.session_state.pig_masks[frame_idx]
                        centroid = st.session_state.pig_centroids[frame_idx]
                        bbox = st.session_state.pig_bboxes[frame_idx]
                        
                        # Visualización de detección
                        detection_viz = visualize_pig_detection(frame, pig_mask, centroid, bbox)
                        
                        # Visualización de movimiento
                        movement_viz = None
                        if frame_idx > 0 and frame_idx < len(st.session_state.movement_maps):
                            movement_map = st.session_state.movement_maps[frame_idx-1]
                            movement_viz = cv2.normalize(movement_map, None, 0, 255, cv2.NORM_MINMAX)
                            movement_viz = cv2.applyColorMap(movement_viz.astype(np.uint8), cv2.COLORMAP_JET)
                        
                        # Mostrar resultados
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(detection_viz, caption="Detección del Cerdo", 
                                    use_container_width=True, clamp=True)
                            
                            if movement_viz is not None:
                                st.image(movement_viz, caption="Mapa de Movimiento (Cambios de Profundidad)", 
                                        use_container_width=True, clamp=True,
                                        output_format="JPEG")
                        
                        with col2:
                            st.image(depth, caption="Mapa de Profundidad", 
                                    use_container_width=True)
                            
                            # Mostrar análisis de postura si está disponible
                            if frame_idx < len(st.session_state.posture_analysis_results):
                                posture = st.session_state.posture_analysis_results[frame_idx]
                                if posture:
                                    st.markdown("**Análisis de Postura**")
                                    cols = st.columns(2)
                                    cols[0].metric("Asimetría", f"{posture['asymmetry']:.4f}")
                                    cols[1].metric("Centroide", f"({posture['centroid'][0]}, {posture['centroid'][1]})")
        
        with tab_ana:
            if st.session_state.original_frames and st.session_state.posture_analysis_results:
                # Análisis de asimetría a lo largo del tiempo
                with card("Evolución de la Asimetría"):
                    asymmetry_data = []
                    for i, posture in enumerate(st.session_state.posture_analysis_results):
                        if posture:
                            asymmetry_data.append({
                                "frame": i,
                                "asymmetry": posture["asymmetry"],
                                "left_depth": posture["left_depth"],
                                "right_depth": posture["right_depth"]
                            })
                    
                    if asymmetry_data:
                        df = pd.DataFrame(asymmetry_data)
                        
                        # Calcular umbral de alerta
                        mean_asym = df["asymmetry"].mean()
                        std_asym = df["asymmetry"].std()
                        alert_threshold = mean_asym + 2 * std_asym
                        
                        # Gráfico de asimetría
                        st.line_chart(df.set_index("frame")["asymmetry"])
                        
                        # Gráfico de profundidades izquierda/derecha
                        st.line_chart(df.set_index("frame")[["left_depth", "right_depth"]])
                        
                        # Detectar frames con posible problema
                        alert_frames = df[df["asymmetry"] > alert_threshold]
                        if not alert_frames.empty:
                            st.warning(f"Se detectaron posibles problemas de postura en {len(alert_frames)} frames")
                            st.write("Frames con asimetría significativa:")
                            st.dataframe(alert_frames)
                
                # Análisis de trayectoria
                with card("Trayectoria del Cerdo"):
                    if st.session_state.pig_centroids:
                        centroids = [c for c in st.session_state.pig_centroids if c != (0, 0)]
                        if centroids:
                            df_traj = pd.DataFrame(centroids, columns=["x", "y"])
                            df_traj["frame"] = df_traj.index
                            
                            # Crear gráfico de trayectoria
                            st.scatter_chart(df_traj, x="x", y="y")
                            
                            # Calcular distancia recorrida
                            total_distance = 0
                            for i in range(1, len(centroids)):
                                x1, y1 = centroids[i-1]
                                x2, y2 = centroids[i]
                                distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                                total_distance += distance
                            
                            st.metric("Distancia Total Recorrida", f"{total_distance:.2f} píxeles")

# ───────────────────── BUCLE DE ACTUALIZACIÓN ─────────────────────
if st.session_state.get("playing"):
    time.sleep(0.05)
    if st.session_state.total_frames > 0:
        next_idx = (st.session_state.current_frame_index + 1) % st.session_state.total_frames
        st.session_state.current_frame_index = next_idx
        st.rerun()