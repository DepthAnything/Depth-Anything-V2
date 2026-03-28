
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import torch
import os
import tempfile
import time
import io
from PIL import Image
from utils import load_css, card
from state import init_session_state, reset_app_state
from model import load_model
from processing import process_video_file, validate_video_file
from visualization import visualize_improved_detection, render_depth_analysis, visualize_sow_detection, show_symmetry_analysis, show_silhouette_analysis, show_3d_depth_visualization, show_regional_analysis
from config import SUPPORTED_FORMATS, DEPTH_CHANGE_THRESHOLD, MIN_PIG_AREA

try:
    from depth_anything_v2.dpt import DepthAnythingV2
except ImportError:
    st.error("Dependencia no encontrada: 'depth_anything_v2'. Por favor, instálala para continuar.")
    st.code("pip install git+https://github.com/LiheYoung/Depth-Anything-V2.git")
    st.stop()

# --- Configuración de la app ---
st.set_page_config(
    page_title="PigDepthTracker Pro",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={"About": "### Sistema avanzado de análisis de cerdos mediante visión por computador"}
)
init_session_state()
load_css()

# --- Encabezado principal ---
st.markdown("""
    <div style='text-align:center; margin-bottom:1.5rem;'>
        <h1><span style='color:#2c3e50;'>Pig</span><span style='color:#e74c3c;'>Depth</span><span style='color:#2c3e50;'>Tracker Pro</span></h1>
        <p style='color:#7f8c8d;'>Sistema avanzado de monitoreo porcinocéntrico</p>
        <div style='display:inline-block; background:#e74c3c20; color:#e74c3c; padding:0.25rem 0.75rem; border-radius:12px; font-size:0.85rem;
                    border:1px solid #e74c3c40; margin-bottom:1rem;'>🔬 Modo de análisis mejorado activado</div>
    </div>
""", unsafe_allow_html=True)

# --- SUBIDA o RESULTADOS ---
if not st.session_state.video_processed:
    cols = st.columns([1, 1.5, 1])
    with cols[1]:
        with card("Iniciar Análisis Avanzado"):
            tab_up, tab_live = st.tabs(["Subir Vídeo", "Grabación en Directo"])
            with tab_up:
                up = st.file_uploader("Selecciona un vídeo de cámara cenital", type=SUPPORTED_FORMATS, label_visibility="collapsed")
                if up:
                    with st.expander("⚙️ Configuración Avanzada", expanded=True):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.session_state.depth_change_threshold = st.slider(
                                "Umbral de Cambio de Profundidad", 0.001, 0.1, DEPTH_CHANGE_THRESHOLD, 0.001, format="%.3f"
                            )
                        with c2:
                            st.session_state.min_pig_area = st.slider(
                                "Área Mínima del Cerdo (px²)", 1000, 30000, MIN_PIG_AREA, step=500
                            )
                    if st.button("🚀 Iniciar Análisis", use_container_width=True, type="primary"):
                        with st.spinner("Procesando vídeo..."):
                            process_video_file(up)
            with tab_live:
                st.info("**Próximamente:** conexión directa con cámaras IP. Contáctanos para acceso anticipado.")
else:
    st.markdown(f"""
        <div style='background:#f8f9fa;padding:1rem;border-radius:8px;border-left:4px solid #1abc9c;'>
            <b style='color:#2c3e50;'>Resumen:</b> {st.session_state.total_frames} frames procesados | 
            Resolución: {st.session_state.original_frames[0].shape[1]}×{st.session_state.original_frames[0].shape[0]} px |
            Umbral: {st.session_state.depth_change_threshold:.3f} | Área mínima: {st.session_state.min_pig_area}px²
        </div>
    """, unsafe_allow_html=True)

    if st.button("🔄 Nuevo Análisis", on_click=reset_app_state):
        st.experimental_rerun()

    tab_view, tab_analysis, tab_export = st.tabs(["Visor", "Análisis", "Exportar"])

    # --- Visor de secuencia ---
    with tab_view:
        with card("Visor de Secuencia"):
            c1, c2, c3, c4 = st.columns([0.1, 0.8, 0.1, 0.2])
            c1.button("⏮️", on_click=lambda: st.session_state.update(current_frame_index=0), use_container_width=True)
            play_pause = c2.empty()
            if play_pause.button("▶️ Play" if not st.session_state.playing else "⏸️ Pausa", use_container_width=True, key="play_button"):
                st.session_state.playing = not st.session_state.playing
            c3.button("⏭️", on_click=lambda: st.session_state.update(current_frame_index=st.session_state.total_frames-1), use_container_width=True)
            frame_slider = c4.slider("Frame", 0, st.session_state.total_frames - 1, st.session_state.current_frame_index, label_visibility="collapsed")
            if frame_slider != st.session_state.current_frame_index:
                st.session_state.current_frame_index = frame_slider
                st.session_state.playing = False

                idx = st.session_state.current_frame_index
                col1, col2 = st.columns(2)
                
                with col1:
                    frame = st.session_state.original_frames[idx]
                    depth_map = st.session_state.depth_maps_colored[idx]
                    sow_mask = st.session_state.pig_masks[idx] if idx < len(st.session_state.pig_masks) else None
                    bbox = st.session_state.pig_bboxes[idx] if idx < len(st.session_state.pig_bboxes) else None
                    centroid = st.session_state.pig_centroids[idx] if idx < len(st.session_state.pig_centroids) else None
                    analysis = st.session_state.depth_analysis[idx] if idx < len(st.session_state.depth_analysis) else None
                    
                    if sow_mask is not None:
                        vis = visualize_sow_detection(
                            frame, 
                            sow_mask,
                            depth_map,
                            analysis
                        )
                        st.image(vis, caption=f"Frame {idx + 1}", use_container_width=True)
                    else:
                        st.warning("No sow detected in this frame")
                        st.image(frame, caption=f"Frame {idx + 1} (No Detection)", use_container_width=True)
                
                with col2:
                    show_silhouette_analysis(
                        frame,
                        depth_map,
                        sow_mask,
                        st.session_state.masked_depths[idx] if idx < len(st.session_state.masked_depths) else None,
                        st.session_state.anomaly_maps[idx] if idx < len(st.session_state.anomaly_maps) else None
                    )
                
                # Add new visualizations
                with card("Análisis de Profundidad 3D"):
                    show_3d_depth_visualization(
                        st.session_state.depth_maps_raw[idx],
                        st.session_state.pig_masks[idx]
                    )
                
                with card("Análisis Regional"):
                    col1, col2 = st.columns(2)
                    with col1:
                        show_regional_analysis(st.session_state.depth_analysis[idx])
                    with col2:
                        show_symmetry_analysis(st.session_state.depth_analysis[idx])

    with tab_analysis:
        render_depth_analysis(st.session_state.depth_analysis)

    with tab_export:
        with card("Exportar Resultados"):
            st.markdown("Descarga los datos procesados en formato CSV, JSON o Excel.")
            fmt = st.radio("Formato", ["CSV", "JSON", "Excel"], horizontal=True)
            if st.button("📤 Generar Archivo"):
                rows = []
                for i in range(st.session_state.total_frames):
                    rows.append({
                        "frame": i,
                        "centroid_x": st.session_state.pig_centroids[i][0],
                        "centroid_y": st.session_state.pig_centroids[i][1],
                        "bbox_area": (st.session_state.pig_bboxes[i][2] * st.session_state.pig_bboxes[i][3]) if st.session_state.pig_bboxes[i] else 0,
                        "mean_depth": st.session_state.depth_analysis[i]["mean_depth"] if st.session_state.depth_analysis[i] else None,
                        "asymmetry": st.session_state.posture_analysis_results[i]["asymmetry"] if i < len(st.session_state.posture_analysis_results) else None,
                        "depth_matrix": st.session_state.depth_segmented_matrices[i].tolist() if i < len(st.session_state.depth_segmented_matrices) else None
                    })
                df = pd.DataFrame(rows)
                if fmt == "CSV":
                    st.download_button("⬇️ Descargar CSV", df.to_csv(index=False).encode(), "resultados.csv", "text/csv")
                elif fmt == "JSON":
                    st.download_button("⬇️ Descargar JSON", df.to_json(indent=2).encode(), "resultados.json", "application/json")
                else:
                    excel = io.BytesIO()
                    with pd.ExcelWriter(excel, engine="xlsxwriter") as writer:
                        df.to_excel(writer, index=False, sheet_name="Resultados")
                    excel.seek(0)
                    st.download_button("⬇️ Descargar Excel", excel, "resultados.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if st.session_state.get("playing"):
    time.sleep(0.05)
    st.session_state.current_frame_index = (st.session_state.current_frame_index + 1) % st.session_state.total_frames
    st.rerun()
