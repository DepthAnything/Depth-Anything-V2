# pig_depth_tracker/visualization.py
import cv2
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
from utils import card

def visualize_improved_detection(frame, pig_mask, centroid, bbox, depth_analysis):
    overlay = frame.copy()

    if pig_mask is not None:
        contours, _ = cv2.findContours(pig_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    if bbox:
        x, y, w, h = bbox
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0), 2)

    if centroid:
        cv2.circle(overlay, centroid, 5, (0, 0, 255), -1)

    if depth_analysis:
        label = f"Media: {depth_analysis['mean_depth']:.2f} | Std: {depth_analysis['std_depth']:.2f}"
        cv2.putText(overlay, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return overlay

def visualize_sow_detection(frame, mask, colored_depth, depth_analysis):
    """Create comprehensive visualization"""
    vis = frame.copy()
    
    if mask is not None:
        # Draw contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
        
        # Draw regions
        y_coords, x_coords = np.where(mask == 255)
        height = y_coords.max() - y_coords.min()
        
        for i, (name, y_range) in enumerate([('Head', 0.3), ('Middle', 0.7), ('Rear', 1.0)]):
            y_pos = int(y_coords.min() + y_range * height)
            cv2.line(vis, 
                    (x_coords.min(), y_pos),
                    (x_coords.max(), y_pos),
                    (255, 0, 0), 1)
            cv2.putText(vis, name, 
                       (x_coords.min() + 10, y_pos - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    if depth_analysis:
        # Display depth info
        info_text = f"Mean: {depth_analysis['mean_depth']:.3f} | " \
                   f"Std: {depth_analysis['std_depth']:.3f} | " \
                   f"Area: {depth_analysis['area_pixels']}px"
        cv2.putText(vis, info_text, 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 2)
    
    return vis

def show_silhouette_analysis(frame, depth_map, pig_mask, masked_depth, anomaly_map=None):
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(frame, caption="Original Frame")
        if pig_mask is not None:
            st.image(pig_mask, caption="Sow Segmentation Mask", clamp=True)
        else:
            st.warning("No sow detected in this frame")

    with col2:
        st.image(depth_map, caption="Depth Map", clamp=True)
        if masked_depth is not None:
            st.image(masked_depth, caption="Sow Depth Analysis", clamp=True)


def render_depth_analysis(depth_analysis):
    """Render depth analysis visualizations with proper error handling"""
    if not depth_analysis or len(depth_analysis) == 0:
        st.warning("No depth analysis data available")
        return

    # Create DataFrame with proper validation
    valid_entries = [x for x in depth_analysis if x is not None and 'mean_depth' in x]
    if not valid_entries:
        st.error("No valid depth analysis entries containing 'mean_depth'")
        return

    df = pd.DataFrame(valid_entries)

    with st.expander("📊 Advanced Depth Analysis", expanded=True):
        tab1, tab2, tab3 = st.tabs(["Distributions", "Temporal Evolution", "Percentiles"])

        with tab1:
            st.subheader("Global Distribution")
            try:
                fig = px.histogram(df, x="mean_depth", nbins=30, 
                                  title="Mean Depth Distribution")
                st.plotly_chart(fig, use_container_width=True)
            except ValueError as e:
                st.error(f"Could not create histogram: {str(e)}")
                st.write("Debug data:", df.columns.tolist())

            try:
                fig2 = px.box(df, y="std_depth", title="Depth Dispersion (STD)")
                st.plotly_chart(fig2, use_container_width=True)
            except ValueError as e:
                st.error(f"Could not create box plot: {str(e)}")

        with tab2:
            st.subheader("Temporal Curves")
            try:
                fig = px.line(df, y=["mean_depth", "min_depth", "max_depth"], 
                              markers=True, title="Depth Evolution")
                st.plotly_chart(fig, use_container_width=True)
            except ValueError as e:
                st.error(f"Could not create line plot: {str(e)}")

            if "anomaly_pixels" in df.columns:
                try:
                    fig2 = px.line(df, y="anomaly_pixels", markers=True,
                                  title="Percentage of Anomalous Pixels")
                    st.plotly_chart(fig2, use_container_width=True)
                except ValueError as e:
                    st.error(f"Could not create anomaly plot: {str(e)}")

        with tab3:
            st.subheader("Percentile Evolution")
            percentiles = ["percentiles.5", "percentiles.25", "percentiles.50", 
                         "percentiles.75", "percentiles.95"]
            valid_percentiles = [p for p in percentiles if p in df.columns]
            
            if valid_percentiles:
                try:
                    fig = px.line(df, y=valid_percentiles, markers=True,
                                 title="Depth Percentile Evolution")
                    st.plotly_chart(fig, use_container_width=True)
                except ValueError as e:
                    st.error(f"Could not create percentile plot: {str(e)}")
            else:
                st.warning("No valid percentile data available")

# Update visualization.py
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def show_3d_depth_visualization(depth_map, pig_mask):
    """Create interactive 3D depth visualization"""
    y, x = np.where(pig_mask == 255)
    z = depth_map[y, x]
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=3,
            color=z,
            colorscale='Viridis',
            opacity=0.8
        )
    )])
    
    fig.update_layout(
        scene=dict(
            xaxis_title='X Position',
            yaxis_title='Y Position',
            zaxis_title='Depth',
            aspectratio=dict(x=1, y=1, z=0.7)
        ),
        title='3D Depth Visualization of Pig',
        height=700
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_regional_analysis(depth_analysis):
    """Visualize regional depth differences"""
    if not depth_analysis:
        return
    
    regions = ['head', 'middle', 'rear', 'left', 'right']
    region_data = []
    
    for region in regions:
        if f'{region}_mean' in depth_analysis:
            region_data.append({
                'region': region,
                'mean': depth_analysis[f'{region}_mean'],
                'std': depth_analysis[f'{region}_std']
            })
    
    if not region_data:
        return
    
    df = pd.DataFrame(region_data)
    
    fig = px.bar(df, x='region', y='mean', error_y='std',
                 title='Regional Depth Analysis',
                 labels={'mean': 'Mean Depth', 'region': 'Body Region'})
    
    st.plotly_chart(fig, use_container_width=True)

def show_symmetry_analysis(depth_analysis):
    """Visualize left-right symmetry"""
    if not depth_analysis or 'symmetry_score' not in depth_analysis:
        return
    
    fig = go.Figure()
    
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=depth_analysis['symmetry_score'],
        title={'text': "Symmetry Score (lower is better)"},
        gauge={'axis': {'range': [0, 1]},
               'steps': [
                   {'range': [0, 0.1], 'color': "green"},
                   {'range': [0.1, 0.3], 'color': "yellow"},
                   {'range': [0.3, 1], 'color': "red"}],
               'threshold': {'line': {'color': "black", 'width': 4},
                             'thickness': 0.75,
                             'value': 0.2}}))
    
    st.plotly_chart(fig, use_container_width=True)
    
