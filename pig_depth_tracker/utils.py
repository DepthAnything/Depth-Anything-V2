# pig_depth_tracker/utils.py
import streamlit as st
from contextlib import contextmanager

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