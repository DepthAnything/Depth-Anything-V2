from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import torch
import numpy as np
import cv2
import os
import base64
from pathlib import Path
from typing import List
from datetime import datetime
from fastapi.staticfiles import StaticFiles

# Model import
try:
    from depth_anything_v2.dpt import DepthAnythingV2
except ImportError:
    raise ImportError("Could not import DepthAnythingV2 - check your model implementation")

app = FastAPI()
app.mount("/session_data", StaticFiles(directory="session_data"), name="session_data")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
device = None
DATA_DIR = Path("session_data")
DATA_DIR.mkdir(exist_ok=True)

# Recording state
RECORDING_STATE = {
    "recording": False,
    "frame_count": 0,
    "session_id": "",
}

# Models
class Point(BaseModel):
    x: int
    y: int

class PointAnalysisOptions(BaseModel):
    points: List[Point]

class AnalysisOptions(BaseModel):
    noise_threshold: float = 0.01

# Middleware for additional CORS headers
@app.middleware("http")
async def add_cache_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/session_data"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# Endpoints
@app.post("/upload")
async def upload_raw_frame(request: Request):
    try:
        img_bytes = await request.body()

        if not RECORDING_STATE["recording"]:
            return JSONResponse(content={"message": "Not recording. Frame ignored."}, status_code=200)

        session_id = RECORDING_STATE["session_id"]
        frame_idx = RECORDING_STATE["frame_count"]
        session_dir = DATA_DIR / session_id / "recorded_frames"
        os.makedirs(session_dir, exist_ok=True)

        frame_path = session_dir / f"frame_{frame_idx:05d}.jpg"
        
        # Convert and save as JPEG
        img = Image.open(BytesIO(img_bytes))
        img = img.convert('RGB')
        img.save(frame_path, 'JPEG', quality=95)
        
        RECORDING_STATE["frame_count"] += 1
        return JSONResponse(content={"message": f"Frame {frame_idx} saved."}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/start-recording")
async def start_recording(session_id: str = Form(...)):
    RECORDING_STATE.update(recording=True, frame_count=0, session_id=session_id)

    session_dir = DATA_DIR / session_id / "recorded_frames"
    os.makedirs(session_dir, exist_ok=True)
    for f in session_dir.glob("frame_*.jpg"):
        f.unlink(missing_ok=True)

    # 👉 devuelve session_id
    return {
        "status": "success",
        "message": f"Recording started for session {session_id}",
        "session_id": session_id          #  ← ESTA LÍNEA ES IMPRESCINDIBLE
    }
@app.post("/stop-recording/{session_id}")
async def stop_recording(session_id: str):
    try:
        if not RECORDING_STATE["recording"]:
            return JSONResponse(
                content={"status": "error", "message": "No hay grabación activa"},
                status_code=400
            )

        if RECORDING_STATE["session_id"] != session_id:
            return JSONResponse(
                content={"status": "error", "message": "Session ID no coincide"},
                status_code=400
            )

        RECORDING_STATE["recording"] = False
        frame_count = RECORDING_STATE["frame_count"]
        
        return JSONResponse(
            content={
                "status": "success",
                "message": "Grabación finalizada",
                "session_id": session_id,
                "frame_count": frame_count,
                "frame_base_url": f"/session_data/{session_id}/recorded_frames/frame_"
            },
            status_code=200
        )

    except Exception as e:
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )

@app.get("/session_data/{session_id}/recorded_frames/frame_{frame_number:05d}.jpg")
async def get_frame(session_id: str, frame_number: int):
    frame_path = DATA_DIR / session_id / "recorded_frames" / f"frame_{frame_number:05d}.jpg"
    
    if not frame_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Frame not found at {frame_path}"
        )
    
    return FileResponse(
        frame_path,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache"}
    )

@app.get("/debug-frames/{session_id}")
async def debug_frames(session_id: str):
    frames_dir = DATA_DIR / session_id / "recorded_frames"
    
    if not frames_dir.exists():
        return {"error": f"Directory not found: {frames_dir}"}
    
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    
    sample_frames = []
    for i, frame_path in enumerate(frame_files[:3]):
        sample_frames.append({
            "number": i,
            "name": frame_path.name,
            "exists": frame_path.exists(),
            "size": frame_path.stat().st_size if frame_path.exists() else 0,
            "path": str(frame_path)
        })
    
    return {
        "session_id": session_id,
        "total_frames": len(frame_files),
        "sample_frames": sample_frames,
        "directory": str(frames_dir)
    }
    
@app.get("/get-frames/{session_id}")
async def get_frames_info(session_id: str):
    frames_dir = DATA_DIR / session_id / "recorded_frames"
    if not frames_dir.exists():
        raise HTTPException(status_code=404, detail="Session directory not found")
    
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    return {
        "frame_count": len(frame_files),
        "frames": [f.name for f in frame_files]
    }

def load_model(encoder="vitl"):
    global model, device
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        raise RuntimeError("GPU is required for this application, but CUDA is not available.")
    
    print(f"Using device: {device}")
    
    cfg = {"encoder": encoder, "features": 256, "out_channels": [256, 512, 1024, 1024]}
    model = DepthAnythingV2(**cfg)
    
    checkpoint_path = f"checkpoints/depth_anything_v2_{encoder}.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    
    print(f"Loading weights from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()
    print("Model loaded successfully")

@app.on_event("startup")
async def startup_event():
    try:
        load_model()
        print("\n--- Directory Verification ---")
        print(f"Data directory: {DATA_DIR.resolve()}")
        print(f"Directory exists: {DATA_DIR.exists()}")
        print(f"Write permissions: {os.access(DATA_DIR, os.W_OK)}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

@app.post("/predict")
async def predict_depth(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    frame_index: int = Form(...)
):
    try:
        session_dir = DATA_DIR / session_id
        session_dir.mkdir(exist_ok=True)

        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        with torch.no_grad():
            depth = model.infer_image(image_bgr) 
        
        raw_depth_np = depth
        save_path = session_dir / f"frame_{frame_index:05d}.npy"
        np.save(save_path, raw_depth_np)

        dmin, dmax = float(depth.min()), float(depth.max())
        norm_depth = ((depth - dmin) / (dmax - dmin + 1e-6)).clip(0, 1)
        
        return {
            "message": f"Frame {frame_index} processed and saved to {save_path}",
            "depth": norm_depth.tolist(),
            "min": dmin, "max": dmax, "mean": float(depth.mean()), "std": float(depth.std())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/{session_id}")
async def analyze_session(session_id: str, options: AnalysisOptions):
    session_dir = DATA_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session ID not found.")

    frame_files = sorted(session_dir.glob("frame_*.npy"))
    if len(frame_files) < 2:
        raise HTTPException(status_code=400, detail="Not enough frames to perform an analysis (at least 2 required).")

    reference_depth = np.load(frame_files[0])
    analysis_results = []
    
    for i in range(1, len(frame_files)):
        current_depth = np.load(frame_files[i])
        depth_difference = current_depth - reference_depth
        significant_diff = np.where(np.abs(depth_difference) > options.noise_threshold, depth_difference, 0)
        
        total_pixels = significant_diff.size
        changed_pixels = np.count_nonzero(significant_diff)
        
        frame_metrics = {
            "frame_index": i,
            "volume_change": float(significant_diff.sum()),
            "added_volume": float(significant_diff[significant_diff > 0].sum()),
            "removed_volume": float(significant_diff[significant_diff < 0].sum()),
            "mean_depth_change": float(significant_diff.mean()),
            "changed_area_percent": (changed_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        }
        analysis_results.append(frame_metrics)
        
    return {"session_id": session_id, "analysis": analysis_results}

@app.post("/analyze-points/{session_id}")
async def analyze_points(session_id: str, options: PointAnalysisOptions):
    session_dir = DATA_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session ID not found.")

    frame_files = sorted(session_dir.glob("frame_*.npy"))
    if not frame_files:
        raise HTTPException(status_code=400, detail="No frames found for this session.")

    try:
        all_frames_data = np.array([np.load(f) for f in frame_files])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading frame data: {e}")

    results = []
    _, height, width = all_frames_data.shape

    for point in options.points:
        if not (0 <= point.x < width and 0 <= point.y < height):
            continue

        depth_evolution = all_frames_data[:, point.y, point.x]
        
        results.append({
            "point": point.dict(),
            "label": f"Point ({point.x}, {point.y})",
            "depth_values": depth_evolution.tolist()
        })

    return {"session_id": session_id, "point_analysis": results}

@app.get("/")
async def serve_frontend():
    html_content = """<!DOCTYPE html>
    <html>
    <head>
        <title>DepthVision Processor</title>
    </head>
    <body>
        <h1>DepthVision Processor is running</h1>
        <p>Use the frontend application to interact with this service.</p>
    </body>
    </html>"""
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)