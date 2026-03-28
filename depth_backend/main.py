import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time

from .image_store_depth import depth_store
from .mjpeg_generator import mjpeg_stream

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting DepthVision Backend")
    yield
    # Shutdown
    print("Shutting down DepthVision Backend")

app = FastAPI(
    title="DepthVision Backend",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

def _to_float(x):
    if isinstance(x, np.ndarray):
        return float(x.item())
    if isinstance(x, (np.floating,)):
        return float(x)
    return x

@app.post("/upload")
async def upload(req: Request):
    try:
        start_time = time.time()
        depth_store.set_frame(await req.body())
        processing_time = (time.time() - start_time) * 1000
        return {
            "status": "ok",
            "processing_time_ms": round(processing_time, 2)
        }
    except Exception as e:
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )

@app.get("/mjpeg/raw")
async def mjpeg_raw():
    return StreamingResponse(
        mjpeg_stream("raw"),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )

@app.get("/mjpeg/depth")
async def mjpeg_depth():
    return StreamingResponse(
        mjpeg_stream("depth"),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )

@app.post("/record/start")
def record_start():
    depth_store.start()
    return {"status": "started"}

@app.post("/record/stop")
def record_stop():
    depth_store.stop()
    return {"status": "stopped"}

@app.get("/record/status")
def record_status():
    return {"recording": depth_store.is_recording()}

@app.get("/metrics/latest")
def metrics_latest():
    mn, mx, me, sd = depth_store.last_stats()
    return {
        "min": _to_float(mn),
        "max": _to_float(mx),
        "mean": _to_float(me),
        "std": _to_float(sd)
    }

@app.get("/metrics/timeseries")
def metrics_timeseries():
    data = depth_store.stats_timeseries()
    if not data:
        return {"t": [], "min": [], "max": [], "mean": [], "std": []}
    
    t, mn, mx, me, sd = zip(*data)
    return {
        "t": [float(ts) for ts in t],
        "min": [_to_float(v) for v in mn],
        "max": [_to_float(v) for v in mx],
        "mean": [_to_float(v) for v in me],
        "std": [_to_float(v) for v in sd]
    }

@app.get("/metrics/hist")
def metrics_hist():
    edges, counts = depth_store.hist()
    return {"edges": edges, "counts": counts}

@app.get("/metrics/csv")
def metrics_csv():
    path = depth_store.stats_to_csv()
    return FileResponse(
        path,
        media_type="text/csv",
        filename="depth_metrics.csv",
        headers={"Content-Disposition": "attachment; filename=depth_metrics.csv"}
    )

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": time.time()}