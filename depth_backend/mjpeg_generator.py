import time
from fastapi.responses import StreamingResponse
from .image_store_depth import depth_store

def mjpeg_stream(kind: str = "depth"):
    """Generate MJPEG stream with optimized frame rate and quality."""
    getter = depth_store.get_frame_depth if kind == "depth" else depth_store.get_frame_raw
    
    async def generate():
        last_frame = None
        last_sent = 0
        min_interval = 1/30  # 30 FPS max
        
        while True:
            current_time = time.time()
            frame = getter()
            
            if frame and (current_time - last_sent) >= min_interval:
                if frame != last_frame:  # Only send if frame changed
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(frame)).encode() + b"\r\n"
                        b"\r\n" + frame + b"\r\n"
                    )
                    last_sent = current_time
                    last_frame = frame
                else:
                    time.sleep(0.001)  # Small sleep to prevent busy waiting
            else:
                time.sleep(0.001)
    
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no"  # Disable buffering for nginx
        }
    )