# ============================================
# IMPORTS
# ============================================

from fastapi import Body, FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import cv2
import numpy as np
from ultralytics import YOLO
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image
import threading
import queue
import time

# ============================================
# CONFIGURATION
# ============================================

app = FastAPI(title="APVV - PPE Detection Service")

# CORS - Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class CameraStartRequest(BaseModel):
    camera_id: int = 0


# Camera settings
CAMERA_INDEX = 0  # 0 for webcam, or RTSP URL
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# Detection settings
CONFIDENCE_THRESHOLD = 0.5
PERSON_CONFIDENCE = 0.4  # Lower threshold for faster detection

# Active connections
active_connections: List[WebSocket] = []

# Frame queue for processing
frame_queue = queue.Queue(maxsize=5)
result_queue = queue.Queue(maxsize=5)

# Shared latest frame for streaming (keeps only newest frame)
latest_frame = None
latest_detections = []
latest_timestamp = None
latest_lock = threading.Lock()
# Lock to protect model inference (YOLO models may not be fully thread-safe)
model_lock = threading.Lock()

# Diagnostics
last_detection_time = 0.0
last_read_time = 0.0
read_counter = 0
detect_counter = 0
last_log_time = time.time()

# ============================================
# AI MODEL - PPE DETECTION
# ============================================

class PPEDetector:
    """
    Safety Gear Detection using YOLOv8
    Detects: Helmets, Safety Vests, Persons without PPE
    """
    
    def __init__(self):
        print("Loading AI models...")
        
        # Load YOLOv8 for person detection
        self.person_model = YOLO('yolov8n.pt')  # Nano model for speed
        
        # For PPE detection, we'll use a custom approach:
        # 1. Detect person
        # 2. Extract head/torso regions
        # 3. Check for helmet (yellow/orange colors in head region)
        # 4. Check for vest (fluorescent colors in torso region)
        
        # In production, use a trained PPE model like:
        # self.ppe_model = YOLO('ppe_detection.pt')
        # Download from: https://github.com/roboflow/ppe-detection
        
        print("✓ Models loaded successfully")
        
        # Color ranges for PPE detection (HSV)
        self.helmet_colors = {
            'yellow': ([20, 100, 100], [30, 255, 255]),
            'orange': ([10, 100, 100], [20, 255, 255]),
            'white': ([0, 0, 200], [180, 30, 255])
        }
        
        self.vest_colors = {
            'fluorescent_yellow': ([20, 100, 100], [35, 255, 255]),
            'fluorescent_orange': ([5, 150, 150], [15, 255, 255]),
            'fluorescent_green': ([40, 100, 100], [80, 255, 255])
        }
    
    def detect_persons(self, frame):
        """Detect all persons in frame"""
        # Protect model inference with a lock to avoid concurrency issues
        with model_lock:
            results = self.person_model(frame, classes=[0], conf=PERSON_CONFIDENCE)
        return results[0]
    
    def check_helmet(self, head_region):
        """Check if helmet is present in head region using color detection"""
        if head_region.size == 0:
            return False, 0.0
        
        hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
        
        total_pixels = head_region.shape[0] * head_region.shape[1]
        max_coverage = 0
        
        for color_name, (lower, upper) in self.helmet_colors.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            coverage = np.sum(mask > 0) / total_pixels
            max_coverage = max(max_coverage, coverage)
        
        # If more than 15% of head region has helmet colors
        has_helmet = max_coverage > 0.15
        confidence = min(max_coverage * 3, 1.0)  # Scale to 0-1
        
        return has_helmet, confidence
    
    def check_vest(self, torso_region):
        """Check if safety vest is present in torso region"""
        if torso_region.size == 0:
            return False, 0.0
        
        hsv = cv2.cvtColor(torso_region, cv2.COLOR_BGR2HSV)
        
        total_pixels = torso_region.shape[0] * torso_region.shape[1]
        max_coverage = 0
        
        for color_name, (lower, upper) in self.vest_colors.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            coverage = np.sum(mask > 0) / total_pixels
            max_coverage = max(max_coverage, coverage)
        
        # If more than 20% of torso has vest colors
        has_vest = max_coverage > 0.20
        confidence = min(max_coverage * 2.5, 1.0)
        
        return has_vest, confidence
    
    def detect_ppe(self, frame):
        """
        Main detection function
        Returns list of detections with PPE status
        """
        detections = []
        
        # Detect persons
        person_results = self.detect_persons(frame)
        
        if person_results.boxes is None or len(person_results.boxes) == 0:
            return detections
        
        frame_height, frame_width = frame.shape[:2]
        
        for box in person_results.boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].cpu().numpy()
            
            # Ensure coordinates are within frame
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(frame_width, int(x2)), min(frame_height, int(y2))
            
            # Extract person region
            person_region = frame[y1:y2, x1:x2]
            
            if person_region.size == 0:
                continue
            
            person_height = y2 - y1
            person_width = x2 - x1
            
            # Extract head region (top 25% of person)
            head_height = int(person_height * 0.25)
            head_region = person_region[0:head_height, :]
            
            # Extract torso region (25% to 75% of person height)
            torso_start = int(person_height * 0.25)
            torso_end = int(person_height * 0.75)
            torso_region = person_region[torso_start:torso_end, :]
            
            # Check for helmet and vest
            has_helmet, helmet_conf = self.check_helmet(head_region)
            has_vest, vest_conf = self.check_vest(torso_region)
            
            # Create detection object
            detection = {
                'bbox': {
                    'x': float(x1),
                    'y': float(y1),
                    'width': float(x2 - x1),
                    'height': float(y2 - y1)
                },
                'person_confidence': float(confidence),
                'has_helmet': has_helmet,
                'helmet_confidence': float(helmet_conf),
                'has_vest': has_vest,
                'vest_confidence': float(vest_conf),
                'compliant': has_helmet and has_vest,
                'timestamp': datetime.now().isoformat()
            }
            
            detections.append(detection)
        
        return detections
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on frame"""
        annotated_frame = frame.copy()
        
        for det in detections:
            bbox = det['bbox']
            x1, y1 = int(bbox['x']), int(bbox['y'])
            x2, y2 = int(bbox['x'] + bbox['width']), int(bbox['y'] + bbox['height'])
            
            # Choose color based on compliance
            if det['compliant']:
                color = (0, 255, 0)  # Green - compliant
                label = "COMPLIANT"
            else:
                color = (0, 0, 255)  # Red - non-compliant
                label = "VIOLATION"
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, 
                         (x1, y1 - 25), 
                         (x1 + label_size[0], y1), 
                         color, -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label, 
                       (x1, y1 - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw PPE status
            status_y = y1 + 20
            
            # Helmet status
            helmet_text = f"Helmet: {'YES' if det['has_helmet'] else 'NO'}"
            helmet_color = (0, 255, 0) if det['has_helmet'] else (0, 0, 255)
            cv2.putText(annotated_frame, helmet_text,
                       (x1, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, helmet_color, 1)
            
            # Vest status
            vest_text = f"Vest: {'YES' if det['has_vest'] else 'NO'}"
            vest_color = (0, 255, 0) if det['has_vest'] else (0, 0, 255)
            cv2.putText(annotated_frame, vest_text,
                       (x1, status_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, vest_color, 1)
        
        return annotated_frame

# Initialize detector
detector = PPEDetector()

# ============================================
# CAMERA CAPTURE
# ============================================

class CameraManager:
    """Manages camera capture and processing"""
    
    def __init__(self):
        self.cap = None
        self.is_running = False
        self.current_camera = CAMERA_INDEX
    
    def start(self, camera_source=CAMERA_INDEX):
        """Start camera capture"""
        self.current_camera = camera_source
        self.cap = cv2.VideoCapture(camera_source)
        
        if not self.cap.isOpened():
            raise Exception(f"Cannot open camera: {camera_source}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, FPS)
        
        self.is_running = True
        print(f"✓ Camera started: {camera_source}")
    
    def stop(self):
        """Stop camera capture"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        print("Camera stopped")
    
    def read_frame(self):
        """Read a frame from camera"""
        if not self.cap or not self.is_running:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        return frame

camera_manager = CameraManager()

# ============================================
# VIDEO PROCESSING LOOP
# ============================================

def process_frames():
    """Background thread to process frames"""
    frame_count = 0
    while True:
        if not camera_manager.is_running:
            time.sleep(0.1)
            continue

        frame = camera_manager.read_frame()
        if frame is None:
            continue

        frame_count += 1
        # For performance, run the heavy PPE detection only on every 2nd frame.
        # Still always encode and publish the latest frame so the client sees
        # a smooth, continuously-updating feed.
        run_detection = (frame_count % 2 == 0)

        # Diagnostics: count reads
        global read_counter, detect_counter, last_read_time, last_log_time
        read_counter += 1
        now = time.time()

        # If running detection this frame, compute detections. Otherwise,
        # keep the last known detections so overlays persist until next pass.
        if run_detection:
            try:
                t0 = time.time()
                detections = detector.detect_ppe(frame)
                detect_counter += 1
                last_detection_time = time.time() - t0
                # log occasionally
                if time.time() - last_log_time > 5.0:
                    print(f"[diagnostic] reads={read_counter}, detects={detect_counter}, last_detect_time={last_detection_time:.3f}s")
                    last_log_time = time.time()
            except Exception as e:
                # On any detection error, log and reuse previous detections
                print(f"[warning] detect_ppe error: {e}")
                with latest_lock:
                    detections = list(latest_detections) if latest_detections is not None else []
        else:
            with latest_lock:
                detections = list(latest_detections) if latest_detections is not None else []

        # Draw detections (empty list will just return the original frame)
        annotated_frame = detector.draw_detections(frame, detections)

        # Convert to JPEG with lower quality for faster transmission
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        _, buffer = cv2.imencode('.jpg', annotated_frame, encode_param)
        frame_bytes = buffer.tobytes()

        # Update the shared latest frame (overwrite oldest immediately)
        try:
            with latest_lock:
                globals()['latest_frame'] = frame_bytes
                globals()['latest_detections'] = detections
                globals()['latest_timestamp'] = datetime.now().isoformat()
        except Exception:
            # In case of any unexpected error, continue processing next frame
            pass

        # Small sleep to avoid tight loop if camera.read is very fast
        time.sleep(0.001)

# Start processing thread
processing_thread = threading.Thread(target=process_frames, daemon=True)
processing_thread.start()

# ============================================
# WEBSOCKET - REAL-TIME STREAMING
# ============================================

@app.websocket("/ws/video")
async def websocket_video(websocket: WebSocket):
    """WebSocket endpoint for real-time video streaming"""
    await websocket.accept()
    active_connections.append(websocket)
    
    print(f"Client connected. Total connections: {len(active_connections)}")
    
    try:
        while True:
            # Read latest shared frame under lock and send if available
            frame_to_send = None
            detections_to_send = []
            ts_to_send = None

            with latest_lock:
                if latest_frame is not None:
                    frame_to_send = latest_frame
                    detections_to_send = latest_detections
                    ts_to_send = latest_timestamp

            if frame_to_send is not None:
                # Encode frame to base64 and send; if sending fails, skip and continue.
                try:
                    frame_base64 = base64.b64encode(frame_to_send).decode('utf-8')
                    # Send as JSON (frame as base64) to keep compatibility with frontend
                    await websocket.send_json({
                        'frame': frame_base64,
                        'detections': detections_to_send,
                        'timestamp': ts_to_send
                    })
                except Exception as e:
                    # Log send errors and continue; a slow client shouldn't block capture.
                    print(f"[warning] websocket send error: {e}")

                # Throttle send rate to ~15-20 FPS to reduce CPU and network pressure
                await asyncio.sleep(0.05)
    
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        print(f"Client disconnected. Total connections: {len(active_connections)}")

# ============================================
# REST API ENDPOINTS
# ============================================

@app.get("/")
async def root():
    return {
        "service": "APVV - PPE Detection Service",
        "status": "running",
        "camera_active": camera_manager.is_running
    }

@app.post("/camera/start")
async def start_camera(
    payload: Optional[CameraStartRequest] = Body(None),
    camera_id: int = 0,
):
    """Start camera capture"""
    requested_camera = payload.camera_id if payload else camera_id

    try:
        if camera_manager.is_running and camera_manager.current_camera == requested_camera:
            return {"success": True, "message": f"Camera {requested_camera} already running"}

        camera_manager.start(requested_camera)
        return {"success": True, "message": f"Camera {requested_camera} started"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/camera/stop")
async def stop_camera():
    """Stop camera capture"""
    camera_manager.stop()
    return {"success": True, "message": "Camera stopped"}

@app.get("/camera/status")
async def camera_status():
    """Get camera status"""
    return {
        "running": camera_manager.is_running,
        "camera": camera_manager.current_camera
    }

@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    """Detect PPE in uploaded image"""
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detect PPE
        detections = detector.detect_ppe(frame)
        
        # Draw detections
        annotated_frame = detector.draw_detections(frame, detections)
        
        # Convert to JPEG
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        
        return StreamingResponse(
            BytesIO(buffer.tobytes()),
            media_type="image/jpeg"
        )
    
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/alerts/summary")
async def alerts_summary():
    """Get summary of recent alerts"""
    # In production, this would query a database
    return {
        "total_detections": 0,
        "violations": 0,
        "compliant": 0
    }

# ============================================
# MAIN - START SERVER
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 50)
    print("APVV - PPE Detection Service")
    print("=" * 50)
    print("\nStarting server...")
    print("Frontend should connect to: ws://localhost:8000/ws/video")
    print("\nEndpoints:")
    print("  - POST /camera/start - Start camera")
    print("  - POST /camera/stop  - Stop camera")
    print("  - WS   /ws/video     - Video stream")
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")