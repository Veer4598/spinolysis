"""
SPINOLYSIS - AI Posture Correction Backend
API and WebSocket server for Spinolysis React Frontend
"""
import os
import time
from fastapi import FastAPI, WebSocket, Depends, HTTPException, Request, Form, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import json

# Import local modules
from .config import *
from .database import init_db, get_db, ExerciseSession, RepDetail, User
from .auth import get_current_user, create_access_token, decode_access_token, authenticate_user
from .routes import auth_routes, exercise_routes
from .websocket_manager import ws_manager
from .d3_backend import PoseAnalyzer

# ==================== INITIALIZATION ====================

app = FastAPI(
    title="Spinolysis API",
    description="Backend for AI Posture Correction",
    version="1.2.0",
    docs_url="/api/docs"
)

# Initialize Pose Analyzer
pose_analyzer = PoseAnalyzer()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== API ROUTES ====================

app.include_router(auth_routes.router)
app.include_router(exercise_routes.router)

@app.get("/api/health")
async def health():
    return {"status": "healthy", "version": "1.2.0"}

@app.post("/api/analyze_pose")
async def analyze_pose_api(request: Request):
    """Analyze pose keypoints sent from frontend"""
    try:
        data = await request.json()
        keypoints_raw = data.get('keypoints') or []
        exercise = data.get('exercise', data.get('exercise_id', 'unknown'))
        
        # Maps frontend MediaPipe/COCO format to analyzer expectations if needed
        # For now we assume keypoints are already in format [x, y, conf]
        formatted_keypoints = []
        for kp in keypoints_raw:
             if isinstance(kp, dict):
                 x = kp.get('x', 0)
                 y = kp.get('y', 0)
                 conf = kp.get('score', kp.get('visibility', 0))
                 formatted_keypoints.append([x, y, conf])
             else:
                 formatted_keypoints.append(kp)
            
        analysis_result = pose_analyzer.analyze_pose(formatted_keypoints, exercise)
        return analysis_result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ==================== WEBSOCKET ====================

@app.websocket("/ws/pose-detection")
async def websocket_endpoint(websocket: WebSocket, token: str = "demo", db: Session = Depends(get_db)):
    try:
        user_id = 1
        if token != "demo":
            payload = decode_access_token(token)
            user_id = payload.get("user_id", 1)
        
        await ws_manager.handle_connection(websocket, user_id, db)
    except Exception as e:
        print(f"WS Error: {e}")
        try:
            await websocket.close(code=1008)
        except:
            pass

# ==================== STARTUP ====================

@app.on_event("startup")
async def startup():
    init_db()
    print("✓ Spinolysis API Backend Started on Port 8000")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
