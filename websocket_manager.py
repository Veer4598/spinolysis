"""
WebSocket manager for real-time pose detection
Integrates d3.py PoseAnalyzer for live camera feed processing
"""
import cv2
import numpy as np
import base64
import json
import time
from typing import Dict, Optional
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from .d3_backend import PoseAnalyzer
from .database import ExerciseSession, RepDetail, User


class PoseDetectionSession:
    """Manages a single pose detection session"""
    
    def __init__(self, user_id: int, exercise_id: int, db: Session):
        self.user_id = user_id
        self.exercise_id = exercise_id
        self.db = db
        
        # Initialize pose analyzer
        self.analyzer = PoseAnalyzer()
        
        # Session state
        self.session_start_time = time.time()
        self.rep_count = 0
        self.correct_reps = 0
        self.incorrect_reps = 0
        self.partial_reps = 0
        
        # Rep tracking
        self.current_rep_start = None
        self.rep_being_counted = False
        self.rep_hold_start_time = 0
        self.current_side = "right"
        
        # Accuracy tracking
        self.frame_count = 0
        self.correct_frames = 0
        self.partial_frames = 0
        self.incorrect_frames = 0
        self.last_frame_time = time.time()
        self.fps = 0.0
        
        # Database session record
        self.db_session = ExerciseSession(
            user_id=user_id,
            exercise_name=self.analyzer.exercises.get(exercise_id, {}).get("display_name", "Unknown"),
            exercise_type=self.analyzer.exercises.get(exercise_id, {}).get("name", "UNKNOWN"),
            started_at=datetime.utcnow()
        )
        self.db.add(self.db_session)
        self.db.commit()
        self.db.refresh(self.db_session)
        
        print(f"✓ Created session {self.db_session.id} for user {user_id}, exercise {exercise_id}")
    
    def process_frame(self, frame_data: str) -> dict:
        """
        Process a single frame and return analysis results
        
        Args:
            frame_data: Base64 encoded image string
            
        Returns:
            dict with processed frame, metrics, and feedback
        """
        try:
            # Decode base64 image
            img_bytes = base64.b64decode(frame_data.split(',')[1] if ',' in frame_data else frame_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return {"error": "Failed to decode frame"}
            
            # Flip frame horizontally for mirror view (Requirement 2)
            frame = cv2.flip(frame, 1)
            
            # Detect pose using d3.py logic
            keypoints = self.detect_pose(frame)
            
            # Analyze pose
            analysis = self.analyzer.analyze_pose(keypoints, self.exercise_id)
            
            # Update frame counter
            self.frame_count += 1
            
            # Update accuracy tracking
            if analysis["pose_status"] == "CORRECT":
                self.correct_frames += 1
            elif analysis["pose_status"] == "PARTIAL":
                self.partial_frames += 1
            elif analysis["pose_status"] == "INCORRECT":
                self.incorrect_frames += 1
            
            # Check for rep completion
            rep_completed = self.check_rep_completion(analysis)

            # Calculate elapsed/FPS for d3-like camera overlay
            elapsed_time = int(time.time() - self.session_start_time)
            now = time.time()
            dt = now - self.last_frame_time
            if dt > 0:
                inst_fps = 1.0 / dt
                self.fps = inst_fps if self.fps == 0 else (0.8 * self.fps + 0.2 * inst_fps)
            self.last_frame_time = now

            # Draw skeleton and keypoints on frame
            annotated_frame = self.draw_pose(frame, keypoints, analysis, elapsed_time)
            
            # Encode processed frame
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            processed_frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # Calculate current accuracy
            total_frames = self.correct_frames + self.partial_frames + self.incorrect_frames
            current_accuracy = (self.correct_frames / total_frames * 100) if total_frames > 0 else 0
            
            return {
                "type": "frame_processed",
                "frame": f"data:image/jpeg;base64,{processed_frame_b64}",
                "metrics": {
                    "exercise_name": self.analyzer.exercises[self.exercise_id]["display_name"],
                    "rep_count": self.rep_count,
                    "pose_status": analysis["pose_status"],
                    "accuracy": round(current_accuracy, 1),
                    "feedback": analysis["feedback"],
                    "elapsed_time": elapsed_time,
                    "keypoints_detected": analysis.get("keypoints_detected", 0),
                    "rep_completed": rep_completed,
                    "tilt_angle": analysis.get("tilt_angle", 0.0),
                    "arm_angle": analysis.get("arm_angle", 0.0),
                    "current_side": self.current_side,
                    "fps": int(self.fps),
                },
                "analysis": analysis
            }
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return {"error": str(e)}
    
    def detect_pose(self, frame):
        """Detect pose keypoints from frame using YOLO on real user feed only."""
        try:
            import os
            # PyTorch>=2.6 changed torch.load default to weights_only=True.
            # Force compatibility for trusted local YOLO checkpoints.
            os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
            from ultralytics import YOLO
            
            # Load YOLO model if not loaded
            if not hasattr(self, '_yolo_load_attempted'):
                self._yolo_load_attempted = False
            if not self._yolo_load_attempted:
                self._yolo_load_attempted = True
                from .config import MODEL_PATH
                try:
                    if os.path.exists(MODEL_PATH):
                        self.yolo_model = YOLO(MODEL_PATH)
                    else:
                        print(f"⚠ Warning: Model not found at {MODEL_PATH}")
                        # Fallback to available local pose models.
                        fallback_paths = ["yolov8m-pose.pt", "yolov8n-pose.pt"]
                        loaded = None
                        for p in fallback_paths:
                            if os.path.exists(p):
                                loaded = p
                                break
                        self.yolo_model = YOLO(loaded) if loaded else None
                except Exception as model_error:
                    self.yolo_model = None
                    print(f"YOLO model load error: {model_error}")
            
            # Run detection with progressive thresholds for real-world webcam lighting.
            if self.yolo_model is not None:
                for conf, iou in ((0.25, 0.5), (0.15, 0.45), (0.08, 0.35)):
                    results = self.yolo_model(frame, verbose=False, conf=conf, iou=iou, imgsz=640)
                    if results and len(results) > 0 and results[0].keypoints is not None:
                        keypoints_data = results[0].keypoints.data.cpu().numpy()
                        if len(keypoints_data) > 0:
                            return keypoints_data[0]

                # Last attempt with upscaled frame for small/far subjects.
                upscaled = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
                results = self.yolo_model(upscaled, verbose=False, conf=0.08, iou=0.35, imgsz=960)
                if results and len(results) > 0 and results[0].keypoints is not None:
                    keypoints_data = results[0].keypoints.data.cpu().numpy()
                    if len(keypoints_data) > 0:
                        kps = keypoints_data[0].copy()
                        kps[:, 0] /= 1.5
                        kps[:, 1] /= 1.5
                        return kps
            return None
            
        except Exception as e:
            print(f"YOLO detection error: {e}")
            return None
    
    def simulate_keypoints(self, frame):
        """Generate simulated keypoints for testing without YOLO"""
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        simulated_keypoints = np.zeros((17, 3))
        positions = {
            0: (center_x, center_y - 100, 0.9),  # Nose
            5: (center_x - 50, center_y - 50, 0.8),  # Left shoulder
            6: (center_x + 50, center_y - 50, 0.8),  # Right shoulder
            7: (center_x - 50, center_y, 0.7),  # Left elbow
            8: (center_x + 50, center_y, 0.7),  # Right elbow
            9: (center_x - 50, center_y + 50, 0.6),  # Left wrist
            10: (center_x + 50, center_y + 50, 0.6),  # Right wrist
            11: (center_x - 30, center_y + 50, 0.8),  # Left hip
            12: (center_x + 30, center_y + 50, 0.8),  # Right hip
            13: (center_x - 30, center_y + 150, 0.7),  # Left knee
            14: (center_x + 30, center_y + 150, 0.7),  # Right knee
            15: (center_x - 30, center_y + 250, 0.6),  # Left ankle
            16: (center_x + 30, center_y + 250, 0.6),  # Right ankle
        }
        
        for idx, (x, y, conf) in positions.items():
            simulated_keypoints[idx] = [x, y, conf]
        
        return simulated_keypoints
    
    def draw_pose(self, frame, keypoints, analysis, elapsed_time: int):
        """Draw d3.py-style skeleton, metrics and status overlays."""
        height, width = frame.shape[:2]
        annotated = frame.copy()
        if keypoints is None:
            keypoints = []
        
        # Skeleton connections (YOLO COCO format)
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Face
            (5, 6),  # Shoulders
            (5, 7), (7, 9),  # Left arm
            (6, 8), (8, 10),  # Right arm
            (5, 11), (6, 12),  # Torso
            (11, 12),  # Hips
            (11, 13), (13, 15),  # Left leg
            (12, 14), (14, 16)   # Right leg
        ]
        
        # Colors (d3.py style)
        colors = {
            'blue': (255, 0, 0),
            'green': (0, 255, 0),
            'red': (0, 0, 255),
            'yellow': (0, 255, 255),
            'orange': (0, 165, 255),
            'cyan': (255, 255, 0),
            'white': (255, 255, 255),
            'gray': (128, 128, 128),
            'pink': (203, 192, 255),
            'purple': (255, 0, 255),
        }
        
        # Color based on pose status
        color_map = {
            "CORRECT": colors['green'],
            "PARTIAL": colors['orange'],
            "INCORRECT": colors['red'],
            "WAITING": colors['gray']
        }
        status_color = color_map.get(analysis.get("pose_status", "WAITING"), colors['white'])
        
        # 1. Draw skeleton
        for start_idx, end_idx in skeleton:
            if (
                start_idx < len(keypoints)
                and end_idx < len(keypoints)
                and keypoints[start_idx][2] > 0.1
                and keypoints[end_idx][2] > 0.1
            ):
                pt1 = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                pt2 = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                cv2.line(annotated, pt1, pt2, colors['blue'], 2)
        
        # 2. Draw keypoints
        for i, kp in enumerate(keypoints):
            if kp[2] > 0.1:
                x, y = int(kp[0]), int(kp[1])
                if i == 0: color, radius = colors['red'], 6
                elif i in [5, 6, 11, 12]: color, radius = colors['green'], 5
                elif i in [7, 8, 9, 10]: color, radius = colors['yellow'], 4
                elif i in [13, 14, 15, 16]: color, radius = colors['orange'], 4
                else: color, radius = colors['cyan'], 3
                
                cv2.circle(annotated, (x, y), radius, color, -1)
                cv2.circle(annotated, (x, y), radius + 2, colors['white'], 1)

        # 3. Draw derived neck and mid-hip markers
        neck_virtual = self.analyzer.derive_virtual_neck(keypoints)
        midhip_virtual = self.analyzer.derive_virtual_midhip(keypoints)
        if neck_virtual:
            nx, ny = int(neck_virtual[0]), int(neck_virtual[1])
            cv2.circle(annotated, (nx, ny), 5, colors['purple'], -1)
            cv2.circle(annotated, (nx, ny), 7, colors['white'], 1)
            cv2.putText(annotated, "Neck", (nx + 8, ny), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors['purple'], 1)
        if midhip_virtual:
            mx, my = int(midhip_virtual[0]), int(midhip_virtual[1])
            cv2.circle(annotated, (mx, my), 5, colors['pink'], -1)
            cv2.circle(annotated, (mx, my), 7, colors['white'], 1)
            cv2.putText(annotated, "MidHip", (mx + 8, my), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors['pink'], 1)

        # 4. Semi-transparent top bar
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
        
        # Title and Status
        exercise_name = self.analyzer.exercises.get(self.exercise_id, {}).get("display_name", "Exercise")
        cv2.putText(annotated, f"YOLOv8 Pose - {exercise_name}", (10, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, colors['cyan'], 2)
        
        status_text = analysis.get("pose_status", "WAITING")
        cv2.putText(annotated, f"Status: {status_text}", (width - 220, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, status_color, 2)
        
        # Accuracy and Metrics
        total_frames = self.correct_frames + self.partial_frames + self.incorrect_frames
        accuracy = (self.correct_frames / total_frames * 100) if total_frames > 0 else 0
        
        metrics_y = 90
        exercise_type = self.analyzer.exercises.get(self.exercise_id, {}).get("name", "")
        keypoints_detected = analysis.get("keypoints_detected", 0)
        if exercise_type == "NECK_SIDE_TILT":
            rep_target = max(1, int(self.analyzer.exercises.get(self.exercise_id, {}).get("reps", 0) * 2))
            cv2.putText(annotated, f"Tilt: {analysis.get('tilt_angle', 0.0):.1f} deg", (10, metrics_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['yellow'], 2)
            cv2.putText(annotated, f"Reps: {self.rep_count}/{rep_target}", (10, metrics_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['pink'], 2)
            cv2.putText(annotated, f"Side: {self.current_side}", (10, metrics_y + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['orange'], 2)
            if self.rep_being_counted:
                hold_time = time.time() - self.rep_hold_start_time
                cv2.putText(annotated, f"Hold: {hold_time:.1f}/2.0s", (10, metrics_y + 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['green'], 2)
        elif exercise_type in {"ARM_RAISE_OVERHEAD", "ARM_ROTATION"}:
            cv2.putText(annotated, f"Arm Angle: {analysis.get('arm_angle', 0.0):.1f} deg", (10, metrics_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['yellow'], 2)
            cv2.putText(annotated, f"Keypoints: {keypoints_detected}/17", (10, metrics_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['pink'], 2)
        else:
            cv2.putText(annotated, f"Keypoints: {keypoints_detected}/17", (10, metrics_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['yellow'], 2)

        perf_y = height - 110
        cv2.putText(annotated, f"FPS: {int(self.fps)}", (width - 170, perf_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['cyan'], 1)
        cv2.putText(annotated, "Detector: YOLOv8", (width - 170, perf_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['green'], 1)
        acc_color = colors['green'] if accuracy > 70 else colors['orange'] if accuracy > 40 else colors['red']
        cv2.putText(annotated, f"Accuracy: {accuracy:.1f}%", (width - 170, perf_y + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, acc_color, 1)
        cv2.putText(annotated, f"Time: {elapsed_time}s", (width - 170, perf_y + 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['white'], 1)

        bottom_overlay = annotated.copy()
        cv2.rectangle(bottom_overlay, (0, height - 40), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(bottom_overlay, 0.45, annotated, 0.55, 0, annotated)
        feedback = analysis.get("feedback", "Adjust pose")
        cv2.putText(annotated, f"Feedback: {feedback}", (10, height - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, status_color, 2)

        return annotated
    
    def check_rep_completion(self, analysis) -> bool:
        """Check if a rep should be counted based on pose analysis"""
        current_exercise = self.analyzer.exercises[self.exercise_id]
        current_time = time.time()
        
        # For rep-based exercises
        if current_exercise["reps"] > 0:
            if analysis["pose_status"] == "CORRECT":
                if not self.rep_being_counted:
                    self.rep_hold_start_time = current_time
                    self.rep_being_counted = True
                    self.current_rep_start = current_time
                
                # Check if held for 2 seconds
                if current_time - self.rep_hold_start_time >= 2:
                    self.current_side = "left" if self.current_side == "right" else "right"
                    self.complete_rep(analysis, current_time)
                    return True
            else:
                self.rep_being_counted = False
        
        return False
    
    def complete_rep(self, analysis, current_time):
        """Mark a rep as complete and save to database"""
        self.rep_count += 1
        
        # Categorize rep quality
        if analysis["pose_status"] == "CORRECT":
            self.correct_reps += 1
            quality = "CORRECT"
        elif analysis["pose_status"] == "PARTIAL":
            self.partial_reps += 1
            quality = "PARTIAL"
        else:
            self.incorrect_reps += 1
            quality = "INCORRECT"
        
        # Calculate rep duration
        duration = current_time - self.current_rep_start if self.current_rep_start else 0
        
        # Save rep to database
        rep_detail = RepDetail(
            session_id=self.db_session.id,
            rep_number=self.rep_count,
            accuracy_score=analysis.get("score", 0.0),
            pose_quality=quality,
            duration_seconds=duration,
            timestamp=datetime.utcnow()
        )
        self.db.add(rep_detail)
        self.db.commit()
        
        # Reset rep tracking
        self.rep_being_counted = False
        self.current_rep_start = None
    
    def finalize_session(self):
        """Finalize session and save summary to database"""
        duration = int(time.time() - self.session_start_time)
        total_frames = self.correct_frames + self.partial_frames + self.incorrect_frames
        avg_accuracy = (self.correct_frames / total_frames * 100) if total_frames > 0 else 0
        
        # Update database session
        self.db_session.total_reps = self.rep_count
        self.db_session.correct_reps = self.correct_reps
        self.db_session.incorrect_reps = self.incorrect_reps
        self.db_session.average_accuracy = round(avg_accuracy, 2)
        self.db_session.duration_seconds = duration
        self.db_session.completed_at = datetime.utcnow()
        
        # Create summary
        summary = {
            "total_frames": total_frames,
            "correct_frames": self.correct_frames,
            "partial_frames": self.partial_frames,
            "incorrect_frames": self.incorrect_frames,
            "duration": duration
        }
        self.db_session.summary = json.dumps(summary)
        
        self.db.commit()
        
        return {
            "session_id": self.db_session.id,
            "exercise": self.analyzer.exercises[self.exercise_id]["display_name"],
            "total_reps": self.rep_count,
            "correct_reps": self.correct_reps,
            "accuracy": round(avg_accuracy, 2),
            "duration": duration
        }


class WebSocketManager:
    """Manages WebSocket connections and pose detection sessions"""
    
    def __init__(self):
        self.active_sessions: Dict[WebSocket, PoseDetectionSession] = {}
    
    async def handle_connection(self, websocket: WebSocket, user_id: int, db: Session):
        """Handle WebSocket connection for pose detection"""
        await websocket.accept()
        print(f"✓ WebSocket connected for user {user_id}")
        
        session: Optional[PoseDetectionSession] = None
        
        try:
            while True:
                # Receive message from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                msg_type = message.get("type")
                
                if msg_type == "start_session":
                    # Start new detection session
                    exercise_id = message.get("exercise_id", 1)
                    session = PoseDetectionSession(user_id, exercise_id, db)
                    self.active_sessions[websocket] = session
                    
                    await websocket.send_json({
                        "type": "session_started",
                        "session_id": session.db_session.id,
                        "exercise": session.analyzer.exercises[exercise_id]["display_name"]
                    })
                
                elif msg_type == "frame":
                    # Process frame
                    if session is None:
                        await websocket.send_json({"error": "No active session"})
                        continue
                    
                    frame_data = message.get("data")
                    result = session.process_frame(frame_data)
                    
                    await websocket.send_json(result)
                
                elif msg_type == "stop_session":
                    # Stop session
                    if session:
                        summary = session.finalize_session()
                        await websocket.send_json({
                            "type": "session_completed",
                            "summary": summary
                        })
                        session = None
                
        except WebSocketDisconnect:
            print(f"✓ WebSocket disconnected for user {user_id}")
            if session:
                session.finalize_session()
        
        except Exception as e:
            print(f"WebSocket error: {e}")
            await websocket.send_json({"error": str(e)})
        
        finally:
            if websocket in self.active_sessions:
                del self.active_sessions[websocket]


# Global WebSocket manager instance
ws_manager = WebSocketManager()
