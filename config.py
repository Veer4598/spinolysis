#!/usr/bin/env python3
"""
Spinolysis Application Configuration
This file contains environment and startup configurations
"""

import os
import sys

# ===== APPLICATION SETTINGS =====
DEBUG = True
HOST = '0.0.0.0'
PORT = 8000

# ===== PATHS =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'yolov8m-pose.pt')
VIDEOS_DIR = os.path.join(BASE_DIR, 'videos')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')

# ===== CORS SETTINGS =====
CORS_ORIGINS = "*"
CORS_METHODS = ['GET', 'POST', 'OPTIONS', 'PUT', 'DELETE']

# ===== AI/ML SETTINGS =====
POSE_DETECTION = {
    'confidence_threshold': 0.3,
    'camera_width': 640,
    'camera_height': 480,
    'model_type': 'YOLOv8'
}

# ===== SESSION SETTINGS =====
SESSION_TIMEOUT = 3600  # 1 hour in seconds
ENABLE_SESSION_PERSISTENCE = False  # Set to True with database

# ===== EXERCISE SETTINGS =====
EXERCISES = {
    1: {
        'name': 'NECK_SIDE_TILT',
        'display_name': 'Neck Side Tilt',
        'reps': 3,
        'hold_time': 0,
        'category': 'neck'
    },
    2: {
        'name': 'ARM_RAISE_OVERHEAD',
        'display_name': 'Hand Raise',
        'reps': 0,
        'hold_time': 10,
        'category': 'arms'
    },
    3: {
        'name': 'ARM_ROTATION',
        'display_name': 'Arm Rotation',
        'reps': 0,
        'hold_time': 10,
        'category': 'arms'
    },
    4: {
        'name': 'CAT_POSE',
        'display_name': 'Cat Pose',
        'reps': 0,
        'hold_time': 15,
        'category': 'spine'
    },
    5: {
        'name': 'CHILD_POSE',
        'display_name': "Child's Pose",
        'reps': 0,
        'hold_time': 15,
        'category': 'spine'
    },
    6: {
        'name': 'BRIDGE_POSE',
        'display_name': 'Bridge Pose',
        'reps': 0,
        'hold_time': 15,
        'category': 'spine'
    },
    7: {
        'name': 'MOUNTAIN_POSE',
        'display_name': 'Mountain Pose',
        'reps': 0,
        'hold_time': 15,
        'category': 'spine'
    }
}

# ===== LOGGING SETTINGS =====
LOG_LEVEL = 'INFO'
LOG_FILE = 'spinolysis.log'

def print_startup_info():
    """Print startup information"""
    print("\n" + "="*60)
    print("  SPINOLYSIS - AI POSTURE CORRECTION API")
    print("="*60)
    print(f"\n📍 API starting on: http://{HOST}:{PORT}")
    print(f"🔧 Debug mode: {'ON' if DEBUG else 'OFF'}")
    print(f"📁 Model: {MODEL_PATH}")
    print(f"\n🏋️  Available Exercises: {len(EXERCISES)}")
    for ex_id, exercise in EXERCISES.items():
        ex_type = f"{exercise['reps']} reps" if exercise['reps'] > 0 else f"{exercise['hold_time']}s hold"
        print(f"   {ex_id}. {exercise['display_name']} ({ex_type})")
    print("\n🌐 API is ready for requests!")
    print("="*60 + "\n")

if __name__ == '__main__':
    print_startup_info()
