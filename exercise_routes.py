from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime

from ..database import get_db, ExerciseSession
from ..models import ExerciseInfo, StartExerciseRequest, SessionSummary
from ..config import EXERCISES
from ..auth import get_current_user

router = APIRouter(prefix="/api/exercises", tags=["exercises"])

@router.get("/", response_model=List[ExerciseInfo])
def get_exercises():
    """Get list of all available exercises"""
    exercises_list = []
    for ex_id, ex_data in EXERCISES.items():
        exercises_list.append({
            "id": ex_id,
            "name": ex_data["name"],
            "display_name": ex_data["display_name"],
            "reps": ex_data["reps"],
            "hold_time": ex_data["hold_time"],
            "side": ex_data.get("side", "both")
        })
    return exercises_list

@router.post("/start", response_model=SessionSummary)
def start_session(
    request: StartExerciseRequest, 
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Start a new exercise session"""
    ex_id = request.exercise_id
    if ex_id not in EXERCISES:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Exercise with ID {ex_id} not found"
        )
    
    ex_data = EXERCISES[ex_id]
    
    new_session = ExerciseSession(
        user_id=current_user.id,
        exercise_name=ex_data["display_name"],
        exercise_type=ex_data["name"],
        started_at=datetime.utcnow()
    )
    
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    
    return new_session

@router.get("/sessions", response_model=List[SessionSummary])
def get_user_sessions(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get exercise history for current user"""
    sessions = db.query(ExerciseSession).filter(
        ExerciseSession.user_id == current_user.id
    ).order_by(ExerciseSession.started_at.desc()).all()
    
    return sessions
