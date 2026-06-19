"""
Database models and configuration for Spinolysis
SQLAlchemy ORM with SQLite backend
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

# Database configuration
DATABASE_URL = "sqlite:///./spinolysis.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# ==================== MODELS ====================

class User(Base):
    """User model for authentication"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    sessions = relationship("ExerciseSession", back_populates="user")


class ExerciseSession(Base):
    """Exercise session model - stores workout sessions"""
    __tablename__ = "exercise_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    exercise_name = Column(String, nullable=False)
    exercise_type = Column(String)  # e.g., "NECK_SIDE_TILT", "ARM_RAISE_OVERHEAD"
    
    # Session metrics
    total_reps = Column(Integer, default=0)
    correct_reps = Column(Integer, default=0)
    incorrect_reps = Column(Integer, default=0)
    average_accuracy = Column(Float, default=0.0)
    duration_seconds = Column(Integer, default=0)
    
    # Timestamps
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    
    # Additional data (JSON-like storage)
    summary = Column(Text)  # JSON string with detailed metrics
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    rep_details = relationship("RepDetail", back_populates="session")


class RepDetail(Base):
    """Individual rep tracking for detailed analysis"""
    __tablename__ = "rep_details"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("exercise_sessions.id"))
    
    rep_number = Column(Integer)
    accuracy_score = Column(Float)  # 0.0 to 1.0
    pose_quality = Column(String)  # "CORRECT", "PARTIAL", "INCORRECT"
    duration_seconds = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    session = relationship("ExerciseSession", back_populates="rep_details")


# ==================== DATABASE UTILITIES ====================

def init_db():
    """Initialize database - create all tables"""
    Base.metadata.create_all(bind=engine)
    print("✓ Database initialized successfully")


def get_db():
    """Dependency for FastAPI routes to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def reset_db():
    """Reset database - drop and recreate all tables (use with caution!)"""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    print("⚠ Database reset completed")


if __name__ == "__main__":
    # Initialize database when run directly
    init_db()
    print("Database setup complete!")
