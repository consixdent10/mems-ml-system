"""
SQLAlchemy Database Configuration for MEMS ML System
Provides persistent storage for sensor data, analysis results, and trained models.
"""

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

# Database URL - using SQLite for simplicity
DATABASE_URL = "sqlite:///./mems_ml_database.db"

# Create engine
engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False}  # Needed for SQLite
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


# ============== Database Models ==============

class SensorDataRecord(Base):
    """Stores individual sensor data sessions"""
    __tablename__ = "sensor_data_records"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(50), index=True)
    sensor_type = Column(String(50))
    data_json = Column(Text)  # JSON string of sensor data
    num_samples = Column(Integer)
    degradation_level = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Computed features
    mean_value = Column(Float, nullable=True)
    std_value = Column(Float, nullable=True)
    snr = Column(Float, nullable=True)
    rul = Column(Float, nullable=True)
    anomaly_count = Column(Integer, default=0)


class TrainedModelRecord(Base):
    """Stores trained ML model metadata"""
    __tablename__ = "trained_models"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100))
    model_type = Column(String(50))
    model_path = Column(String(255))  # Path to saved model file
    
    # Performance metrics
    accuracy = Column(Float)
    mse = Column(Float)
    r2_score = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    
    # Metadata
    training_samples = Column(Integer)
    training_time = Column(Float)  # seconds
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)


class AnalysisResult(Base):
    """Stores XAI and analysis results"""
    __tablename__ = "analysis_results"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(50), index=True)
    analysis_type = Column(String(50))  # 'xai', 'anomaly', 'prediction'
    results_json = Column(Text)  # JSON string of results
    created_at = Column(DateTime, default=datetime.utcnow)


class UploadedFile(Base):
    """Tracks uploaded data files"""
    __tablename__ = "uploaded_files"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255))
    original_filename = Column(String(255))
    file_size = Column(Integer)
    num_rows = Column(Integer)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed = Column(Boolean, default=False)


# ============== Database Functions ==============

def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)
    print("[DB] Database tables created successfully")


def get_db():
    """Get database session - use as dependency in FastAPI"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_database():
    """Initialize the database"""
    create_tables()
    return True


# ============== CRUD Operations ==============

class DatabaseOperations:
    """Database CRUD operations"""
    
    @staticmethod
    def save_sensor_data(db, session_id: str, sensor_type: str, data: list, 
                         features: dict, rul: float, anomaly_count: int, degradation_level: int = 0):
        """Save sensor data session to database"""
        import json
        
        record = SensorDataRecord(
            session_id=session_id,
            sensor_type=sensor_type,
            data_json=json.dumps(data),
            num_samples=len(data),
            degradation_level=degradation_level,
            mean_value=float(features.get('mean', 0)),
            std_value=float(features.get('std', 0)),
            snr=float(features.get('snr', 0)),
            rul=rul,
            anomaly_count=anomaly_count
        )
        
        db.add(record)
        db.commit()
        db.refresh(record)
        return record.id
    
    @staticmethod
    def save_trained_model(db, model_name: str, model_type: str, model_path: str, metrics: dict):
        """Save trained model metadata to database"""
        record = TrainedModelRecord(
            model_name=model_name,
            model_type=model_type,
            model_path=model_path,
            accuracy=float(metrics.get('accuracy', 0)),
            mse=float(metrics.get('mse', 0)),
            r2_score=float(metrics.get('r2Score', 0)),
            precision=float(metrics.get('precision', 0)),
            recall=float(metrics.get('recall', 0)),
            f1_score=float(metrics.get('f1Score', 0)),
            training_samples=int(metrics.get('trainingSize', 0)),
            training_time=float(metrics.get('trainingTime', 0))
        )
        
        db.add(record)
        db.commit()
        db.refresh(record)
        return record.id
    
    @staticmethod
    def get_all_sensor_records(db, limit: int = 100):
        """Get all sensor data records"""
        return db.query(SensorDataRecord).order_by(
            SensorDataRecord.created_at.desc()
        ).limit(limit).all()
    
    @staticmethod
    def get_all_trained_models(db, active_only: bool = True):
        """Get all trained models"""
        query = db.query(TrainedModelRecord)
        if active_only:
            query = query.filter(TrainedModelRecord.is_active == True)
        return query.order_by(TrainedModelRecord.created_at.desc()).all()
    
    @staticmethod
    def get_sensor_data_by_session(db, session_id: str):
        """Get sensor data by session ID"""
        return db.query(SensorDataRecord).filter(
            SensorDataRecord.session_id == session_id
        ).first()
    
    @staticmethod
    def delete_model(db, model_id: int):
        """Soft delete a trained model"""
        model = db.query(TrainedModelRecord).filter(TrainedModelRecord.id == model_id).first()
        if model:
            model.is_active = False
            db.commit()
            return True
        return False
