from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy.orm import Session
import numpy as np
import pandas as pd
from datetime import datetime
import json
import sys
import os
import uuid
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Add models directory to path
sys.path.append(os.path.dirname(__file__))

from models.ml_models import MLModelTrainer
from models.data_processor import DataProcessor
from models.xai_analyzer import XAIAnalyzer
from database.database import init_database, get_db, DatabaseOperations
from data.dataset_loader import dataset_loader, RealDatasetLoader

# API Tags for Swagger documentation
tags_metadata = [
    {
        "name": "Data Generation",
        "description": "Generate synthetic MEMS sensor data with configurable parameters",
    },
    {
        "name": "Data Upload",
        "description": "Upload and process custom CSV sensor data files",
    },
    {
        "name": "ML Models",
        "description": "Train and manage machine learning models",
    },
    {
        "name": "XAI Analysis",
        "description": "Explainable AI analysis and feature importance",
    },
    {
        "name": "Predictions",
        "description": "Make predictions using trained models",
    },
    {
        "name": "Sensor Characteristics",
        "description": "Analyze sensor parameters: sensitivity, resonant frequency, noise",
    },
    {
        "name": "Real Datasets",
        "description": "Load real-world datasets: NASA IMS Bearing, CWRU, MEMS Vibration",
    },
    {
        "name": "History",
        "description": "View historical sensor data and analysis records",
    },
    {
        "name": "System",
        "description": "Health checks and system information",
    },
    {
        "name": "Email Alerts",
        "description": "Send email notifications for sensor alerts",
    },
    {
        "name": "Deep Learning",
        "description": "LSTM neural network for time-series RUL prediction",
    },
]

app = FastAPI(
    title="MEMS Sensor ML Analysis API",
    description="""
## Advanced Machine Learning-Based Performance Analysis and Predictive Maintenance

This API provides endpoints for:
- **Synthetic Data Generation**: Generate realistic MEMS sensor data
- **Data Upload**: Process uploaded CSV files with sensor readings
- **ML Model Training**: Train Random Forest, KNN, SVM, and other models
- **XAI Analysis**: Explainable AI with SHAP values and feature importance
- **Predictions**: Real-time RUL (Remaining Useful Life) predictions
- **Anomaly Detection**: Identify abnormal sensor readings

### Key Features:
- Support for accelerometer, gyroscope, pressure, and temperature sensors
- Real-time performance metrics (SNR, drift, noise analysis)
- Multiple ML algorithms for comparison
- SHAP-based explainability
    """,
    version="2.0.0",
    openapi_tags=tags_metadata,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processors
ml_trainer = MLModelTrainer()
data_processor = DataProcessor()
xai_analyzer = XAIAnalyzer()

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database tables on application startup"""
    print("[DB] Initializing database...")
    init_database()
    print("[DB] Database ready!")

# Pydantic models
class SensorDataPoint(BaseModel):
    time: float
    value: float
    temperature: float
    humidity: Optional[float] = 50.0
    
class GenerateDataRequest(BaseModel):
    sensor_type: str
    num_samples: int = 1000
    degradation_level: int = 0

class TrainModelsRequest(BaseModel):
    sensor_data: List[dict]
    
class PredictionRequest(BaseModel):
    features: dict

class EmailAlertRequest(BaseModel):
    to_email: str
    alert_type: str
    sensor_type: str
    rul: str
    status: str
    timestamp: str

# Email configuration - Gmail SMTP with App Password
# Set GMAIL_EMAIL and GMAIL_APP_PASSWORD environment variables to enable email alerts
EMAIL_CONFIG = {
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "sender_email": os.environ.get("GMAIL_EMAIL", ""),
    "sender_password": os.environ.get("GMAIL_APP_PASSWORD", "")
}


@app.get("/", tags=["System"])
async def root():
    return {
        "message": "MEMS Sensor ML Analysis API",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "generate_data": "/api/generate-data",
            "upload_data": "/api/upload-data",
            "train_models": "/api/train-models",
            "xai_analysis": "/api/xai-analysis"
        }
    }


@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "service": "MEMS ML API"
    }


@app.post("/api/generate-data", tags=["Data Generation"])
async def generate_sensor_data(request: GenerateDataRequest):
    """
    Generate synthetic MEMS sensor data.
    
    Supports the following sensor types:
    - **accelerometer**: Acceleration measurements (m/s²)
    - **gyroscope**: Angular velocity (deg/s)
    - **pressure**: Atmospheric pressure (kPa)
    - **temperature**: Temperature readings (°C)
    
    Parameters:
    - **sensor_type**: Type of MEMS sensor to simulate
    - **num_samples**: Number of data points (default: 1000)
    - **degradation_level**: Simulated wear level 0-10 (0=new, 10=degraded)
    """
    try:
        print(f"Generating data for {request.sensor_type} with degradation {request.degradation_level}")
        
        data = data_processor.generate_mems_data(
            sensor_type=request.sensor_type,
            num_samples=request.num_samples,
            degradation_level=request.degradation_level
        )
        
        # Calculate features
        features = data_processor.extract_features(data)
        
        # Detect anomalies
        anomalies = data_processor.detect_anomalies(data)
        
        # Calculate RUL
        rul = data_processor.calculate_rul(data, request.degradation_level)
        
        # Extract sensor characteristics (sensitivity, resonant frequency, noise)
        sensor_characteristics = data_processor.extract_sensor_characteristics(
            data, request.sensor_type
        )
        
        print(f"Successfully generated {len(data)} data points")
        
        return {
            "data": data.to_dict('records'),
            "features": features,
            "anomalies": anomalies,
            "rul": float(rul),
            "sensor_characteristics": sensor_characteristics,
            "metadata": {
                "sensor_type": request.sensor_type,
                "num_samples": len(data),
                "generated_at": datetime.now().isoformat()
            }
        }
    except Exception as e:
        print(f"Error generating data: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload-data", tags=["Data Upload"])
async def upload_sensor_data(file: UploadFile = File(...)):
    """
    Upload and process CSV sensor data.
    
    Accepts CSV files with sensor readings. Required columns: time, value
    Optional columns: temperature, humidity, drift, noise
    """
    try:
        print(f"Uploading file: {file.filename}")
        
        # Read CSV
        contents = await file.read()
        data = pd.read_csv(pd.io.common.BytesIO(contents))
        
        print(f"File read successfully, {len(data)} rows")
        
        # Validate and process
        processed_data = data_processor.process_uploaded_data(data)
        
        # Calculate features
        features = data_processor.extract_features(processed_data)
        
        # Detect anomalies
        anomalies = data_processor.detect_anomalies(processed_data)
        
        # Calculate RUL
        rul = data_processor.calculate_rul(processed_data)
        
        print(f"Data processed successfully")
        
        return {
            "data": processed_data.to_dict('records'),
            "features": features,
            "anomalies": anomalies,
            "rul": float(rul),
            "metadata": {
                "filename": file.filename,
                "num_samples": len(processed_data),
                "uploaded_at": datetime.now().isoformat()
            }
        }
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")


@app.post("/api/train-models", tags=["ML Models"])
async def train_models(request: TrainModelsRequest):
    """
    Train ML models on sensor data.
    
    Trains multiple models: Random Forest, KNN, SVM, Linear Regression.
    Returns accuracy, MSE, R² score, precision, recall, and F1 score.
    """
    try:
        print(f"Training models on {len(request.sensor_data)} data points")
        
        # Convert to DataFrame
        data = pd.DataFrame(request.sensor_data)
        
        # Train all models
        results = ml_trainer.train_all_models(data)
        
        print(f"Successfully trained {len(results)} models")
        
        return {
            "models": results,
            "metadata": {
                "trained_at": datetime.now().isoformat(),
                "num_models": len(results),
                "training_samples": len(data)
            }
        }
    except Exception as e:
        print(f"Error training models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/xai-analysis", tags=["XAI Analysis"])
async def generate_xai_analysis(request: TrainModelsRequest):
    """
    Generate Explainable AI analysis.
    
    Returns SHAP values, feature importance, and prediction explanations.
    """
    try:
        print(f"Generating XAI analysis for {len(request.sensor_data)} data points")
        
        data = pd.DataFrame(request.sensor_data)
        
        # Generate XAI insights
        xai_results = xai_analyzer.analyze(data)
        
        print("XAI analysis completed successfully")
        
        return {
            "feature_importance": xai_results['feature_importance'],
            "shap_values": xai_results['shap_values'],
            "prediction_explanation": xai_results['explanation'],
            "confidence": xai_results['confidence'],
            "metadata": {
                "analyzed_at": datetime.now().isoformat()
            }
        }
    except Exception as e:
        print(f"Error in XAI analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict", tags=["Predictions"])
async def make_prediction(request: PredictionRequest):
    """Make RUL prediction using trained models."""
    try:
        prediction = ml_trainer.predict(request.features)
        
        return {
            "prediction": prediction['value'],
            "confidence": prediction['confidence'],
            "model_used": prediction['model'],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== Model Persistence Endpoints ==============

@app.post("/api/models/save", tags=["ML Models"])
async def save_trained_models(session_id: Optional[str] = None):
    """
    Save all trained models to disk.
    
    Models are saved using joblib in the saved_models directory.
    Returns the session ID that can be used to load the models later.
    """
    try:
        result = ml_trainer.save_models(session_id)
        return {
            "status": "success",
            "session_id": result["session_id"],
            "models_saved": [m["name"] for m in result["models"]],
            "directory": result["directory"],
            "saved_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/models/load/{session_id}", tags=["ML Models"])
async def load_saved_models(session_id: str):
    """
    Load previously saved models from a session.
    
    Use the session_id from a previous save operation.
    """
    try:
        result = ml_trainer.load_model(session_id)
        return {
            "status": "success",
            "session_id": result["session_id"],
            "models_loaded": result["models_loaded"],
            "loaded_at": datetime.now().isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models/saved", tags=["ML Models"])
async def list_saved_model_sessions():
    """
    List all saved model sessions.
    """
    try:
        from models.ml_models import MLModelTrainer
        sessions = MLModelTrainer.list_saved_sessions()
        return {
            "sessions": sessions,
            "total": len(sessions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/models/saved/{session_id}", tags=["ML Models"])
async def delete_saved_model_session(session_id: str):
    """
    Delete a saved model session.
    """
    try:
        from models.ml_models import MLModelTrainer
        result = MLModelTrainer.delete_session(session_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models/info", tags=["ML Models"])
async def get_models_info():
    """Get information about available models"""
    return {
        "available_models": [
            "Random Forest",
            "Gradient Boosting",
            "Support Vector Machine",
            "Neural Network"
        ],
        "algorithms": {
            "Random Forest": "Ensemble of decision trees with bagging",
            "Gradient Boosting": "Boosted ensemble with gradient optimization",
            "SVM": "Support Vector Machine for regression",
            "Neural Network": "Multi-layer perceptron"
        },
        "features_used": [
            "value",
            "temperature",
            "drift",
            "noise"
        ]
    }


# ============== Real Datasets Endpoints ==============

@app.get("/api/datasets", tags=["Real Datasets"])
async def list_available_datasets():
    """
    List all available real-world datasets.
    
    Returns information about NASA IMS Bearing, CWRU Bearing, and MEMS Vibration datasets.
    """
    try:
        datasets = dataset_loader.list_datasets()
        return {
            "datasets": datasets,
            "total": len(datasets)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class LoadDatasetRequest(BaseModel):
    dataset_id: str
    degradation_stage: Optional[int] = 0
    fault_type: Optional[str] = 'normal'
    scenario: Optional[str] = 'normal'


@app.post("/api/datasets/load", tags=["Real Datasets"])
async def load_real_dataset(request: LoadDatasetRequest):
    """
    Load a real-world dataset for analysis.
    
    Available datasets:
    - **nasa_bearing**: NASA IMS Bearing run-to-failure data (degradation_stage: 0-4)
    - **cwru_bearing**: CWRU motor bearing fault data (fault_type: normal, inner_race, outer_race, ball)
    - **mems_vibration**: MEMS accelerometer data (scenario: normal, high_vibration, shock_event, drift)
    """
    try:
        # Prepare kwargs based on dataset type
        kwargs = {}
        if request.dataset_id == 'nasa_bearing':
            kwargs['degradation_stage'] = request.degradation_stage
        elif request.dataset_id == 'cwru_bearing':
            kwargs['fault_type'] = request.fault_type
        elif request.dataset_id == 'mems_vibration':
            kwargs['scenario'] = request.scenario
        
        data, info = dataset_loader.load_dataset(request.dataset_id, **kwargs)
        
        # Extract features for the loaded data
        features = data_processor.extract_features(data)
        
        # Detect anomalies
        anomalies = data_processor.detect_anomalies(data)
        
        # Calculate RUL based on degradation
        if request.dataset_id == 'nasa_bearing':
            rul = max(0, 100 - request.degradation_stage * 25)
        else:
            rul = data_processor.calculate_rul(data)
        
        # Extract sensor characteristics
        sensor_characteristics = data_processor.extract_sensor_characteristics(
            data, 'accelerometer'
        )
        
        return {
            "data": data.to_dict('records'),
            "features": features,
            "anomalies": anomalies,
            "rul": float(rul),
            "sensor_characteristics": sensor_characteristics,
            "dataset_info": {
                "name": info.name,
                "description": info.description,
                "source": info.source,
                "sampling_rate": info.sampling_rate,
                "sensor_type": info.sensor_type
            },
            "metadata": {
                "dataset_id": request.dataset_id,
                "num_samples": len(data),
                "loaded_at": datetime.now().isoformat()
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============== History Endpoints ==============

@app.get("/api/history/sensor-data", tags=["History"])
async def get_sensor_data_history(limit: int = 50, db: Session = Depends(get_db)):
    """
    Get historical sensor data records.
    
    Returns a list of previous sensor data sessions with metadata.
    """
    try:
        records = DatabaseOperations.get_all_sensor_records(db, limit=limit)
        return {
            "records": [
                {
                    "id": r.id,
                    "session_id": r.session_id,
                    "sensor_type": r.sensor_type,
                    "num_samples": r.num_samples,
                    "degradation_level": r.degradation_level,
                    "mean_value": r.mean_value,
                    "snr": r.snr,
                    "rul": r.rul,
                    "anomaly_count": r.anomaly_count,
                    "created_at": r.created_at.isoformat() if r.created_at else None
                }
                for r in records
            ],
            "total": len(records)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history/trained-models", tags=["History"])
async def get_trained_models_history(db: Session = Depends(get_db)):
    """
    Get list of all trained models with performance metrics.
    """
    try:
        models = DatabaseOperations.get_all_trained_models(db)
        return {
            "models": [
                {
                    "id": m.id,
                    "model_name": m.model_name,
                    "model_type": m.model_type,
                    "accuracy": m.accuracy,
                    "mse": m.mse,
                    "r2_score": m.r2_score,
                    "f1_score": m.f1_score,
                    "training_samples": m.training_samples,
                    "created_at": m.created_at.isoformat() if m.created_at else None
                }
                for m in models
            ],
            "total": len(models)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history/sensor-data/{session_id}", tags=["History"])
async def get_sensor_data_by_session(session_id: str, db: Session = Depends(get_db)):
    """
    Get specific sensor data session by ID.
    """
    try:
        record = DatabaseOperations.get_sensor_data_by_session(db, session_id)
        if not record:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "id": record.id,
            "session_id": record.session_id,
            "sensor_type": record.sensor_type,
            "num_samples": record.num_samples,
            "data": json.loads(record.data_json) if record.data_json else [],
            "features": {
                "mean": record.mean_value,
                "std": record.std_value,
                "snr": record.snr
            },
            "rul": record.rul,
            "anomaly_count": record.anomaly_count,
            "created_at": record.created_at.isoformat() if record.created_at else None
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== EMAIL ALERTS ====================

@app.post("/api/send-email", tags=["Email Alerts"])
async def send_email_alert(request: EmailAlertRequest):
    """
    Send email alert notification via Gmail SMTP.
    
    Requires Gmail App Password (not regular password).
    Set environment variables: GMAIL_EMAIL and GMAIL_APP_PASSWORD
    """
    try:
        # Create email content
        subject = f"[{request.alert_type}] MEMS Sensor Alert - {request.sensor_type}"
        
        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; background-color: #1a1a2e; color: #ffffff; padding: 20px;">
            <div style="max-width: 600px; margin: 0 auto; background-color: #16213e; border-radius: 10px; padding: 30px;">
                <h1 style="color: #e94560; margin-bottom: 20px;">MEMS Sensor Alert</h1>
                
                <div style="background-color: #0f3460; border-radius: 8px; padding: 20px; margin-bottom: 20px;">
                    <h2 style="color: #ffffff; margin-top: 0;">Alert Details</h2>
                    <table style="width: 100%; color: #ffffff;">
                        <tr>
                            <td style="padding: 8px 0; color: #94a3b8;">Alert Type:</td>
                            <td style="padding: 8px 0; color: {'#ef4444' if request.alert_type.upper() == 'CRITICAL' else '#f59e0b'}; font-weight: bold;">{request.alert_type.upper()}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; color: #94a3b8;">Sensor:</td>
                            <td style="padding: 8px 0;">{request.sensor_type.upper()}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; color: #94a3b8;">Remaining Useful Life:</td>
                            <td style="padding: 8px 0; color: {'#22c55e' if float(request.rul) > 50 else '#ef4444'};">{request.rul}%</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; color: #94a3b8;">Status:</td>
                            <td style="padding: 8px 0;">{request.status}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; color: #94a3b8;">Time:</td>
                            <td style="padding: 8px 0;">{request.timestamp}</td>
                        </tr>
                    </table>
                </div>
                
                <p style="color: #94a3b8; font-size: 12px; margin-top: 20px;">
                    This is an automated alert from MEMS Sensor ML Analysis System.<br>
                    Predictive Maintenance Platform
                </p>
            </div>
        </body>
        </html>
        """
        
        # Create message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = EMAIL_CONFIG["sender_email"]
        msg["To"] = request.to_email
        
        # Attach HTML content
        msg.attach(MIMEText(html_body, "html"))
        
        # Check if credentials are configured
        if EMAIL_CONFIG["sender_email"] == "your-email@gmail.com":
            # Demo mode - simulate sending
            print(f"[DEMO MODE] Email would be sent to: {request.to_email}")
            print(f"[DEMO MODE] Subject: {subject}")
            return {
                "success": True,
                "message": "Email sent successfully (Demo Mode)",
                "demo_mode": True,
                "recipient": request.to_email
            }
        
        # Send email via Gmail SMTP
        with smtplib.SMTP(EMAIL_CONFIG["smtp_server"], EMAIL_CONFIG["smtp_port"]) as server:
            server.starttls()
            server.login(EMAIL_CONFIG["sender_email"], EMAIL_CONFIG["sender_password"])
            server.sendmail(EMAIL_CONFIG["sender_email"], request.to_email, msg.as_string())
        
        print(f"Email sent successfully to: {request.to_email}")
        return {
            "success": True,
            "message": "Email sent successfully",
            "recipient": request.to_email
        }
        
    except smtplib.SMTPAuthenticationError:
        raise HTTPException(
            status_code=401, 
            detail="Gmail authentication failed. Check your App Password."
        )
    except Exception as e:
        print(f"Email error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to send email: {str(e)}")


# ==================== DEEP LEARNING (LSTM) ====================

# Try to import LSTM model - graceful fallback if TensorFlow not available
try:
    from models.lstm_model import LSTMPredictor, generate_synthetic_rul_data, TENSORFLOW_AVAILABLE
    print(f"[LSTM] TensorFlow available: {TENSORFLOW_AVAILABLE}")
except Exception as e:
    print(f"[LSTM] Import failed: {type(e).__name__}: {e}")
    TENSORFLOW_AVAILABLE = False
    LSTMPredictor = None

# Store LSTM predictor instance
lstm_predictor = None

class LSTMTrainRequest(BaseModel):
    sensor_data: List[dict]
    sequence_length: int = 30
    epochs: int = 20
    batch_size: int = 32

class LSTMPredictRequest(BaseModel):
    sensor_data: List[dict]


@app.get("/api/lstm-status", tags=["Deep Learning"])
async def get_lstm_status():
    """
    Check LSTM/TensorFlow availability and model status
    """
    global lstm_predictor
    
    return {
        "tensorflow_available": TENSORFLOW_AVAILABLE,
        "model_trained": lstm_predictor is not None and lstm_predictor.is_trained,
        "message": "TensorFlow is available" if TENSORFLOW_AVAILABLE else "TensorFlow not installed. Run: pip install tensorflow"
    }


@app.post("/api/lstm-train", tags=["Deep Learning"])
async def train_lstm_model(request: LSTMTrainRequest):
    """
    Train LSTM model for RUL prediction
    
    Requires TensorFlow to be installed.
    """
    global lstm_predictor
    
    if not TENSORFLOW_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="TensorFlow not installed. Run: pip install tensorflow"
        )
    
    try:
        # Extract features from sensor data
        sensor_data = request.sensor_data
        
        # Prepare features and RUL labels
        features = []
        rul_values = []
        
        for i, sample in enumerate(sensor_data):
            # Extract relevant features
            feature_vector = [
                sample.get('value', 0),
                sample.get('noise', 0),
                sample.get('drift', 0),
                sample.get('snr', 50),
                i / len(sensor_data)  # Normalized time
            ]
            features.append(feature_vector)
            
            # RUL decreases over time (simulating degradation)
            rul = 100 - (i / len(sensor_data)) * 100
            rul_values.append(rul)
        
        features = np.array(features)
        rul_values = np.array(rul_values)
        
        # Normalize features
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        
        # Create LSTM predictor
        lstm_predictor = LSTMPredictor(
            sequence_length=request.sequence_length,
            n_features=features.shape[1]
        )
        
        # Build model
        lstm_predictor.build_model(lstm_units=64, dense_units=32)
        
        # Prepare sequences
        X, y = lstm_predictor.prepare_sequences(features, rul_values)
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train model
        history = lstm_predictor.train(
            X_train, y_train,
            X_val, y_val,
            epochs=request.epochs,
            batch_size=request.batch_size,
            verbose=0
        )
        
        # Save model
        model_path = lstm_predictor.save_model()
        
        return {
            "success": True,
            "message": f"LSTM model trained successfully for {history['epochs_trained']} epochs",
            "training_history": {
                "loss": [float(l) for l in history['loss']],
                "mae": [float(m) for m in history['mae']],
                "val_loss": [float(l) for l in history['val_loss']],
                "val_mae": [float(m) for m in history['val_mae']],
                "epochs_trained": history['epochs_trained']
            },
            "model_path": model_path,
            "final_loss": float(history['loss'][-1]),
            "final_val_loss": float(history['val_loss'][-1]) if history['val_loss'] else None
        }
        
    except Exception as e:
        print(f"LSTM Training Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"LSTM training failed: {str(e)}")


@app.post("/api/lstm-predict", tags=["Deep Learning"])
async def predict_with_lstm(request: LSTMPredictRequest):
    """
    Make RUL predictions using trained LSTM model
    """
    global lstm_predictor
    
    if not TENSORFLOW_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="TensorFlow not installed"
        )
    
    if lstm_predictor is None or not lstm_predictor.is_trained:
        raise HTTPException(
            status_code=400,
            detail="LSTM model not trained. Train the model first using /api/lstm-train"
        )
    
    try:
        # Extract features
        features = []
        for sample in request.sensor_data:
            feature_vector = [
                sample.get('value', 0),
                sample.get('noise', 0),
                sample.get('drift', 0),
                sample.get('snr', 50),
                0  # Time placeholder
            ]
            features.append(feature_vector)
        
        features = np.array(features)
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        
        # Prepare sequences (use last sequence_length samples)
        if len(features) >= lstm_predictor.sequence_length:
            X = lstm_predictor.prepare_sequences(features)
            predictions = lstm_predictor.predict(X)
            
            return {
                "success": True,
                "predictions": predictions.tolist(),
                "current_rul": float(predictions[-1]) if len(predictions) > 0 else None,
                "average_rul": float(np.mean(predictions)),
                "min_rul": float(np.min(predictions)),
                "max_rul": float(np.max(predictions))
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least {lstm_predictor.sequence_length} samples for prediction"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("Starting MEMS Sensor ML Analysis API Server")
    print("="*60)
    print(f"API URL: http://localhost:8000")
    print(f"Documentation: http://localhost:8000/docs")
    print(f"Health Check: http://localhost:8000/health")
    print(f"Database: SQLite (mems_ml_database.db)")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)