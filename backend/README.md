# MEMS Sensor ML Analysis - Backend API

## 🐍 Python FastAPI Backend

This is the backend API server for the MEMS Sensor ML Analysis System.

## 📦 Installation
```bash
# Install dependencies
pip install -r requirements.txt
```

## 🚀 Running the Server
```bash
# From the backend directory
python main.py

# Or with uvicorn
uvic

## 📚 API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 🔌 Endpoints

- `POST /api/generate-data` - Generate synthetic sensor data
- `POST /api/upload-data` - Upload CSV file
- `POST /api/train-models` - Train ML models
- `POST /api/xai-analysis` - Get explainability analysis
- `POST /api/predict` - Make prediction
- `GET /api/models/info` - Get model information

## 🧪 Technologies

- **FastAPI** - Modern web framework
- **scikit-learn** - Machine learning
- **pandas** - Data processing
- **NumPy** - Numerical computing
- **Pydantic** - Data validation