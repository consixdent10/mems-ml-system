# 🔬 MEMS Sensor ML Analysis & Prediction System

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18+-blue.svg)](https://reactjs.org)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.x-cyan.svg)](https://tailwindcss.com)

Advanced Machine Learning-Based Performance Analysis and Predictive Maintenance System for MEMS Sensors.

![Dashboard Preview](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)

---

## 🚀 Features

### Core Capabilities
- **Multi-Sensor Support**: Accelerometer, Gyroscope, Pressure, Temperature sensors
- **Real-time Data Visualization**: Interactive charts with Recharts
- **ML Model Training**: Random Forest, Gradient Boosting, SVM, Neural Network
- **Explainable AI (XAI)**: SHAP values and feature importance analysis
- **Anomaly Detection**: Z-score based isolation forest approach
- **Predictive Maintenance**: RUL (Remaining Useful Life) prediction

### Advanced Features
- **Deep Learning (LSTM)**: Neural network for time-series RUL prediction
- **Signal Processing**: FFT and Wavelet Transform analysis
- **Model Persistence**: Save/load trained models using joblib
- **PDF Report Generation**: Professional reports with jspdf
- **Email Alerts**: Gmail SMTP integration for maintenance notifications
- **Live Monitoring**: Real-time sensor data simulation
- **Database Storage**: SQLite with SQLAlchemy ORM
- **API Documentation**: Swagger UI at `/docs`
- **Docker Ready**: Full containerization with Docker Compose

---

## 📁 Project Structure

```
mems-ml-system/
├── backend/                    # FastAPI Backend
│   ├── main.py                 # API endpoints
│   ├── requirements.txt        # Python dependencies
│   ├── Dockerfile              # Docker configuration
│   ├── database/
│   │   └── database.py         # SQLAlchemy models
│   ├── models/
│   │   ├── ml_models.py        # ML training & persistence
│   │   ├── data_processor.py   # Data processing utilities
│   │   └── xai_analyzer.py     # XAI analysis
│   └── saved_models/           # Persisted ML models
│
├── frontend/                   # React Frontend
│   ├── src/
│   │   ├── MEMSDashboard.jsx   # Main dashboard component
│   │   └── main.jsx            # Entry point
│   ├── package.json            # Node dependencies
│   ├── vite.config.js          # Vite configuration
│   └── Dockerfile              # Docker configuration
│
└── docker-compose.yml          # Full-stack deployment
```

---

## 🛠️ Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- npm or yarn

### Backend Setup

```bash
cd backend

# Create virtual environment (optional)
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Start the server
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### Access the Application

| Service | URL |
|---------|-----|
| Frontend Dashboard | http://localhost:5173 |
| Backend API | http://localhost:8000 |
| API Documentation | http://localhost:8000/docs |
| Alternative Docs | http://localhost:8000/redoc |

---

## 🐳 Docker Deployment

```bash
# Build and run all services
docker-compose up --build

# Run in background
docker-compose up -d

# Stop services
docker-compose down
```

---

## 📊 API Endpoints

### Data Generation
- `POST /api/generate-data` - Generate synthetic sensor data

### Data Upload
- `POST /api/upload-data` - Upload CSV sensor data

### ML Models
- `POST /api/train-models` - Train ML models
- `POST /api/models/save` - Save trained models
- `POST /api/models/load/{session_id}` - Load saved models
- `GET /api/models/saved` - List saved model sessions
- `DELETE /api/models/saved/{session_id}` - Delete saved session
- `GET /api/models/info` - Get model information

### XAI Analysis
- `POST /api/xai-analysis` - Generate explainability analysis

### Predictions
- `POST /api/predict` - Make predictions

### History
- `GET /api/history/sensor-data` - Get historical sensor data
- `GET /api/history/trained-models` - Get trained model history

### System
- `GET /health` - Health check
- `GET /docs` - Swagger UI

---

## 🧪 Technologies Used

### Backend
| Technology | Purpose |
|------------|---------|
| FastAPI | REST API framework |
| SQLAlchemy | Database ORM |
| scikit-learn | Machine learning |
| NumPy/Pandas | Data processing |
| joblib | Model persistence |
| Uvicorn | ASGI server |

### Frontend
| Technology | Purpose |
|------------|---------|
| React 18 | UI framework |
| Vite | Build tool |
| Tailwind CSS | Styling |
| Recharts | Data visualization |
| jsPDF | PDF generation |
| Lucide React | Icons |

---

## 📈 ML Models

| Model | Description | Use Case |
|-------|-------------|----------|
| Random Forest | Ensemble of decision trees | Robust predictions |
| Gradient Boosting | Sequential boosted trees | High accuracy |
| SVM | Support Vector Machine | Complex boundaries |
| Neural Network | Multi-layer perceptron | Non-linear patterns |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MEMS ML Analysis System                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       Frontend (React + Vite)                        │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────────────┐ │   │
│  │  │   Data    │  │ Analysis  │  │   XAI     │  │    Live Monitor   │ │   │
│  │  │   Tab     │  │    Tab    │  │   Tab     │  │       Tab         │ │   │
│  │  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └────────┬──────────┘ │   │
│  └────────┼──────────────┼──────────────┼─────────────────┼────────────┘   │
│           │              │              │                 │                │
│           └──────────────┴──────────────┴─────────────────┘                │
│                                    │                                        │
│                            REST API Calls                                   │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Backend (FastAPI + Python)                      │   │
│  │                                                                      │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │   │
│  │  │ Data         │  │   ML Model   │  │      XAI Analyzer        │   │   │
│  │  │ Processor    │  │   Trainer    │  │   (SHAP, Importance)     │   │   │
│  │  └──────┬───────┘  └──────┬───────┘  └────────────┬─────────────┘   │   │
│  │         │                 │                       │                  │   │
│  │         ▼                 ▼                       ▼                  │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │                    scikit-learn / NumPy                      │   │   │
│  │  │   Random Forest │ Gradient Boosting │ SVM │ Neural Network   │   │   │
│  │  └──────────────────────────────────────────────────────────────┘   │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌───────────────────────┐  ┌────────────────────────────────────────┐     │
│  │   SQLite Database     │  │        Saved Models (joblib)           │     │
│  │  (SQLAlchemy ORM)     │  │    saved_models/{session_id}/          │     │
│  └───────────────────────┘  └────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Diagram

```
┌──────────────┐     ┌───────────────┐     ┌──────────────────┐
│   Synthetic  │     │  Real Dataset │     │    CSV Upload    │
│    Data      │     │  (NASA/CWRU)  │     │                  │
└──────┬───────┘     └───────┬───────┘     └────────┬─────────┘
       │                     │                      │
       └─────────────────────┼──────────────────────┘
                             ▼
                 ┌───────────────────────┐
                 │   Data Preprocessing  │
                 │  • Feature Extraction │
                 │  • Noise Analysis     │
                 │  • FFT / Wavelet      │
                 └───────────┬───────────┘
                             ▼
                 ┌───────────────────────┐
                 │    ML Model Training  │
                 │  • Model Selection    │
                 │  • Cross-Validation   │
                 │  • Hyperparameters    │
                 └───────────┬───────────┘
                             ▼
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  Predictions  │   │  XAI Analysis │   │    Alerts     │
│  • RUL %      │   │  • SHAP       │   │  • Email      │
│  • Anomalies  │   │  • Features   │   │  • Dashboard  │
└───────────────┘   └───────────────┘   └───────────────┘
```

---

## 🔧 Configuration

### Backend Environment Variables
```bash
# Database (default: SQLite)
DATABASE_URL=sqlite:///./mems_ml_database.db
```

### Frontend Configuration
The frontend automatically uses environment variables for production:
```javascript
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
```

---

## 🚀 Deployment

### Frontend (Vercel)
1. Push to GitHub
2. Connect to [Vercel](https://vercel.com)
3. Set environment variable: `VITE_API_URL=https://your-backend.onrender.com`

### Backend (Render)
1. Connect to [Render](https://render.com)
2. Set environment variables:
   - `GMAIL_EMAIL` - For email alerts
   - `GMAIL_APP_PASSWORD` - Gmail App Password

---

## 📄 License

This project is licensed under the MIT License.

---




## 🙏 Acknowledgments

- NASA Prognostics Center for bearing dataset inspiration
- Case Western Reserve University for CWRU Bearing Dataset
- scikit-learn and FastAPI communities
- React and Recharts visualization libraries
