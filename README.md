# 🔬 MEMS Sensor ML Analysis & Predictive Maintenance System

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18+-blue.svg)](https://reactjs.org)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.x-cyan.svg)](https://tailwindcss.com)

A full-stack ML-powered platform for **MEMS sensor health monitoring**, **Remaining Useful Life (RUL) prediction**, and **explainable AI analysis**.

![Dashboard Preview](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)

---

## 🌐 Live Demo

| Service | URL |
|---------|-----|
| **Frontend Dashboard** | [mems-ml-system.vercel.app](https://mems-ml-system.vercel.app) |
| **Backend API** | [mems-ml-system.onrender.com](https://mems-ml-system.onrender.com) |
| **API Documentation** | [/docs](https://mems-ml-system.onrender.com/docs) |

---

## 🚀 Features

### Core Capabilities
- **Multi-Sensor Support**: Accelerometer, Gyroscope, Pressure, Temperature sensors
- **Real-time Visualization**: Interactive charts with Recharts
- **ML Model Training**: Random Forest, XGBoost, SVR, Gradient Boosting
- **Explainable AI (XAI)**: SHAP values, feature importance, decision rules
- **Rolling Z-Score Anomaly Detection**: Configurable threshold (2.0-4.0) and window size (20-200)
- **RUL Prediction**: Remaining Useful Life estimation with confidence intervals

### Advanced Processing
- **Fast Fourier Transform (FFT)**: Backend numpy.fft for O(N log N) performance
- **Wavelet Transform**: Energy ratio analysis with interpretation
- **Unified Health Report API**: Single source of truth for RUL, status, risks, forecast

### Dashboard Tabs
| Tab | Description |
|-----|-------------|
| 📊 **Data** | Sensor data visualization and export |
| 🤖 **Models** | ML model training and comparison leaderboards |
| 🎯 **Prediction** | RUL forecast with confidence bounds |
| 🔍 **XAI** | SHAP feature importance and explanations |
| ⚠️ **Anomaly** | Rolling Z-score detection with dynamic thresholding |

### Production Features
- **PDF Report Generation**: Professional reports with charts
- **Email Alerts**: Gmail SMTP integration for maintenance notifications
- **Model Download**: Export trained models as .joblib files
- **Database Storage**: SQLite with SQLAlchemy ORM
- **Docker Ready**: Full containerization

---

## 📁 Project Structure

```
mems-ml-system/
├── backend/                          # FastAPI Backend
│   ├── main.py                       # API endpoints
│   ├── requirements.txt              # Python dependencies
│   ├── Dockerfile                    # Docker configuration
│   ├── database/
│   │   └── database.py               # SQLAlchemy models
│   ├── models/
│   │   ├── ml_models.py              # ML training & persistence
│   │   ├── data_processor.py         # Data processing utilities
│   │   └── xai_analyzer.py           # XAI analysis
│   ├── data/
│   │   └── dataset_loader.py         # NASA/CWRU dataset loaders
│   ├── utils/
│   │   └── health_report.py          # Unified health report builder
│   └── saved_models/                 # Persisted ML models
│
├── frontend/                         # React + Vite Frontend
│   ├── src/
│   │   ├── MEMSDashboard.jsx         # Main dashboard
│   │   ├── main.jsx                  # Entry point
│   │   ├── services/
│   │   │   └── api.js                # API client
│   │   ├── utils/
│   │   │   ├── signalProcessing.js   # FFT/Wavelet
│   │   │   └── anomalyDetection.js   # Rolling Z-score detection
│   │   └── components/               # Reusable components
│   ├── package.json                  # Node dependencies
│   ├── vite.config.js                # Vite configuration
│   └── Dockerfile                    # Docker configuration
│
├── docker-compose.yml                # Full-stack deployment
└── render.yaml                       # Render deployment config
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

# Create virtual environment
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

### Local Access

| Service | URL |
|---------|-----|
| Frontend Dashboard | http://localhost:5173 |
| Backend API | http://localhost:8000 |
| API Documentation | http://localhost:8000/docs |
| ReDoc | http://localhost:8000/redoc |

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

### Signal Processing
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/signal/fft` | POST | Fast Fourier Transform (numpy.fft) |

### Data Generation
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/generate-data` | POST | Generate synthetic sensor data |
| `/api/datasets/list` | GET | List available real datasets |

### ML Models
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/train-models` | POST | Train ML models |
| `/api/models/download-best` | GET | Download best model (.joblib) |
| `/api/models/save` | POST | Save trained models |
| `/api/models/load/{session_id}` | POST | Load saved models |

### Health & XAI
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health-report` | POST | Unified health report |
| `/api/xai-analysis` | POST | SHAP explainability |

### Notifications
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/send-alert` | POST | Email alert via Gmail SMTP |

---

## 🧪 Technology Stack

### Backend
| Technology | Purpose |
|------------|---------|
| FastAPI | REST API framework |
| SQLAlchemy | Database ORM |
| scikit-learn | Machine learning |
| XGBoost | Gradient boosting |
| NumPy/Pandas | Data processing |
| SHAP | Explainable AI |
| joblib | Model persistence |

### Frontend
| Technology | Purpose |
|------------|---------|
| React 18 | UI framework |
| Vite | Build tool |
| Tailwind CSS | Styling |
| Recharts | Data visualization |
| html2canvas | Chart capture |
| jsPDF | PDF generation |
| Lucide React | Icons |

---

## 📈 ML Models

| Model | Algorithm | Use Case |
|-------|-----------|----------|
| Random Forest | Ensemble trees | Robust baseline |
| XGBoost | Gradient boosting | High accuracy |
| SVR | Support Vector Regression | Small datasets |
| Gradient Boosting | Sequential boosting | Complex patterns |

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    MEMS ML System Pipeline                           │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  📊 Sensor Data → 🔧 Preprocessing → ⚙️ Feature Extraction            │
│       │                                     │                         │
│       ▼                                     ▼                         │
│  🧠 ML Models ─────────────────────→ 📈 RUL Prediction               │
│       │                                     │                         │
│       ▼                                     ▼                         │
│  🔍 XAI Analysis ──────────────────→ 🖥️ Dashboard/Alerts            │
│                                             │                         │
│                                             ▼                         │
│                                      📄 Report Export                 │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Data Sources
- **Synthetic Data**: Configurable degradation simulation
- **NASA Bearing Dataset**: Real bearing failure data
- **CWRU Bearing Dataset**: Classification benchmarks
- **CSV Upload**: Custom sensor data

---

## 🔧 Configuration

### Frontend Environment (`.env`)
```bash
VITE_API_URL=https://mems-ml-system.onrender.com
```

### Backend Environment (Render / Local)
```bash
GMAIL_EMAIL=your-email@gmail.com
GMAIL_APP_PASSWORD=your-app-password
```

---

## 🚀 Deployment

### Frontend → Vercel
1. Push to GitHub
2. Connect repository to [Vercel](https://vercel.com)
3. Set root directory: `frontend`
4. Add environment variable: `VITE_API_URL`

### Backend → Render
1. Connect repository to [Render](https://render.com)
2. Use `render.yaml` for auto-configuration
3. Add Gmail credentials for email alerts

---

## 📄 License

This project is licensed under the MIT License.
