# MEMS Sensor ML Analysis & Predictive Maintenance System

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18+-blue.svg)](https://reactjs.org)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.x-cyan.svg)](https://tailwindcss.com)

A full-stack predictive maintenance platform designed to monitor MEMS sensor data, detect statistical anomalies, and predict Remaining Useful Life (RUL) using standard Machine Learning regression algorithms.

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
- **Sensor Monitoring**: Visualizes time-series data from Accelerometers, Gyroscopes, Pressure, and Temperature sensors.
- **Statistical Anomaly Detection**: Uses a Rolling Z-Score algorithm with configurable threshold (2.0-4.0) and window size parameters to flag sensor anomalies in real-time.
- **Machine Learning Regression**: Trains `scikit-learn` models (Random Forest, Gradient Boosting, SVM, and Multi-Layer Perceptron) to estimate Remaining Useful Life (RUL).
- **Model Explainability**: Integrates SHAP (SHapley Additive exPlanations) to extract feature importance and understand exactly which sensor metrics are driving the RUL predictions.
- **Signal Processing**: Utilizes backend `numpy.fft` for Fast Fourier Transforms (FFT) and Wavelet transforms to analyze sensor noise and drift frequencies.

### Dashboard Architecture
The system is divided into 5 focused modules:
1. 📊 **Data**: Raw sensor data visualization and CSV export.
2. 🤖 **Models**: ML model training, benchmarking, and comparison tracking.
3. 🎯 **Prediction**: RUL forecasting based on the current sensor degradation curve.
4. 🔍 **Explainability (XAI)**: SHAP-powered charts detailing which features impact the model most.
5. ⚠️ **Anomaly**: Live Z-score anomaly plotting and historical lists.

### Alerts & Exporting
- **Maintenance Alerts**: Integrated Python `smtplib` to dispatch automated Gmail SMTP alerts to maintenance teams when RUL drops below critical thresholds.
- **PDF Reports**: Generates downloadable PDF summaries of sensor conditions using `html2canvas` and `jsPDF`.

---

## 📁 Project Structure

```
mems-ml-system/
├── backend/                          # FastAPI Backend
│   ├── main.py                       # Core API routing
│   ├── requirements.txt              # Pipeline dependencies (scikit-learn, fastapi, shap)
│   ├── database/                     # SQLite operations
│   ├── models/                       
│   │   ├── ml_models.py              # ML model definitions and training logic
│   │   ├── data_processor.py         # Data preprocessing
│   │   └── xai_analyzer.py           # SHAP analysis
│   ├── data/                         # CSV loaders for CWRU datasets
│   └── utils/                        # Backend utilities
│
├── frontend/                         # React + Vite Frontend
│   ├── src/
│   │   ├── MEMSDashboard.jsx         # Central React application
│   │   ├── services/api.js           # Fetch requests to backend
│   │   └── utils/                    # Frontend fallback logic
│   ├── package.json                  
│   ├── tailwind.config.js            
│   └── vite.config.js                
│
└── docker-compose.yml                
```

---

## 🛠️ Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Start the local server
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start React development server
npm run dev
```

---

## 🔧 Deployment Configuration

### Frontend (Vercel)
Requires one environment variable in the Vercel dashboard:
```bash
VITE_API_URL=https://mems-ml-system.onrender.com
```

### Backend (Render / Local `.env`)
To enable the free email alert system, generate a Google App Password and provide it to the backend:
```bash
GMAIL_EMAIL=your-email@gmail.com
GMAIL_APP_PASSWORD=your-16-char-app-password
```

---

## 📄 License
This project is licensed under the MIT License.
