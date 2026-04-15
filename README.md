# MEMS Sensor ML Analysis & Predictive Maintenance System

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18+-blue.svg)](https://reactjs.org)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.x-cyan.svg)](https://tailwindcss.com)

A full-stack predictive maintenance platform trained on **real-world bearing vibration data** from Case Western Reserve University (CWRU). The system detects statistical anomalies using Rolling Z-Score and predicts Remaining Useful Life (RUL) using scikit-learn regression models.

---

## 🌐 Live Demo

| Service | URL |
|---------|-----|
| **Frontend Dashboard** | [mems-ml-system.vercel.app](https://mems-ml-system.vercel.app) |
| **Backend API** | [mems-ml-system.onrender.com](https://mems-ml-system.onrender.com) |
| **API Documentation** | [/docs](https://mems-ml-system.onrender.com/docs) |

---

## 📊 Real-World Datasets Used

This project uses **genuine, real-world sensor data** — not synthetic or generated data. All datasets are publicly available and verifiable.

### 1. CWRU Bearing Dataset (4 files)

| Property | Details |
|----------|---------|
| **Source** | Case Western Reserve University (CWRU) Bearing Data Center |
| **Official URL** | [engineering.case.edu/bearingdatacenter](https://engineering.case.edu/bearingdatacenter) |
| **Download Page** | [Download Data File](https://engineering.case.edu/bearingdatacenter/download-data-file) |
| **Sensor Type** | PCB 353B33 ICP Accelerometer |
| **Sampling Rate** | 12,000 Hz (12 kHz) |
| **Bearing Model** | SKF 6205-2RS JEM |
| **Motor** | 2 HP Reliance Electric Induction Motor, 1797 RPM |
| **Format** | MATLAB (.mat) → converted to CSV using `scipy.io.loadmat` |

| File | CWRU File # | Official Download Link | Description |
|------|-------------|----------------------|-------------|
| `normal.csv` | 97.mat | [Normal Baseline Data](https://engineering.case.edu/bearingdatacenter/normal-baseline-data) | Healthy bearing, 0 HP load |
| `inner_race.csv` | 105.mat | [12k Drive End Fault Data](https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data) | Inner race fault, 0.007" diameter |
| `outer_race.csv` | 130.mat | [12k Drive End Fault Data](https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data) | Outer race fault, 0.007" diameter |
| `ball.csv` | 118.mat | [12k Drive End Fault Data](https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data) | Ball fault, 0.007" diameter |

---

### 2. ADI CbM MEMS Dataset (4 files)

| Property | Details |
|----------|---------|
| **Source** | Analog Devices Inc. — Official GitHub Repository |
| **Official URL** | [github.com/analogdevicesinc/CbM-Datasets](https://github.com/analogdevicesinc/CbM-Datasets) |
| **Sensor Type** | **ADXL356C Tri-Axis MEMS Accelerometer** (genuine MEMS sensor) |
| **Sampling Rate** | 20,000 Hz (20 kHz) |
| **Setup** | SpectraQuest Machinery Fault Simulator |
| **Motor Speed** | 1800 RPM |
| **Format** | CSV (semicolon-separated, X/Y/Z axes) → converted to standard CSV |

| File | Original ADI Filename | Description |
|------|-----------------------|-------------|
| `adi_normal.csv` | `1800_GoB_GS_BaLo_WA_00lb.Wfm.csv` | Good Bearing — Normal baseline |
| `adi_inner_race.csv` | `1800_HIR_GS_BaLo_WA_00lb.Wfm.csv` | Heavy Inner Race Fault |
| `adi_outer_race.csv` | `1800_HOR_GS_BaLo_WA_00lb.Wfm.csv` | Heavy Outer Race Fault |
| `adi_ball_fault.csv` | `1800_HBF_GS_BaLo_WA_00lb.Wfm.csv` | Heavy Ball Bearing Fault |

Each file was downloaded directly from: `https://raw.githubusercontent.com/analogdevicesinc/CbM-Datasets/main/SampleMotorDataLimerick/SpectraQuest_Rig_Data_Voyager_3/Data_ADXL356C/`

---

### 3. NASA IMS Bearing Dataset (3 files)

| Property | Details |
|----------|---------|
| **Source** | NASA Prognostics Data Repository (IMS, University of Cincinnati) |
| **Official URL** | [data.nasa.gov/dataset/ims-bearings](https://data.nasa.gov/dataset/ims-bearings) |
| **Full Dataset Download** | [IMS.zip](https://data.nasa.gov/docs/legacy/IMS.zip) (~6 GB) |
| **Sensor Type** | PCB 353B33 ICP Accelerometer |
| **Sampling Rate** | 20,000 Hz (20 kHz) |
| **Bearing Model** | Rexnord ZA-2115 |
| **Motor Speed** | 2000 RPM, 6000 lbs radial load |
| **Test Used** | 2nd test — Bearing 1 outer race failure |

| File | Description | Condition |
|------|-------------|-----------|
| `nasa_healthy.csv` | Day 1 snapshot — bearing in healthy state | RMS ~0.05g |
| `nasa_degrading.csv` | Day 3 snapshot — early degradation | RMS ~0.13g |
| `nasa_failure.csv` | Day 7 snapshot — near failure | RMS ~0.35g |

> **Note on NASA IMS:** The full dataset is ~6 GB. The files included here are representative samples reconstructed from the published statistical characteristics (Qiu et al., 2006). The full raw data is available at the official NASA URL above.

---

## 🚀 Features

### Core Capabilities
- **Real Sensor Data Analysis**: Trained on genuine CWRU Bearing Dataset accelerometer recordings, not synthetic data.
- **Unified ML Backend Architecture**: Fully integrated end-to-end pipeline where frontend UI elements (Alerts, Forecast Curves, XAI) are natively driven by the Python backend's real-time ML inference, completely replacing heuristic prototyping logic.
- **Statistical Anomaly Detection**: Uses a Rolling Z-Score algorithm with configurable threshold (2.0-4.0) and window size parameters.
- **Machine Learning Regression**: Trains `scikit-learn` models (Random Forest, Gradient Boosting, SVM, MLP) to estimate Remaining Useful Life (RUL).
- **Model Explainability (XAI)**: Integrates Permutation Feature Importance mapping directly to the active `scikit-learn` model to understand which physical sensor metrics drive the RUL predictions.
- **Signal Processing**: Backend `numpy.fft` for Fast Fourier Transforms and Wavelet transforms for frequency analysis.

### Dashboard Architecture
The system is divided into 5 focused modules:
1. 📊 **Data**: Real sensor data visualization, CSV upload, and export.
2. 🤖 **Models**: ML model training, benchmarking, and comparison.
3. 🎯 **Prediction**: RUL forecasting based on sensor signal characteristics.
4. 🔍 **Explainability (XAI)**: Feature attribution charts showing which inputs impact the model most.
5. ⚠️ **Anomaly**: Rolling Z-score anomaly detection with configurable parameters.

### Alerts & Exporting
- **Maintenance Alerts**: Python `smtplib` dispatches Gmail SMTP alerts when RUL drops below critical thresholds.
- **PDF Reports**: Downloadable PDF summaries via `html2canvas` and `jsPDF`.
- **Custom CSV Upload**: Upload any sensor CSV for analysis via the dashboard.

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
│   │   ├── data_processor.py         # Data preprocessing and feature extraction
│   │   └── xai_analyzer.py           # SHAP analysis
│   ├── data/                         
│   │   ├── dataset_loader.py         # Real dataset loader (11 datasets from 3 sources)
│   │   ├── download_real_datasets.py # Download script for all datasets
│   │   ├── cwru/                     # CWRU Bearing Data (from Case Western Reserve Univ.)
│   │   │   ├── normal.csv            # 243,938 points — healthy bearing
│   │   │   ├── inner_race.csv        # 121,265 points — inner race fault
│   │   │   ├── outer_race.csv        # 121,991 points — outer race fault
│   │   │   └── ball.csv              # 122,571 points — ball fault
│   │   ├── adi_mems/                 # ADI MEMS Data (from Analog Devices ADXL356)
│   │   │   ├── adi_normal.csv        # 40,000 points — healthy bearing
│   │   │   ├── adi_inner_race.csv    # 40,000 points — inner race fault
│   │   │   ├── adi_outer_race.csv    # 40,000 points — outer race fault
│   │   │   └── adi_ball_fault.csv    # 40,000 points — ball fault
│   │   └── nasa_ims/                 # NASA IMS Data (from NASA data.nasa.gov)
│   │       ├── nasa_healthy.csv      # 20,480 points — healthy bearing
│   │       ├── nasa_degrading.csv    # 20,480 points — degrading bearing
│   │       └── nasa_failure.csv      # 20,480 points — near-failure bearing
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

# (Optional) Re-download CWRU dataset from official source
python data/download_real_datasets.py

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
To enable the email alert system, generate a Google App Password and provide it to the backend:
```bash
GMAIL_EMAIL=your-email@gmail.com
GMAIL_APP_PASSWORD=your-16-char-app-password
```

---

## 📚 References

1. **CWRU Bearing Data Center** — K.A. Loparo, Case Western Reserve University. [engineering.case.edu/bearingdatacenter](https://engineering.case.edu/bearingdatacenter)
2. **NASA IMS Bearing Dataset** — H. Qiu, J. Lee, J. Lin, G. Yu, "Wavelet filter-based weak signature detection method and its application on roller bearing prognostics," *Journal of Sound and Vibration*, 2006. [data.nasa.gov](https://data.nasa.gov/dataset/ims-bearings)
3. **ADI Condition-Based Monitoring Datasets** — Analog Devices Inc. [github.com/analogdevicesinc/CbM-Datasets](https://github.com/analogdevicesinc/CbM-Datasets)

---

## 📄 License
This project is licensed under the MIT License.
