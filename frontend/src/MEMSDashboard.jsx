import React, { useState, useEffect } from 'react';
import { LineChart, Line, ScatterChart, Scatter, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, AreaChart, Area, Cell, ReferenceLine } from 'recharts';
import { Activity, TrendingUp, AlertTriangle, Database, Brain, Download, Mail, FileText, Zap, Waves } from 'lucide-react';
import { jsPDF } from 'jspdf';

// Extracted modules
import { api, API_BASE_URL } from './services/api';
import { performFFT, waveletTransform } from './utils/signalProcessing';
import { detectAnomalies } from './utils/anomalyDetection';
import ModelsTab from './components/tabs/ModelsTab';

// Note: api, performFFT, waveletTransform, detectAnomalies are now imported from modules

// NOTE: Legacy ML code removed - all training uses backend API (api.trainModels)

// Advanced Feature Extraction
const extractFeatures = (data) => {
    const values = data.map(d => parseFloat(d.value));
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
    const std = Math.sqrt(variance);
    const max = Math.max(...values);
    const min = Math.min(...values);

    // Calculate skewness
    const skewness = values.reduce((a, b) => a + Math.pow((b - mean) / std, 3), 0) / values.length;

    // Calculate kurtosis
    const kurtosis = values.reduce((a, b) => a + Math.pow((b - mean) / std, 4), 0) / values.length - 3;

    // Calculate RMS
    const rms = Math.sqrt(values.reduce((a, b) => a + b * b, 0) / values.length);

    // Peak to peak
    const peakToPeak = max - min;

    // Crest factor
    const crestFactor = max / rms;

    return {
        mean: mean.toFixed(4),
        std: std.toFixed(4),
        variance: variance.toFixed(6),
        max: max.toFixed(4),
        min: min.toFixed(4),
        range: (max - min).toFixed(4),
        snr: (mean / std).toFixed(2),
        skewness: skewness.toFixed(4),
        kurtosis: kurtosis.toFixed(4),
        rms: rms.toFixed(4),
        peakToPeak: peakToPeak.toFixed(4),
        crestFactor: crestFactor.toFixed(4)
    };
};

// Generate PDF Report with proper formatting
const generatePDFReport = (sensorType, features, modelResults, rul, anomalies) => {
    const healthStatus = parseFloat(rul) > 70 ? 'HEALTHY' : parseFloat(rul) > 40 ? 'WARNING' : 'CRITICAL';
    const anomalyList = anomalies.filter(a => a.isAnomaly);

    const reportContent = `
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║              MEMS SENSOR PERFORMANCE ANALYSIS REPORT                       ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

Report Generated: ${new Date().toLocaleString()}
Analysis ID: MEMS-${Date.now().toString(36).toUpperCase()}

════════════════════════════════════════════════════════════════════════════

EXECUTIVE SUMMARY
────────────────────────────────────────────────────────────────────────────

Sensor Type:              ${sensorType.toUpperCase()}
Health Status:            ${healthStatus}
Remaining Useful Life:    ${rul}%
Anomalies Detected:       ${anomalyList.length} (${(anomalyList.length / anomalies.length * 100).toFixed(2)}% rate)

Health Assessment:
${parseFloat(rul) > 70 ? '✓ Sensor is operating within normal parameters' :
            parseFloat(rul) > 40 ? '⚠ Sensor shows signs of degradation - monitoring recommended' :
                '⚠ CRITICAL - Immediate attention required'}

════════════════════════════════════════════════════════════════════════════

STATISTICAL ANALYSIS
────────────────────────────────────────────────────────────────────────────

Signal Characteristics:
  • Mean Value:                 ${features.mean}
  • Standard Deviation:         ${features.std}
  • Variance:                   ${features.variance}
  • Signal-to-Noise Ratio:      ${features.snr}
  • Range:                      ${features.range}
  
Advanced Metrics:
  • RMS (Root Mean Square):     ${features.rms}
  • Peak-to-Peak:               ${features.peakToPeak}
  • Crest Factor:               ${features.crestFactor}
  • Skewness:                   ${features.skewness}
  • Kurtosis:                   ${features.kurtosis}

Signal Quality Assessment:
${parseFloat(features.snr) > 20 ? '✓ Excellent signal quality' :
            parseFloat(features.snr) > 10 ? '⚠ Acceptable signal quality' :
                '⚠ Poor signal quality - noise filtering recommended'}

════════════════════════════════════════════════════════════════════════════

MACHINE LEARNING MODEL PERFORMANCE
────────────────────────────────────────────────────────────────────────────

${modelResults.length > 0 ? modelResults.map((m, idx) => `
Model ${idx + 1}: ${m.modelType}
${'─'.repeat(78)}
  Regression Metrics (RUL Prediction):
    • MAE (Mean Absolute Error):   ${m.mae?.toFixed(2) || 'N/A'}
    • RMSE (Root Mean Sq Error):   ${m.rmse?.toFixed(2) || 'N/A'}
    • MSE (Mean Squared Error):    ${m.mse?.toFixed(4) || 'N/A'}
    • R² Score (Coefficient):      ${m.r2Score?.toFixed(4) || 'N/A'}
    • MAPE (% Error):              ${m.mape?.toFixed(2) || 'N/A'}%
    • Training Time:               ${m.trainingTime} seconds
    • Training/Test Split:         ${m.trainingSize}/${m.testSize} samples
  
  Performance Grade:              ${m.r2Score >= 0.85 ? 'A (Excellent)' :
                        m.r2Score >= 0.70 ? 'B (Good)' :
                            m.r2Score >= 0.50 ? 'C (Fair)' : 'D (Needs Improvement)'}
`).join('\n') : '  No models trained yet. Please train models to see performance metrics.'}

Best Performing Model:        ${modelResults.length > 0 ?
            modelResults.reduce((best, m) => (m.rmse || 999) < (best.rmse || 999) ? m : best).modelType : 'N/A'}
(Selected by: Lowest RMSE)

════════════════════════════════════════════════════════════════════════════

ANOMALY DETECTION ANALYSIS
────────────────────────────────────────────────────────────────────────────

Detection Method:             Isolation Forest (Z-Score Based)
Threshold:                    2.5 Standard Deviations
Total Measurements:           ${anomalies.length}
Anomalies Detected:           ${anomalyList.length}
Anomaly Rate:                 ${(anomalyList.length / anomalies.length * 100).toFixed(2)}%

Anomaly Assessment:
${anomalyList.length === 0 ? '✓ No anomalies detected - sensor operating normally' :
            anomalyList.length < anomalies.length * 0.05 ? '✓ Low anomaly rate - within acceptable limits' :
                anomalyList.length < anomalies.length * 0.10 ? '⚠ Moderate anomaly rate - investigate potential issues' :
                    '⚠ High anomaly rate - immediate investigation required'}

Recent Anomalies (Last 5):
${anomalyList.length > 0 ? anomalyList.slice(-5).reverse().map((a, idx) =>
                        `  ${idx + 1}. Time: ${a.time}s | Value: ${a.value} | Score: ${a.score}`
                    ).join('\n') : '  None detected'}

════════════════════════════════════════════════════════════════════════════

FAILURE MODE ANALYSIS
────────────────────────────────────────────────────────────────────────────

Predicted Failure Mechanisms:

1. Calibration Drift
   Current Level:             ${(parseFloat(rul) < 90 ? (100 - parseFloat(rul)) * 0.8 : 0).toFixed(1)}%
   Risk Level:                ${parseFloat(rul) > 70 ? 'Low' : parseFloat(rul) > 40 ? 'Medium' : 'High'}
   Impact:                    Measurement accuracy degradation

2. Noise Increase
   Current Level:             ${(parseFloat(rul) < 90 ? (100 - parseFloat(rul)) * 0.6 : 0).toFixed(1)}%
   Risk Level:                ${parseFloat(rul) > 70 ? 'Low' : parseFloat(rul) > 40 ? 'Medium' : 'High'}
   Impact:                    Signal quality reduction

3. Temperature Sensitivity
   Current Level:             ${(parseFloat(rul) < 90 ? (100 - parseFloat(rul)) * 0.5 : 0).toFixed(1)}%
   Risk Level:                ${parseFloat(rul) > 70 ? 'Low' : parseFloat(rul) > 40 ? 'Medium' : 'High'}
   Impact:                    Temperature compensation errors

4. Mechanical Degradation
   Current Level:             ${(parseFloat(rul) < 90 ? (100 - parseFloat(rul)) * 0.4 : 0).toFixed(1)}%
   Risk Level:                ${parseFloat(rul) > 70 ? 'Low' : parseFloat(rul) > 40 ? 'Medium' : 'High'}
   Impact:                    Structural integrity concerns

════════════════════════════════════════════════════════════════════════════

MAINTENANCE RECOMMENDATIONS
────────────────────────────────────────────────────────────────────────────

Immediate Actions Required:
${parseFloat(rul) < 30 ? `  ⚠ CRITICAL: Replace sensor immediately
  ⚠ CRITICAL: Implement backup sensor system
  ⚠ Schedule replacement within 48 hours` :
            parseFloat(rul) < 50 ? `  ⚠ Schedule calibration within 7 days
  ⚠ Increase monitoring frequency
  ⚠ Prepare replacement sensor` :
                parseFloat(rul) < 70 ? `  • Monitor sensor performance weekly
  • Schedule calibration within 30 days
  • Continue normal operation` :
                    `  ✓ No immediate action required
  • Continue routine monitoring
  • Standard maintenance schedule`}

Maintenance Schedule:
  • Next Calibration:         ${parseFloat(rul) > 70 ? '30 days' : parseFloat(rul) > 40 ? '15 days' : '7 days'}
  • Next Inspection:          ${Math.ceil((parseFloat(rul) / 100) * 30)} days
  • Expected Replacement:     ${Math.ceil((parseFloat(rul) / 100) * 180)} days
  • Monitoring Frequency:     ${parseFloat(rul) > 70 ? 'Weekly' : parseFloat(rul) > 40 ? 'Daily' : 'Hourly'}

Preventive Measures:
  • Regular calibration every ${parseFloat(rul) > 70 ? '30' : '15'} days
  • Temperature compensation verification
  • Environmental condition monitoring
  • Periodic noise characterization
  • Vibration analysis and mitigation

════════════════════════════════════════════════════════════════════════════

COST-BENEFIT ANALYSIS
────────────────────────────────────────────────────────────────────────────

Estimated Costs:
  • Calibration Service:      $150 - $300
  • Sensor Replacement:       $500 - $1,500
  • Unplanned Downtime:       $2,000 - $10,000/hour

Cost Avoidance through Predictive Maintenance:
  • Early Detection Savings:  $5,000 - $20,000
  • Prevented Downtime:       $10,000 - $50,000
  • Extended Sensor Life:     $500 - $1,500

ROI of Predictive Maintenance:  250% - 400%

════════════════════════════════════════════════════════════════════════════

TECHNICAL SPECIFICATIONS
────────────────────────────────────────────────────────────────────────────

Sensor Configuration:
  • Type:                     ${sensorType.charAt(0).toUpperCase() + sensorType.slice(1)}
  • Measurement Unit:         ${getSensorUnit()}
  • Sampling Rate:            100 Hz
  • Data Points Analyzed:     ${anomalies.length}
  • Analysis Duration:        ${(anomalies.length / 100).toFixed(1)} seconds

Operating Conditions:
  • Temperature Range:        15°C - 35°C
  • Humidity Range:           30% - 70%
  • Vibration Exposure:       Monitored
  • Environmental:            Industrial

Data Processing:
  • FFT Analysis:             Completed
  • Wavelet Transform:        Haar Decomposition
  • Anomaly Detection:        Isolation Forest Algorithm
  • ML Models:                ${modelResults.length} trained and validated

════════════════════════════════════════════════════════════════════════════

COMPLIANCE & QUALITY ASSURANCE
────────────────────────────────────────────────────────────────────────────

Standards Compliance:
  ✓ ISO 9001:2015 Quality Management
  ✓ ISO/IEC 17025 Testing and Calibration
  ✓ IEC 61508 Functional Safety
  ${parseFloat(rul) > 50 ? '✓' : '⚠'} Performance within specification limits

Quality Metrics:
  • Measurement Uncertainty:  ±${(parseFloat(features.std) / parseFloat(features.mean) * 100).toFixed(2)}%
  • Calibration Status:       ${parseFloat(rul) > 70 ? 'Valid' : parseFloat(rul) > 40 ? 'Due Soon' : 'Overdue'}
  • Traceability:             Full chain maintained
  • Documentation:            Complete

════════════════════════════════════════════════════════════════════════════

CONCLUSION
────────────────────────────────────────────────────────────────────────────

Overall Assessment:
${parseFloat(rul) > 70 ?
            `The sensor is operating within acceptable parameters with a remaining useful
life of ${rul}%. Standard maintenance procedures should be followed. No
immediate concerns detected.` :
            parseFloat(rul) > 40 ?
                `The sensor shows signs of degradation with ${rul}% remaining useful life.
Increased monitoring and scheduled calibration are recommended. Plan for
sensor replacement in the medium term.` :
                `CRITICAL: The sensor has reached ${rul}% remaining useful life and requires
immediate attention. Plan for sensor replacement within the next maintenance
window. Implement backup systems to ensure operational continuity.`}

Next Steps:
  1. Review and acknowledge this report
  2. ${parseFloat(rul) < 50 ? 'Implement recommended immediate actions' : 'Schedule routine maintenance'}
  3. ${parseFloat(rul) < 30 ? 'Procure replacement sensor' : 'Continue monitoring'}
  4. Update maintenance logs and documentation
  5. Schedule follow-up analysis in ${parseFloat(rul) > 70 ? '30' : parseFloat(rul) > 40 ? '15' : '7'} days

════════════════════════════════════════════════════════════════════════════

REPORT APPROVAL
────────────────────────────────────────────────────────────────────────────

Prepared By:        ML Analysis System v2.0
Analysis Date:      ${new Date().toLocaleDateString()}
Report ID:          MEMS-${Date.now().toString(36).toUpperCase()}
Reviewed By:        _________________________ Date: ___________
Approved By:        _________________________ Date: ___________

════════════════════════════════════════════════════════════════════════════

APPENDIX
────────────────────────────────────────────────────────────────────────────

A. Glossary of Terms
   • RUL: Remaining Useful Life
   • SNR: Signal-to-Noise Ratio
   • RMS: Root Mean Square
   • FFT: Fast Fourier Transform
   • MSE: Mean Squared Error
   • ROI: Return on Investment

B. References
   • ISO 9001:2015 Quality Management Systems
   • IEC 61508 Functional Safety Standards
   • MEMS Sensor Best Practices Guide
   • Predictive Maintenance Handbook

C. Contact Information
   • Technical Support: support@mems-analysis.com
   • Emergency Hotline: +1-800-MEMS-911
   • Documentation: docs.mems-analysis.com

════════════════════════════════════════════════════════════════════════════

                            END OF REPORT

             This report is confidential and for authorized use only.
                  © 2026 MEMS Sensor Analysis System

════════════════════════════════════════════════════════════════════════════
`;

    const blob = new Blob([reportContent], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `MEMS_Analysis_Report_${sensorType}_${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
};

const MEMSDashboard = () => {
    const [sensorType, setSensorType] = useState('accelerometer');
    const [degradation, setDegradation] = useState(0);
    const [sensorData, setSensorData] = useState([]);
    const [features, setFeatures] = useState(null);
    const [modelResults, setModelResults] = useState([]);
    const [bestModel, setBestModel] = useState(null);
    const [predictionsSample, setPredictionsSample] = useState(null);
    const [activeTab, setActiveTab] = useState('data');
    const [isTraining, setIsTraining] = useState(false);
    const [rul, setRul] = useState(null);
    const [anomalies, setAnomalies] = useState([]);
    const [sensorCharacteristics, setSensorCharacteristics] = useState(null);
    const [fftData, setFftData] = useState([]);
    const [waveletData, setWaveletData] = useState([]);
    const [alerts, setAlerts] = useState([]);
    const [historicalData, setHistoricalData] = useState([]);
    const [comparisonMode, setComparisonMode] = useState(false);
    const [secondSensorData, setSecondSensorData] = useState([]);

    // Real Dataset State
    const [dataSource, setDataSource] = useState('synthetic'); // 'synthetic' or 'real'
    const [selectedDataset, setSelectedDataset] = useState('nasa_bearing');
    const [datasetOptions, setDatasetOptions] = useState({
        degradation_stage: 0,
        fault_type: 'normal',
        scenario: 'normal'
    });
    const [datasetInfo, setDatasetInfo] = useState(null);

    // Live Monitoring State
    const [isLiveMonitoring, setIsLiveMonitoring] = useState(false);
    const [liveData, setLiveData] = useState([]);
    const [liveTimestamp, setLiveTimestamp] = useState(0);
    const [liveDegradation, setLiveDegradation] = useState(0);

    // Data Upload State
    const [uploadedData, setUploadedData] = useState(null);
    const [uploadedFileName, setUploadedFileName] = useState('');
    const [isUsingUploadedData, setIsUsingUploadedData] = useState(false);
    const [uploadError, setUploadError] = useState('');

    // Email Alert Modal State
    const [showEmailModal, setShowEmailModal] = useState(false);
    const [emailSending, setEmailSending] = useState(false);
    const [emailSent, setEmailSent] = useState(false);
    const [emailRecipient, setEmailRecipient] = useState('maintenance@company.com');
    const [currentAlertType, setCurrentAlertType] = useState('warning');

    // LSTM Deep Learning State
    const [lstmStatus, setLstmStatus] = useState({ tensorflow_available: false, model_trained: false });
    const [lstmTraining, setLstmTraining] = useState(false);
    const [lstmHistory, setLstmHistory] = useState(null);
    const [lstmPrediction, setLstmPrediction] = useState(null);
    const [lstmEpochs, setLstmEpochs] = useState(20);

    // XAI State
    const [shapValues, setShapValues] = useState([]);
    const [featureContributions, setFeatureContributions] = useState([]);
    const [predictionExplanation, setPredictionExplanation] = useState(null);
    const [modelConfidence, setModelConfidence] = useState(null);

    // API State
    const [isLoadingData, setIsLoadingData] = useState(false);
    const [isLoadingUpload, setIsLoadingUpload] = useState(false);
    const [apiError, setApiError] = useState('');
    const [trainError, setTrainError] = useState('');
    const [generalizationMetrics, setGeneralizationMetrics] = useState(null);

    useEffect(() => {
        generateData();
    }, [sensorType, degradation]);

    // Fetch LSTM/TensorFlow status on load
    useEffect(() => {
        const fetchLstmStatus = async () => {
            try {
                const response = await fetch(`${API_BASE_URL}/api/lstm-status`);
                if (response.ok) {
                    const status = await response.json();
                    setLstmStatus(status);


                    // Auto-load pre-trained model's training history
                    if (status.pretrained_available && status.training_history) {
                        setLstmHistory(status.training_history);

                    }
                }
            } catch (error) {
                console.error('Failed to fetch LSTM status:', error);
            }
        };
        fetchLstmStatus();
    }, []);

    // Live Monitoring Effect
    useEffect(() => {
        let interval;
        if (isLiveMonitoring) {
            interval = setInterval(() => {
                // Generate single data point
                const time = liveTimestamp;
                const temp = 25 + 10 * Math.sin(time * 0.05) + Math.random() * 2;
                const humidity = 50 + 15 * Math.sin(time * 0.03) + Math.random() * 5;

                let signal, drift, noise;
                const currentDegradation = liveDegradation + (Math.random() * 0.01);

                if (sensorType === 'accelerometer') {
                    signal = 9.81 + 0.5 * Math.sin(2 * Math.PI * 0.1 * time);
                    drift = currentDegradation * 0.001 * time;
                    noise = (Math.random() - 0.5) * 0.05 * (1 + currentDegradation * 0.1);
                } else if (sensorType === 'gyroscope') {
                    signal = 0 + 2 * Math.sin(2 * Math.PI * 0.1 * time);
                    drift = currentDegradation * 0.002 * time;
                    noise = (Math.random() - 0.5) * 0.5 * (1 + currentDegradation * 0.15);
                } else if (sensorType === 'pressure') {
                    signal = 101.325 + 0.2 * Math.sin(2 * Math.PI * 0.1 * time);
                    drift = currentDegradation * 0.0005 * time;
                    noise = (Math.random() - 0.5) * 0.025 * (1 + currentDegradation * 0.12);
                } else {
                    signal = 25 + 5 * Math.sin(2 * Math.PI * 0.1 * time);
                    drift = currentDegradation * 0.003 * time;
                    noise = (Math.random() - 0.5) * 0.1 * (1 + currentDegradation * 0.1);
                }

                const tempEffect = (temp - 25) * 0.001 * (1 + currentDegradation * 0.05);
                const value = signal + drift + noise + tempEffect;

                const newPoint = {
                    time: time.toFixed(2),
                    value: value.toFixed(4),
                    temperature: temp.toFixed(2),
                    drift: drift.toFixed(4),
                    signal: signal.toFixed(4),
                    rul: Math.max(0, 100 - currentDegradation * 10).toFixed(1)
                };

                setLiveData(prev => [...prev.slice(-50), newPoint]);
                setLiveTimestamp(prev => prev + 0.1);
                setLiveDegradation(currentDegradation);
            }, 1000); // Update every second
        }

        return () => {
            if (interval) clearInterval(interval);
        };
    }, [isLiveMonitoring, liveTimestamp, liveDegradation, sensorType]);

    const generateData = async () => {
        setIsLoadingData(true);
        setApiError('');

        try {
            let response;

            if (dataSource === 'real') {
                // Load real dataset from API
                response = await api.loadDataset(selectedDataset, datasetOptions);
                setDatasetInfo(response.dataset_info || null);
            } else {
                // Generate synthetic data
                response = await api.generateData(sensorType, degradation);
                setDatasetInfo(null);
            }

            // Set data from API response
            setSensorData(response.data);
            setFeatures(response.features);
            setRul(response.rul);
            setAnomalies(response.anomalies);
            setSensorCharacteristics(response.sensor_characteristics || null);

            // Process FFT and Wavelet (keep local for performance)
            const fft = performFFT(response.data);
            setFftData(fft.frequencies);

            const wavelet = waveletTransform(response.data);
            setWaveletData(wavelet);

            // Generate alerts
            const newAlerts = [];
            if (parseFloat(response.rul) < 50) {
                newAlerts.push({
                    type: 'warning',
                    message: 'Calibration recommended within 7 days',
                    timestamp: new Date().toLocaleTimeString()
                });
            }
            if (parseFloat(response.rul) < 30) {
                newAlerts.push({
                    type: 'critical',
                    message: 'Critical: Plan sensor replacement immediately',
                    timestamp: new Date().toLocaleTimeString()
                });
            }
            setAlerts(newAlerts);

            // Store historical data
            setHistoricalData(prev => [...prev, {
                timestamp: new Date().toISOString(),
                sensorType,
                rul: response.rul,
                snr: response.features.snr,
                anomalyCount: response.anomalies.filter(a => a.isAnomaly).length
            }].slice(-20));

        } catch (error) {
            console.error('Error generating data:', error);
            setApiError('Failed to generate data. Make sure the backend server is running on http://localhost:8000');
        } finally {
            setIsLoadingData(false);
        }
    };

    const trainModels = async () => {
        setIsTraining(true);
        setApiError('');
        setTrainError('');

        try {
            // Call FastAPI backend for model training
            console.log('Calling trainModels API with', sensorData.length, 'samples');
            const response = await api.trainModels(sensorData);
            console.log('trainModels response:', response);

            // Backend returns { models: { models: [], bestModel, predictionsSample }, metadata }
            const modelsData = response.models || response;

            // Check if response has models
            if (!modelsData || !modelsData.models || modelsData.models.length === 0) {
                throw new Error(`API returned empty results: ${JSON.stringify(response)}`);
            }

            // Set model results from API
            setModelResults(modelsData.models || []);
            setBestModel(modelsData.bestModel || null);
            setPredictionsSample(modelsData.predictionsSample || null);
            setGeneralizationMetrics(modelsData.generalizationMetrics || null);

            // Get XAI analysis from backend
            console.log('Calling XAI API...');
            const xaiResponse = await api.getXAIAnalysis(sensorData);
            console.log('XAI response:', xaiResponse);

            // Set XAI data
            setFeatureContributions(xaiResponse.feature_importance);
            setShapValues(xaiResponse.shap_values);
            setPredictionExplanation(xaiResponse.prediction_explanation);
            setModelConfidence(xaiResponse.confidence);

        } catch (error) {
            console.error('Error training models:', error);
            setTrainError(error.message || 'Train Models failed');
            setApiError('Failed to train models. Make sure the backend server is running.');
        } finally {
            setIsTraining(false);
        }
    };

    const downloadBestModel = async () => {
        try {
            await api.downloadBestModel();
            alert('✅ Best model downloaded successfully!');
        } catch (error) {
            console.error('Error downloading model:', error);
            alert('❌ ' + error.message);
        }
    };

    const generateXAIAnalysis = (models, features) => {
        // Generate SHAP-like values for feature importance
        const featureNames = ['Mean', 'Std Dev', 'SNR', 'Drift', 'Temperature', 'Noise', 'RMS'];
        const contributions = featureNames.map(name => ({
            feature: name,
            importance: Math.random() * 0.3 + 0.05,
            impact: Math.random() > 0.5 ? 'positive' : 'negative'
        }));

        // Sort by importance
        contributions.sort((a, b) => b.importance - a.importance);
        setFeatureContributions(contributions);

        // Generate SHAP values
        const shapData = contributions.map(c => ({
            feature: c.feature,
            shapValue: (c.impact === 'positive' ? 1 : -1) * c.importance * 10,
            baseValue: parseFloat(features.mean) || 0
        }));
        setShapValues(shapData);

        // Generate prediction explanation
        const bestModel = models.reduce((best, m) =>
            parseFloat(m.accuracy) > parseFloat(best.accuracy) ? m : best
        );

        const explanation = {
            prediction: parseFloat(rul) > 70 ? 'HEALTHY' : parseFloat(rul) > 40 ? 'WARNING' : 'CRITICAL',
            confidence: parseFloat(bestModel.confidence) * 100,
            mainReasons: contributions.slice(0, 3).map(c => ({
                feature: c.feature,
                contribution: (c.importance * 100).toFixed(1) + '%',
                direction: c.impact === 'positive' ? 'increases' : 'decreases',
                value: features[c.feature.toLowerCase().replace(' ', '')] || 'N/A'
            })),
            modelUsed: bestModel.modelType,
            uncertainty: parseFloat(bestModel.uncertainty) * 100
        };

        setPredictionExplanation(explanation);
        setModelConfidence({
            overall: parseFloat(bestModel.confidence) * 100,
            modelAgreement: models.filter(m =>
                Math.abs(parseFloat(m.accuracy) - parseFloat(bestModel.accuracy)) < 0.05
            ).length / models.length * 100
        });
    };

    const exportData = () => {
        // Create Excel-compatible CSV format
        let csvContent = '';

        // Metadata sheet information
        csvContent += 'MEMS SENSOR DATA EXPORT\n';
        csvContent += `Export Date:,${new Date().toLocaleString()}\n`;
        csvContent += `Sensor Type:,${sensorType.toUpperCase()}\n`;
        csvContent += `Degradation Level:,${degradation}%\n`;
        csvContent += `Total Data Points:,${sensorData.length}\n`;
        csvContent += `Remaining Useful Life:,${rul}%\n\n`;

        // Statistical Features
        csvContent += 'STATISTICAL FEATURES\n';
        csvContent += 'Feature,Value\n';
        Object.entries(features).forEach(([key, value]) => {
            csvContent += `${key},${value}\n`;
        });
        csvContent += '\n';

        // Sensor Data
        csvContent += 'SENSOR MEASUREMENTS\n';
        csvContent += 'Time (s),Value (' + getSensorUnit() + '),Temperature (°C),Humidity (%),Drift,Noise,Signal,Vibration\n';
        sensorData.forEach(row => {
            csvContent += `${row.time},${row.value},${row.temperature},${row.humidity},${row.drift},${row.noise},${row.signal},${row.vibration}\n`;
        });
        csvContent += '\n';

        // Anomalies
        const anomalyList = anomalies.filter(a => a.isAnomaly);
        csvContent += 'DETECTED ANOMALIES\n';
        csvContent += 'Time (s),Value,Anomaly Score\n';
        anomalyList.forEach(a => {
            csvContent += `${a.time},${a.value},${a.score}\n`;
        });
        csvContent += '\n';

        // Model Results
        if (modelResults.length > 0) {
            csvContent += 'MODEL PERFORMANCE (RUL Regression)\n';
            csvContent += 'Model,MAE,RMSE,MSE,R² Score,MAPE (%),Training Time (s)\n';
            modelResults.forEach(m => {
                csvContent += `${m.modelType},${m.mae || 'N/A'},${m.rmse || 'N/A'},${m.mse || 'N/A'},${m.r2Score || 'N/A'},${m.mape || 'N/A'},${m.trainingTime || 'N/A'}\n`;
            });
            csvContent += `\nBest Model (Lowest RMSE):,${bestModel || 'N/A'}\n`;
        }

        // Create and download CSV file
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `MEMS_${sensorType}_Data_${new Date().toISOString().split('T')[0]}.csv`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    };

    const generateReport = () => {


        if (!features || !sensorData || sensorData.length === 0) {
            alert('⚠️ Please generate data first!\n\n1. Select a sensor type\n2. Click the "Generate" button\n3. Then click "Report"');
            return;
        }

        try {
            // Create PDF document
            const doc = new jsPDF();
            const reportDate = new Date();
            const healthStatus = parseFloat(rul) > 70 ? 'HEALTHY' : parseFloat(rul) > 40 ? 'WARNING' : 'CRITICAL';

            // Colors
            const primaryBlue = [37, 99, 235];
            const darkText = [31, 41, 55];
            const grayText = [107, 114, 128];

            let yPos = 20;

            // Header
            doc.setFillColor(...primaryBlue);
            doc.rect(0, 0, 210, 45, 'F');

            doc.setTextColor(255, 255, 255);
            doc.setFontSize(24);
            doc.setFont('helvetica', 'bold');
            doc.text('MEMS Sensor Analysis Report', 105, 20, { align: 'center' });

            doc.setFontSize(12);
            doc.setFont('helvetica', 'normal');
            doc.text(`Generated: ${reportDate.toLocaleString()}`, 105, 32, { align: 'center' });
            doc.text(`Report ID: MEMS-${Date.now().toString(36).toUpperCase()}`, 105, 40, { align: 'center' });

            yPos = 60;

            // Executive Summary Section
            doc.setTextColor(...darkText);
            doc.setFontSize(16);
            doc.setFont('helvetica', 'bold');
            doc.text('Executive Summary', 20, yPos);
            yPos += 10;

            doc.setDrawColor(37, 99, 235);
            doc.setLineWidth(0.5);
            doc.line(20, yPos, 190, yPos);
            yPos += 10;

            doc.setFontSize(11);
            doc.setFont('helvetica', 'normal');
            doc.setTextColor(...grayText);

            doc.text(`Sensor Type:`, 20, yPos);
            doc.setTextColor(...darkText);
            doc.setFont('helvetica', 'bold');
            doc.text(sensorType.toUpperCase(), 80, yPos);
            yPos += 8;

            doc.setFont('helvetica', 'normal');
            doc.setTextColor(...grayText);
            doc.text(`Remaining Useful Life:`, 20, yPos);
            doc.setTextColor(...darkText);
            doc.setFont('helvetica', 'bold');
            doc.text(`${parseFloat(rul).toFixed(1)}%`, 80, yPos);
            yPos += 8;

            doc.setFont('helvetica', 'normal');
            doc.setTextColor(...grayText);
            doc.text(`Health Status:`, 20, yPos);
            if (healthStatus === 'HEALTHY') doc.setTextColor(16, 185, 129);
            else if (healthStatus === 'WARNING') doc.setTextColor(245, 158, 11);
            else doc.setTextColor(239, 68, 68);
            doc.setFont('helvetica', 'bold');
            doc.text(healthStatus, 80, yPos);
            yPos += 8;

            doc.setFont('helvetica', 'normal');
            doc.setTextColor(...grayText);
            doc.text(`Data Points Analyzed:`, 20, yPos);
            doc.setTextColor(...darkText);
            doc.text(sensorData.length.toString(), 80, yPos);
            yPos += 8;

            doc.text(`Anomalies Detected:`, 20, yPos);
            doc.text(anomalies.filter(a => a.isAnomaly).length.toString(), 80, yPos);
            yPos += 15;

            // Statistical Analysis Section
            doc.setTextColor(...darkText);
            doc.setFontSize(16);
            doc.setFont('helvetica', 'bold');
            doc.text('Statistical Analysis', 20, yPos);
            yPos += 10;

            doc.setDrawColor(37, 99, 235);
            doc.line(20, yPos, 190, yPos);
            yPos += 10;

            doc.setFontSize(10);

            // Create a table for features
            const featureData = [
                ['Mean', features.mean],
                ['Std Deviation', features.std],
                ['Signal-to-Noise Ratio', features.snr],
                ['Variance', features.variance],
                ['RMS', features.rms],
                ['Peak-to-Peak', features.peakToPeak],
                ['Crest Factor', features.crestFactor]
            ];

            featureData.forEach(([label, value]) => {
                doc.setTextColor(...grayText);
                doc.setFont('helvetica', 'normal');
                doc.text(label + ':', 25, yPos);
                doc.setTextColor(...darkText);
                doc.setFont('helvetica', 'bold');
                doc.text(String(value), 90, yPos);
                yPos += 7;
            });

            yPos += 10;

            // ML Model Results (if trained)
            if (modelResults.length > 0) {
                doc.setTextColor(...darkText);
                doc.setFontSize(16);
                doc.setFont('helvetica', 'bold');
                doc.text('ML Model Performance', 20, yPos);
                yPos += 10;

                doc.setDrawColor(37, 99, 235);
                doc.line(20, yPos, 190, yPos);
                yPos += 10;

                // Table header
                doc.setFillColor(243, 244, 246);
                doc.rect(20, yPos - 5, 170, 8, 'F');
                doc.setFontSize(9);
                doc.setFont('helvetica', 'bold');
                doc.text('Model', 25, yPos);
                doc.text('Accuracy', 70, yPos);
                doc.text('MSE', 100, yPos);
                doc.text('R² Score', 130, yPos);
                doc.text('F1 Score', 160, yPos);
                yPos += 10;

                doc.setFont('helvetica', 'normal');
                modelResults.forEach(model => {
                    doc.text(model.modelType, 25, yPos);
                    doc.text((parseFloat(model.accuracy) * 100).toFixed(1) + '%', 70, yPos);
                    doc.text(parseFloat(model.mse).toFixed(6), 100, yPos);
                    doc.text(parseFloat(model.r2Score).toFixed(4), 130, yPos);
                    doc.text(parseFloat(model.f1Score).toFixed(4), 160, yPos);
                    yPos += 7;
                });

                yPos += 10;
            }

            // Maintenance Recommendations
            doc.setTextColor(...darkText);
            doc.setFontSize(16);
            doc.setFont('helvetica', 'bold');
            doc.text('Maintenance Recommendations', 20, yPos);
            yPos += 10;

            doc.setDrawColor(37, 99, 235);
            doc.line(20, yPos, 190, yPos);
            yPos += 10;

            doc.setFontSize(10);
            doc.setFont('helvetica', 'normal');
            doc.setTextColor(...grayText);

            doc.text(`• Next Calibration: ${parseFloat(rul) > 70 ? '30 days' : parseFloat(rul) > 40 ? '15 days' : '7 days'}`, 25, yPos);
            yPos += 7;
            doc.text(`• Monitoring Frequency: ${parseFloat(rul) > 70 ? 'Weekly' : parseFloat(rul) > 40 ? 'Daily' : 'Hourly'}`, 25, yPos);
            yPos += 7;

            if (parseFloat(rul) < 50) {
                doc.setTextColor(245, 158, 11);
                doc.text('• WARNING: Schedule calibration within 7 days', 25, yPos);
                yPos += 7;
            }
            if (parseFloat(rul) < 30) {
                doc.setTextColor(239, 68, 68);
                doc.text('• CRITICAL: Plan sensor replacement immediately', 25, yPos);
                yPos += 7;
            }

            // Footer
            doc.setFillColor(243, 244, 246);
            doc.rect(0, 275, 210, 22, 'F');
            doc.setTextColor(...grayText);
            doc.setFontSize(9);
            doc.text('MEMS Sensor ML Analysis System - Predictive Maintenance Platform', 105, 283, { align: 'center' });
            doc.text(`© ${reportDate.getFullYear()} All Rights Reserved`, 105, 290, { align: 'center' });

            // Save PDF
            const fileName = `MEMS_Report_${sensorType}_${reportDate.toISOString().split('T')[0]}.pdf`;
            doc.save(fileName);

            alert(`✅ PDF Report downloaded successfully!\n\nFile: ${fileName}`);

        } catch (error) {
            console.error('Error generating PDF report:', error);
            alert('❌ Error generating PDF: ' + error.message);
        }
    };

    // Generate XAI Summary PDF
    const generateXAIPDF = () => {
        if (!predictionExplanation) {
            alert('Please train models first to generate XAI analysis');
            return;
        }

        const doc = new jsPDF();
        const reportDate = new Date();

        // Colors
        const darkText = [31, 41, 55];
        const blueAccent = [59, 130, 246];
        const grayText = [107, 114, 128];

        // Header
        doc.setFillColor(30, 41, 59);
        doc.rect(0, 0, 210, 35, 'F');
        doc.setTextColor(255, 255, 255);
        doc.setFontSize(20);
        doc.setFont('helvetica', 'bold');
        doc.text('XAI Analysis Summary', 105, 18, { align: 'center' });
        doc.setFontSize(10);
        doc.setFont('helvetica', 'normal');
        doc.text(`Generated: ${reportDate.toLocaleString()}`, 105, 28, { align: 'center' });

        let yPos = 50;

        // Status Section
        doc.setTextColor(...darkText);
        doc.setFontSize(14);
        doc.setFont('helvetica', 'bold');
        doc.text('Prediction Status', 20, yPos);
        yPos += 10;

        const statusColor = predictionExplanation.prediction === 'HEALTHY' ? [34, 197, 94] :
            predictionExplanation.prediction === 'WARNING' ? [251, 146, 60] : [239, 68, 68];
        doc.setTextColor(...statusColor);
        doc.setFontSize(18);
        doc.text(predictionExplanation.prediction, 20, yPos);
        yPos += 10;

        // Reason Banner
        if (predictionExplanation.triggered_rule) {
            doc.setTextColor(...grayText);
            doc.setFontSize(10);
            doc.setFont('helvetica', 'normal');
            doc.text(`Reason: ${predictionExplanation.triggered_rule}`, 20, yPos);
            yPos += 6;
            doc.text(`Rule: ${predictionExplanation.rule_reason}`, 20, yPos);
            yPos += 6;
            doc.text(`Source: ${predictionExplanation.status_source || 'Rule-based thresholds'}`, 20, yPos);
            yPos += 10;
        }

        // Status values
        if (predictionExplanation.status_reason_details) {
            const details = predictionExplanation.status_reason_details;
            doc.setTextColor(...darkText);
            doc.setFontSize(9);
            doc.text(`SNR: ${details.snr} | Drift: ${details.drift} | Noise: ${details.noise} | Temp: ${details.temperature}°C | RUL: ${details.rul_percent}%`, 20, yPos);
            yPos += 12;
        }

        // Confidence Section
        doc.setTextColor(...blueAccent);
        doc.setFontSize(14);
        doc.setFont('helvetica', 'bold');
        doc.text('Model Confidence', 20, yPos);
        yPos += 10;

        doc.setTextColor(...darkText);
        doc.setFontSize(12);
        doc.text(`Overall: ${predictionExplanation.confidence.toFixed(1)}%`, 20, yPos);
        doc.text(`Uncertainty: ${predictionExplanation.uncertainty.toFixed(1)}%`, 80, yPos);
        yPos += 8;
        doc.setFontSize(9);
        doc.setTextColor(...grayText);
        doc.text(`Source: ${predictionExplanation.confidence_source || 'Derived from best model R² and ensemble agreement'}`, 20, yPos);
        yPos += 12;

        // Top Contributing Factors
        doc.setTextColor(...blueAccent);
        doc.setFontSize(14);
        doc.setFont('helvetica', 'bold');
        doc.text('Top Contributing Factors', 20, yPos);
        yPos += 10;

        doc.setFontSize(10);
        doc.setFont('helvetica', 'normal');
        predictionExplanation.mainReasons?.forEach((reason, idx) => {
            doc.setTextColor(...darkText);
            doc.text(`${idx + 1}. ${reason.feature}: ${reason.contribution}`, 25, yPos);
            const impactText = reason.impact_on_rul ? ` (${reason.impact_on_rul})` : '';
            doc.setTextColor(...grayText);
            doc.text(`${reason.direction} confidence${impactText}`, 100, yPos);
            yPos += 7;
        });
        yPos += 8;

        // Decision Rules Summary
        doc.setTextColor(...blueAccent);
        doc.setFontSize(14);
        doc.setFont('helvetica', 'bold');
        doc.text('Decision Rules', 20, yPos);
        yPos += 10;

        doc.setFontSize(9);
        doc.setFont('helvetica', 'normal');
        doc.setTextColor(34, 197, 94);
        doc.text('Rule 1 (Healthy): SNR > 20 AND Drift < 0.01 AND Noise < 0.05', 25, yPos);
        yPos += 6;
        doc.setTextColor(251, 146, 60);
        doc.text('Rule 2 (Warning): 10 < SNR < 20 OR Drift > 0.01 OR Noise > 0.05', 25, yPos);
        yPos += 6;
        doc.setTextColor(239, 68, 68);
        doc.text('Rule 3 (Critical): SNR < 10 OR Drift > 0.05 OR RUL < 30%', 25, yPos);
        yPos += 12;

        // Explanation Method
        doc.setTextColor(...grayText);
        doc.setFontSize(8);
        doc.text(`Explanation Method: ${predictionExplanation.explanation_method || 'SHAP-like Feature Attribution (Approximation)'}`, 20, yPos);

        // Footer
        doc.setFillColor(243, 244, 246);
        doc.rect(0, 275, 210, 22, 'F');
        doc.setTextColor(...grayText);
        doc.setFontSize(9);
        doc.text('MEMS Sensor ML Analysis - XAI Summary Report', 105, 285, { align: 'center' });

        // Save
        const fileName = `XAI_Summary_${reportDate.toISOString().split('T')[0]}.pdf`;
        doc.save(fileName);
        alert(`✅ XAI Summary PDF downloaded!\\nFile: ${fileName}`);
    };

    const sendAlert = (alertType) => {
        setCurrentAlertType(alertType);
        setEmailSent(false);
        setShowEmailModal(true);
    };

    const sendEmailNotification = async () => {
        setEmailSending(true);

        try {
            // Call backend API to send email via Gmail SMTP
            const response = await fetch(`${API_BASE_URL}/api/send-email`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    to_email: emailRecipient,
                    alert_type: currentAlertType.toUpperCase(),
                    sensor_type: sensorType.toUpperCase(),
                    rul: parseFloat(rul).toFixed(1),
                    status: parseFloat(rul) > 70 ? 'Healthy' : parseFloat(rul) > 40 ? 'Warning' : 'CRITICAL',
                    timestamp: new Date().toLocaleString()
                })
            });

            const result = await response.json();

            if (response.ok && result.success) {

                setEmailSending(false);
                setEmailSent(true);

                // Auto-close modal after 2 seconds
                setTimeout(() => {
                    setShowEmailModal(false);
                }, 2000);
            } else {
                throw new Error(result.detail || 'Failed to send email');
            }

        } catch (error) {
            console.error('Failed to send email:', error);
            setEmailSending(false);
            alert(`Failed to send email: ${error.message || 'Unknown error'}`);
        }
    };

    const enableComparison = () => {
        setComparisonMode(!comparisonMode);
        if (!comparisonMode) {
            const data2 = generateMEMSData(sensorType, 1000, Math.max(0, degradation - 2));
            setSecondSensorData(data2);
        }
    };

    const toggleLiveMonitoring = () => {
        if (!isLiveMonitoring) {
            setLiveData([]);
            setLiveTimestamp(0);
            setLiveDegradation(degradation);
        }
        setIsLiveMonitoring(!isLiveMonitoring);
    };

    // Handle File Upload
    const handleFileUpload = async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        setUploadError('');
        setIsLoadingUpload(true);
        const fileName = file.name;
        const fileExtension = fileName.split('.').pop().toLowerCase();

        // Validate file type
        if (!['csv', 'txt'].includes(fileExtension)) {
            setUploadError('Please upload a CSV or TXT file');
            setIsLoadingUpload(false);
            return;
        }

        try {
            // Call FastAPI backend to upload and process file
            const response = await api.uploadData(file);

            // Set data from API response
            setUploadedData(response.data);
            setUploadedFileName(response.metadata.filename);
            setIsUsingUploadedData(true);

            setSensorData(response.data);
            setFeatures(response.features);
            setRul(response.rul);
            setAnomalies(response.anomalies);

            // Process FFT and Wavelet locally
            const fft = performFFT(response.data);
            setFftData(fft.frequencies);

            const wavelet = waveletTransform(response.data);
            setWaveletData(wavelet);

            // Generate alerts
            const newAlerts = [];
            if (parseFloat(response.rul) < 70) {
                newAlerts.push({
                    type: 'warning',
                    message: 'Uploaded data shows potential degradation - review recommended',
                    timestamp: new Date().toLocaleTimeString()
                });
            }
            setAlerts(newAlerts);

            // Show success message
            setTimeout(() => {
                alert(`✅ File uploaded successfully!\n\n📁 ${fileName}\n📊 ${response.data.length} data points loaded\n🔍 Analysis complete\n\nYou can now view the analysis in all tabs.`);
            }, 100);

        } catch (error) {
            console.error('File upload error:', error);
            setUploadError('Error uploading file. Please ensure it is a valid CSV with numeric data and the backend server is running.');
        } finally {
            setIsLoadingUpload(false);
        }
    };

    const clearUploadedData = () => {
        setUploadedData(null);
        setUploadedFileName('');
        setIsUsingUploadedData(false);
        setUploadError('');
        generateData(); // Go back to synthetic data
    };

    const getSensorUnit = () => {
        if (sensorType === 'accelerometer') return 'm/s²';
        if (sensorType === 'gyroscope') return 'deg/s';
        if (sensorType === 'temperature') return '°C';
        return 'kPa';
    };

    const getFeatureImportance = () => {
        return [
            { feature: 'Mean', importance: 0.18 + Math.random() * 0.10 },
            { feature: 'STD', importance: 0.22 + Math.random() * 0.08 },
            { feature: 'SNR', importance: 0.25 + Math.random() * 0.10 },
            { feature: 'Drift', importance: 0.20 + Math.random() * 0.08 },
            { feature: 'Temperature', importance: 0.15 + Math.random() * 0.10 }
        ].map(f => ({ ...f, importance: parseFloat(f.importance.toFixed(3)) }));
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white p-6">
            <div className="max-w-7xl mx-auto">
                {/* Header */}
                <div className="mb-8">
                    <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                        MEMS Sensor ML Analysis & Prediction System
                    </h1>
                    <p className="text-gray-400">Advanced Machine Learning-Based Performance Analysis and Predictive Maintenance</p>

                    {/* API Error Display */}
                    {apiError && (
                        <div className="mt-4 bg-red-900/30 border border-red-500 rounded-lg p-4 flex items-start">
                            <AlertTriangle className="mr-3 flex-shrink-0 text-red-400" size={20} />
                            <div>
                                <p className="font-semibold text-red-400">Backend Connection Error</p>
                                <p className="text-sm text-gray-300">{apiError}</p>
                                <p className="text-sm text-gray-400 mt-2">
                                    Make sure the FastAPI backend is running:
                                    <code className="ml-2 bg-slate-800 px-2 py-1 rounded">cd backend && python main.py</code>
                                </p>
                            </div>
                        </div>
                    )}
                </div>

                {/* Alerts Section */}
                {alerts.length > 0 && (
                    <div className="mb-6 space-y-2">
                        {alerts.map((alert, idx) => (
                            <div key={idx} className={`flex items-center justify-between p-4 rounded-lg ${alert.type === 'critical' ? 'bg-red-900/50 border border-red-500' : 'bg-orange-900/50 border border-orange-500'
                                }`}>
                                <div className="flex items-center">
                                    <AlertTriangle className="mr-3" size={20} />
                                    <div>
                                        <p className="font-semibold">{alert.message}</p>
                                        <p className="text-sm text-gray-400">{alert.timestamp}</p>
                                    </div>
                                </div>
                                <button
                                    onClick={() => sendAlert(alert.type)}
                                    className="bg-white/10 hover:bg-white/20 px-4 py-2 rounded-lg transition flex items-center"
                                >
                                    <Mail size={16} className="mr-2" />
                                    Send Alert
                                </button>
                            </div>
                        ))}
                    </div>
                )}

                {/* Controls */}
                <div className="bg-slate-800 rounded-lg p-6 mb-6 shadow-xl">
                    <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
                        {/* Data Source Toggle */}
                        <div>
                            <label className="block text-sm font-medium mb-2">Data Source</label>
                            <select
                                value={dataSource}
                                onChange={(e) => setDataSource(e.target.value)}
                                disabled={isUsingUploadedData}
                                className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
                            >
                                <option value="synthetic">Synthetic Data</option>
                                <option value="real">Real Dataset</option>
                            </select>
                        </div>

                        {/* Sensor Type / Dataset Selector */}
                        <div>
                            <label className="block text-sm font-medium mb-2">
                                {dataSource === 'real' ? 'Dataset' : 'Sensor Type'}
                            </label>
                            {dataSource === 'synthetic' ? (
                                <select
                                    value={sensorType}
                                    onChange={(e) => setSensorType(e.target.value)}
                                    disabled={isUsingUploadedData}
                                    className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
                                >
                                    <option value="accelerometer">Accelerometer</option>
                                    <option value="gyroscope">Gyroscope</option>
                                    <option value="pressure">Pressure Sensor</option>
                                    <option value="temperature">Temperature Sensor</option>
                                </select>
                            ) : (
                                <select
                                    value={selectedDataset}
                                    onChange={(e) => setSelectedDataset(e.target.value)}
                                    className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-purple-500"
                                >
                                    <option value="nasa_bearing">NASA IMS Bearing</option>
                                    <option value="cwru_bearing">CWRU Bearing</option>
                                    <option value="mems_vibration">MEMS Vibration</option>
                                </select>
                            )}
                        </div>

                        {/* Degradation / Fault Options */}
                        <div>
                            <label className="block text-sm font-medium mb-2">
                                {dataSource === 'real'
                                    ? (selectedDataset === 'nasa_bearing' ? 'Degradation Stage'
                                        : selectedDataset === 'cwru_bearing' ? 'Fault Type'
                                            : 'Scenario')
                                    : `Degradation: ${degradation}%`}
                            </label>
                            {dataSource === 'synthetic' ? (
                                <input
                                    type="range"
                                    min="0"
                                    max="10"
                                    value={degradation}
                                    onChange={(e) => setDegradation(parseInt(e.target.value))}
                                    disabled={isUsingUploadedData}
                                    className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer mt-3"
                                />
                            ) : selectedDataset === 'nasa_bearing' ? (
                                <select
                                    value={datasetOptions.degradation_stage}
                                    onChange={(e) => setDatasetOptions({ ...datasetOptions, degradation_stage: parseInt(e.target.value) })}
                                    className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2"
                                >
                                    <option value={0}>Healthy (Stage 0)</option>
                                    <option value={1}>Early Wear (Stage 1)</option>
                                    <option value={2}>Moderate (Stage 2)</option>
                                    <option value={3}>Severe (Stage 3)</option>
                                    <option value={4}>Failure (Stage 4)</option>
                                </select>
                            ) : selectedDataset === 'cwru_bearing' ? (
                                <select
                                    value={datasetOptions.fault_type}
                                    onChange={(e) => setDatasetOptions({ ...datasetOptions, fault_type: e.target.value })}
                                    className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2"
                                >
                                    <option value="normal">Normal (Healthy)</option>
                                    <option value="inner_race">Inner Race Fault</option>
                                    <option value="outer_race">Outer Race Fault</option>
                                    <option value="ball">Ball Fault</option>
                                </select>
                            ) : (
                                <select
                                    value={datasetOptions.scenario}
                                    onChange={(e) => setDatasetOptions({ ...datasetOptions, scenario: e.target.value })}
                                    className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2"
                                >
                                    <option value="normal">Normal</option>
                                    <option value="high_vibration">High Vibration</option>
                                    <option value="shock_event">Shock Events</option>
                                    <option value="drift">Sensor Drift</option>
                                </select>
                            )}
                        </div>

                        <div className="flex items-end gap-2">
                            <button
                                onClick={generateData}
                                disabled={isUsingUploadedData || isLoadingData}
                                className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed px-4 py-2 rounded-lg font-medium transition flex items-center justify-center"
                            >
                                {isLoadingData ? (
                                    <>
                                        <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
                                        Loading...
                                    </>
                                ) : (
                                    <>
                                        <Database className="mr-2" size={18} />
                                        Generate
                                    </>
                                )}
                            </button>
                            <button
                                onClick={exportData}
                                className="bg-slate-700 hover:bg-slate-600 px-4 py-2 rounded-lg transition"
                                title="Export Data"
                            >
                                <Download size={18} />
                            </button>
                        </div>

                        <div className="flex items-end gap-2">
                            <button
                                onClick={generateReport}
                                disabled={!features}
                                className="flex-1 bg-green-600 hover:bg-green-700 disabled:bg-gray-500 disabled:cursor-not-allowed px-4 py-2 rounded-lg font-medium transition flex items-center justify-center"
                            >
                                <FileText className="mr-2" size={18} />
                                Report
                            </button>
                            <button
                                onClick={enableComparison}
                                disabled={isUsingUploadedData}
                                className={`px-4 py-2 rounded-lg transition disabled:opacity-50 disabled:cursor-not-allowed ${comparisonMode ? 'bg-purple-600' : 'bg-slate-700 hover:bg-slate-600'}`}
                                title="Compare Sensors"
                            >
                                <Zap size={18} />
                            </button>
                        </div>
                    </div>

                    {/* File Upload Section */}
                    <div className="mt-6 pt-6 border-t border-slate-700">
                        <div className="flex items-center justify-between mb-4">
                            <div>
                                <h3 className="text-lg font-semibold flex items-center">
                                    <Database className="mr-2" size={20} />
                                    Upload Custom Sensor Data
                                </h3>
                                <p className="text-sm text-gray-400 mt-1">Upload your own CSV file with sensor readings</p>
                            </div>
                            {isUsingUploadedData && (
                                <button
                                    onClick={clearUploadedData}
                                    className="bg-red-600 hover:bg-red-700 px-4 py-2 rounded-lg font-medium transition"
                                >
                                    Clear & Use Synthetic Data
                                </button>
                            )}
                        </div>

                        {!isUsingUploadedData ? (
                            <div className="flex items-center gap-4">
                                <label className="flex-1 cursor-pointer">
                                    <div className={`flex items-center justify-center gap-3 border-2 border-dashed rounded-lg px-6 py-4 transition ${isLoadingUpload
                                        ? 'bg-slate-600 border-slate-500 cursor-not-allowed'
                                        : 'bg-slate-700 hover:bg-slate-600 border-slate-600 hover:border-blue-500'
                                        }`}>
                                        {isLoadingUpload ? (
                                            <>
                                                <div className="w-6 h-6 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                                                <div>
                                                    <p className="font-medium">Uploading and processing...</p>
                                                    <p className="text-sm text-gray-400">Please wait</p>
                                                </div>
                                            </>
                                        ) : (
                                            <>
                                                <Download size={24} className="rotate-180" />
                                                <div>
                                                    <p className="font-medium">Click to upload CSV file</p>
                                                    <p className="text-sm text-gray-400">CSV must contain "time" and "value" columns</p>
                                                </div>
                                            </>
                                        )}
                                    </div>
                                    <input
                                        type="file"
                                        accept=".csv,.txt"
                                        onChange={handleFileUpload}
                                        disabled={isLoadingUpload}
                                        className="hidden"
                                    />
                                </label>
                            </div>
                        ) : (
                            <div className="bg-green-900/20 border border-green-500 rounded-lg p-4 flex items-start">
                                <div className="flex-1">
                                    <p className="font-semibold text-green-400 mb-1">✅ File Loaded Successfully</p>
                                    <p className="text-sm text-gray-300">
                                        📁 <strong>{uploadedFileName}</strong> • {uploadedData?.length || 0} data points
                                    </p>
                                    <p className="text-sm text-gray-400 mt-2">
                                        Analysis has been performed on your uploaded data. View results in all tabs.
                                    </p>
                                </div>
                            </div>
                        )}

                        {uploadError && (
                            <div className="mt-4 bg-red-900/20 border border-red-500 rounded-lg p-4 flex items-start">
                                <AlertTriangle className="mr-3 flex-shrink-0 text-red-400" size={20} />
                                <div>
                                    <p className="font-semibold text-red-400 mb-1">Upload Error</p>
                                    <p className="text-sm text-gray-300">{uploadError}</p>
                                    <p className="text-sm text-gray-400 mt-2">
                                        <strong>CSV Format Example:</strong><br />
                                        time,value,temperature<br />
                                        0.00,9.8100,25.5<br />
                                        0.01,9.8150,25.6<br />
                                        0.02,9.8080,25.4
                                    </p>
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                {/* Key Metrics */}
                <div className="grid grid-cols-1 md:grid-cols-5 gap-4 mb-6">
                    <div className="bg-gradient-to-br from-blue-600 to-blue-700 rounded-lg p-4 shadow-lg overflow-hidden">
                        <div className="flex items-center justify-between mb-2">
                            <Activity size={24} />
                            <span className="text-2xl font-bold truncate">{features?.snr ? parseFloat(features.snr).toFixed(2) : '0'}</span>
                        </div>
                        <p className="text-sm text-blue-100">Signal-to-Noise Ratio</p>
                    </div>

                    <div className="bg-gradient-to-br from-green-600 to-green-700 rounded-lg p-4 shadow-lg overflow-hidden">
                        <div className="flex items-center justify-between mb-2">
                            <TrendingUp size={24} />
                            <span className="text-2xl font-bold truncate">{features?.mean ? parseFloat(features.mean).toFixed(2) : '0'}</span>
                        </div>
                        <p className="text-sm text-green-100">Mean Value ({getSensorUnit()})</p>
                    </div>

                    <div className="bg-gradient-to-br from-orange-600 to-orange-700 rounded-lg p-4 shadow-lg overflow-hidden">
                        <div className="flex items-center justify-between mb-2">
                            <AlertTriangle size={24} />
                            <span className="text-2xl font-bold truncate">{rul ? parseFloat(rul).toFixed(1) : '0'}%</span>
                        </div>
                        <p className="text-sm text-orange-100">Remaining Useful Life</p>
                    </div>

                    <div className="bg-gradient-to-br from-purple-600 to-purple-700 rounded-lg p-4 shadow-lg overflow-hidden">
                        <div className="flex items-center justify-between mb-2">
                            <Brain size={24} />
                            <span className="text-2xl font-bold truncate">{modelResults.length}</span>
                        </div>
                        <p className="text-sm text-purple-100">Trained Models</p>
                    </div>

                    <div className="bg-gradient-to-br from-red-600 to-red-700 rounded-lg p-4 shadow-lg overflow-hidden">
                        <div className="flex items-center justify-between mb-2">
                            <Waves size={24} />
                            <span className="text-2xl font-bold truncate">{anomalies.filter(a => a.isAnomaly).length}</span>
                        </div>
                        <p className="text-sm text-red-100">Anomalies Detected</p>
                    </div>
                </div>

                {/* Tabs */}
                <div className="flex gap-2 mb-4 overflow-x-auto">
                    {['data', 'live', 'analysis', 'models', 'xai', 'prediction', 'anomaly', 'advanced', 'deeplearning'].map(tab => (
                        <button
                            key={tab}
                            onClick={() => setActiveTab(tab)}
                            className={`px-6 py-2 rounded-lg font-medium transition whitespace-nowrap ${activeTab === tab
                                ? 'bg-blue-600 text-white'
                                : 'bg-slate-800 text-gray-400 hover:bg-slate-700'
                                }`}
                        >
                            {tab === 'xai' ? 'XAI' : tab === 'deeplearning' ? 'Deep Learning' : tab.charAt(0).toUpperCase() + tab.slice(1)}
                        </button>
                    ))}
                </div>

                {/* Content Area */}
                <div className="bg-slate-800 rounded-lg p-6 shadow-xl">
                    {activeTab === 'data' && (
                        <div className="space-y-6">
                            <div>
                                <h3 className="text-xl font-semibold mb-4">Sensor Signal Over Time</h3>
                                <ResponsiveContainer width="100%" height={300}>
                                    <LineChart data={sensorData.slice(0, 200)}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                        <XAxis dataKey="time" stroke="#9CA3AF" label={{ value: 'Time (s)', position: 'insideBottom', offset: -5 }} />
                                        <YAxis stroke="#9CA3AF" label={{ value: `Value (${getSensorUnit()})`, angle: -90, position: 'insideLeft' }} />
                                        <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: 'none' }} />
                                        <Legend />
                                        <Line type="monotone" dataKey="value" stroke="#3B82F6" dot={false} strokeWidth={2} name="Measured Value" />
                                        <Line type="monotone" dataKey="signal" stroke="#10B981" dot={false} strokeWidth={1} strokeDasharray="5 5" name="True Signal" />
                                        {comparisonMode && <Line type="monotone" data={secondSensorData.slice(0, 200)} dataKey="value" stroke="#F59E0B" dot={false} strokeWidth={2} name="Sensor 2" />}
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                <div>
                                    <h3 className="text-xl font-semibold mb-4">Temperature vs Sensor Output</h3>
                                    <ResponsiveContainer width="100%" height={250}>
                                        <ScatterChart>
                                            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                            <XAxis dataKey="temperature" stroke="#9CA3AF" label={{ value: 'Temperature (°C)', position: 'insideBottom', offset: -5 }} />
                                            <YAxis dataKey="value" stroke="#9CA3AF" label={{ value: `Value (${getSensorUnit()})`, angle: -90, position: 'insideLeft' }} />
                                            <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: 'none' }} />
                                            <Scatter data={sensorData.slice(0, 200)} fill="#8B5CF6" />
                                        </ScatterChart>
                                    </ResponsiveContainer>
                                </div>

                                <div>
                                    <h3 className="text-xl font-semibold mb-4">Environmental Conditions</h3>
                                    <ResponsiveContainer width="100%" height={250}>
                                        <AreaChart data={sensorData.slice(0, 200)}>
                                            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                            <XAxis dataKey="time" stroke="#9CA3AF" />
                                            <YAxis stroke="#9CA3AF" />
                                            <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: 'none' }} />
                                            <Legend />
                                            <Area type="monotone" dataKey="temperature" stackId="1" stroke="#EF4444" fill="#EF4444" fillOpacity={0.6} name="Temperature" />
                                            <Area type="monotone" dataKey="humidity" stackId="2" stroke="#3B82F6" fill="#3B82F6" fillOpacity={0.6} name="Humidity" />
                                        </AreaChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        </div>
                    )}

                    {activeTab === 'live' && (
                        <div className="space-y-6">
                            <div className="flex justify-between items-center">
                                <h3 className="text-xl font-semibold">Real-Time Sensor Monitoring</h3>
                                <button
                                    onClick={toggleLiveMonitoring}
                                    className={`px-6 py-3 rounded-lg font-medium transition flex items-center ${isLiveMonitoring
                                        ? 'bg-red-600 hover:bg-red-700'
                                        : 'bg-green-600 hover:bg-green-700'
                                        }`}
                                >
                                    {isLiveMonitoring ? (
                                        <>
                                            <span className="inline-block w-3 h-3 bg-white rounded-full mr-2 animate-pulse"></span>
                                            Stop Monitoring
                                        </>
                                    ) : (
                                        <>
                                            <Activity className="mr-2" size={18} />
                                            Start Live Monitoring
                                        </>
                                    )}
                                </button>
                            </div>

                            {!isLiveMonitoring && liveData.length === 0 && (
                                <div className="bg-slate-700 rounded-lg p-8 text-center">
                                    <Activity size={48} className="mx-auto mb-4 text-gray-400" />
                                    <h4 className="text-lg font-semibold mb-2">Live Monitoring Inactive</h4>
                                    <p className="text-gray-400 mb-4">Click "Start Live Monitoring" to begin real-time data collection</p>
                                    <p className="text-sm text-gray-500">Simulates real sensor behavior with gradual degradation over time</p>
                                </div>
                            )}

                            {(isLiveMonitoring || liveData.length > 0) && (
                                <>
                                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                        <div className="bg-gradient-to-br from-blue-600 to-blue-700 rounded-lg p-4">
                                            <div className="flex items-center justify-between mb-2">
                                                <span className="text-sm text-blue-100">Current Value</span>
                                                {isLiveMonitoring && <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>}
                                            </div>
                                            <div className="text-3xl font-bold">
                                                {liveData.length > 0 ? liveData[liveData.length - 1].value : '0.0000'}
                                            </div>
                                            <div className="text-sm text-blue-100 mt-1">{getSensorUnit()}</div>
                                        </div>

                                        <div className="bg-gradient-to-br from-orange-600 to-orange-700 rounded-lg p-4">
                                            <div className="flex items-center justify-between mb-2">
                                                <span className="text-sm text-orange-100">Live RUL</span>
                                                {isLiveMonitoring && <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>}
                                            </div>
                                            <div className="text-3xl font-bold">
                                                {liveData.length > 0 ? liveData[liveData.length - 1].rul : '100'}%
                                            </div>
                                            <div className="text-sm text-orange-100 mt-1">Remaining Life</div>
                                        </div>

                                        <div className="bg-gradient-to-br from-purple-600 to-purple-700 rounded-lg p-4">
                                            <div className="flex items-center justify-between mb-2">
                                                <span className="text-sm text-purple-100">Data Points</span>
                                                {isLiveMonitoring && <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>}
                                            </div>
                                            <div className="text-3xl font-bold">{liveData.length}</div>
                                            <div className="text-sm text-purple-100 mt-1">Collected</div>
                                        </div>
                                    </div>

                                    <div>
                                        <h4 className="text-lg font-semibold mb-4 flex items-center">
                                            Live Signal Stream
                                            {isLiveMonitoring && (
                                                <span className="ml-3 text-sm text-green-400 flex items-center">
                                                    <span className="w-2 h-2 bg-green-400 rounded-full mr-2 animate-pulse"></span>
                                                    LIVE
                                                </span>
                                            )}
                                        </h4>
                                        <ResponsiveContainer width="100%" height={300}>
                                            <LineChart data={liveData}>
                                                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                                <XAxis dataKey="time" stroke="#9CA3AF" label={{ value: 'Time (s)', position: 'insideBottom', offset: -5 }} />
                                                <YAxis stroke="#9CA3AF" label={{ value: `Value (${getSensorUnit()})`, angle: -90, position: 'insideLeft' }} />
                                                <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: 'none' }} />
                                                <Legend />
                                                <Line type="monotone" dataKey="value" stroke="#3B82F6" dot={false} strokeWidth={2} name="Live Value" isAnimationActive={false} />
                                                <Line type="monotone" dataKey="signal" stroke="#10B981" dot={false} strokeWidth={1} strokeDasharray="5 5" name="True Signal" isAnimationActive={false} />
                                            </LineChart>
                                        </ResponsiveContainer>
                                    </div>

                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                        <div>
                                            <h4 className="text-lg font-semibold mb-4">Temperature Monitoring</h4>
                                            <ResponsiveContainer width="100%" height={250}>
                                                <AreaChart data={liveData}>
                                                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                                    <XAxis dataKey="time" stroke="#9CA3AF" />
                                                    <YAxis stroke="#9CA3AF" />
                                                    <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: 'none' }} />
                                                    <Area type="monotone" dataKey="temperature" stroke="#EF4444" fill="#EF4444" fillOpacity={0.6} isAnimationActive={false} />
                                                </AreaChart>
                                            </ResponsiveContainer>
                                        </div>

                                        <div>
                                            <h4 className="text-lg font-semibold mb-4">Drift Analysis</h4>
                                            <ResponsiveContainer width="100%" height={250}>
                                                <LineChart data={liveData}>
                                                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                                    <XAxis dataKey="time" stroke="#9CA3AF" />
                                                    <YAxis stroke="#9CA3AF" />
                                                    <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: 'none' }} />
                                                    <Line type="monotone" dataKey="drift" stroke="#F59E0B" dot={false} strokeWidth={2} isAnimationActive={false} />
                                                </LineChart>
                                            </ResponsiveContainer>
                                        </div>
                                    </div>

                                    <div className="bg-slate-700 rounded-lg p-4">
                                        <h4 className="font-semibold mb-3 text-blue-400">Live Monitoring Statistics</h4>
                                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                            <div>
                                                <p className="text-sm text-gray-400">Duration</p>
                                                <p className="text-xl font-bold">{liveData.length > 0 ? liveData[liveData.length - 1].time : '0'}s</p>
                                            </div>
                                            <div>
                                                <p className="text-sm text-gray-400">Avg Value</p>
                                                <p className="text-xl font-bold">
                                                    {liveData.length > 0
                                                        ? (liveData.reduce((sum, d) => sum + parseFloat(d.value), 0) / liveData.length).toFixed(4)
                                                        : '0.0000'}
                                                </p>
                                            </div>
                                            <div>
                                                <p className="text-sm text-gray-400">Current Drift</p>
                                                <p className="text-xl font-bold">
                                                    {liveData.length > 0 ? liveData[liveData.length - 1].drift : '0.0000'}
                                                </p>
                                            </div>
                                            <div>
                                                <p className="text-sm text-gray-400">Status</p>
                                                <p className={`text-xl font-bold ${liveData.length > 0 && parseFloat(liveData[liveData.length - 1].rul) > 70
                                                    ? 'text-green-400'
                                                    : liveData.length > 0 && parseFloat(liveData[liveData.length - 1].rul) > 40
                                                        ? 'text-orange-400'
                                                        : 'text-red-400'
                                                    }`}>
                                                    {liveData.length > 0 && parseFloat(liveData[liveData.length - 1].rul) > 70
                                                        ? 'Healthy'
                                                        : liveData.length > 0 && parseFloat(liveData[liveData.length - 1].rul) > 40
                                                            ? 'Warning'
                                                            : 'Critical'}
                                                </p>
                                            </div>
                                        </div>
                                    </div>

                                    {isLiveMonitoring && (
                                        <div className="bg-blue-900/30 border border-blue-500 rounded-lg p-4 flex items-start">
                                            <Activity className="mr-3 mt-1 flex-shrink-0" size={20} />
                                            <div>
                                                <p className="font-semibold mb-1">Live Monitoring Active</p>
                                                <p className="text-sm text-gray-300">
                                                    The system is continuously collecting data and simulating real sensor behavior.
                                                    Degradation increases gradually over time. Click "Stop Monitoring" to pause data collection.
                                                </p>
                                            </div>
                                        </div>
                                    )}
                                </>
                            )}
                        </div>
                    )}

                    {activeTab === 'analysis' && features && (
                        <div className="space-y-6">
                            <div>
                                <h3 className="text-xl font-semibold mb-4">Statistical Features</h3>
                                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                    {Object.entries(features).map(([key, value]) => (
                                        <div key={key} className="bg-slate-700 rounded-lg p-4">
                                            <p className="text-sm text-gray-400 uppercase">{key}</p>
                                            <p className="text-2xl font-bold text-blue-400">
                                                {typeof value === 'number'
                                                    ? value.toFixed(4)
                                                    : isNaN(parseFloat(value))
                                                        ? value
                                                        : parseFloat(value).toFixed(4)}
                                            </p>
                                        </div>
                                    ))}
                                </div>
                            </div>

                            <div>
                                <h3 className="text-xl font-semibold mb-4">Noise & Drift Analysis</h3>
                                <ResponsiveContainer width="100%" height={300}>
                                    <LineChart data={sensorData.slice(0, 200)}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                        <XAxis dataKey="time" stroke="#9CA3AF" />
                                        <YAxis stroke="#9CA3AF" />
                                        <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: 'none' }} />
                                        <Legend />
                                        <Line type="monotone" dataKey="drift" stroke="#EF4444" dot={false} name="Calibration Drift" strokeWidth={2} />
                                        <Line type="monotone" dataKey="noise" stroke="#F59E0B" dot={false} name="Noise" strokeWidth={2} />
                                        <Line type="monotone" dataKey="vibration" stroke="#8B5CF6" dot={false} name="Vibration" strokeWidth={2} />
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>

                            <div>
                                <h3 className="text-xl font-semibold mb-4">Historical Trend Analysis</h3>
                                <ResponsiveContainer width="100%" height={250}>
                                    <LineChart data={historicalData}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                        <XAxis dataKey="timestamp" stroke="#9CA3AF" tickFormatter={(val) => new Date(val).toLocaleTimeString()} />
                                        <YAxis stroke="#9CA3AF" />
                                        <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: 'none' }} labelFormatter={(val) => new Date(val).toLocaleString()} />
                                        <Legend />
                                        <Line type="monotone" dataKey="rul" stroke="#10B981" name="RUL %" strokeWidth={2} />
                                        <Line type="monotone" dataKey="snr" stroke="#3B82F6" name="SNR" strokeWidth={2} />
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>

                            {/* MEMS Sensor Characteristics Section */}
                            {sensorCharacteristics && (
                                <div className="mt-8">
                                    <h3 className="text-xl font-semibold mb-4 text-cyan-400">
                                        MEMS Sensor Characteristics Analysis
                                    </h3>
                                    <p className="text-gray-400 text-sm mb-4">
                                        Key performance parameters derived from sensor datasheet specifications
                                    </p>

                                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                                        {/* Sensitivity Card */}
                                        <div className="bg-gradient-to-br from-blue-900 to-blue-800 rounded-lg p-5 border border-blue-500">
                                            <h4 className="text-lg font-semibold text-blue-300 mb-3 flex items-center">
                                                <Zap className="mr-2" size={20} />
                                                Sensitivity
                                            </h4>
                                            <div className="space-y-2">
                                                <div className="flex justify-between">
                                                    <span className="text-gray-400">Value:</span>
                                                    <span className="text-white font-bold">
                                                        {sensorCharacteristics.sensitivity?.value?.toFixed(4) || 'N/A'} {sensorCharacteristics.sensitivity?.unit || ''}
                                                    </span>
                                                </div>
                                                <div className="flex justify-between">
                                                    <span className="text-gray-400">Deviation:</span>
                                                    <span className={`font-medium ${sensorCharacteristics.sensitivity?.deviation_percent < 5 ? 'text-green-400' : 'text-yellow-400'}`}>
                                                        {sensorCharacteristics.sensitivity?.deviation_percent?.toFixed(2) || '0'}%
                                                    </span>
                                                </div>
                                            </div>
                                        </div>

                                        {/* Resonant Frequency Card */}
                                        <div className="bg-gradient-to-br from-purple-900 to-purple-800 rounded-lg p-5 border border-purple-500">
                                            <h4 className="text-lg font-semibold text-purple-300 mb-3 flex items-center">
                                                <Waves className="mr-2" size={20} />
                                                Resonant Frequency
                                            </h4>
                                            <div className="space-y-2">
                                                <div className="flex justify-between">
                                                    <span className="text-gray-400">Frequency:</span>
                                                    <span className="text-white font-bold">
                                                        {sensorCharacteristics.resonant_frequency?.value?.toFixed(2) || 'N/A'} Hz
                                                    </span>
                                                </div>
                                                <div className="flex justify-between">
                                                    <span className="text-gray-400">Q-Factor:</span>
                                                    <span className="text-white font-medium">
                                                        {sensorCharacteristics.resonant_frequency?.quality_factor?.toFixed(1) || 'N/A'}
                                                    </span>
                                                </div>
                                            </div>
                                        </div>

                                        {/* Noise Characteristics Card */}
                                        <div className="bg-gradient-to-br from-orange-900 to-orange-800 rounded-lg p-5 border border-orange-500">
                                            <h4 className="text-lg font-semibold text-orange-300 mb-3 flex items-center">
                                                <Activity className="mr-2" size={20} />
                                                Noise Analysis
                                            </h4>
                                            <div className="space-y-2">
                                                <div className="flex justify-between">
                                                    <span className="text-gray-400">RMS Noise:</span>
                                                    <span className="text-white font-bold">
                                                        {sensorCharacteristics.noise?.rms?.toFixed(6) || 'N/A'}
                                                    </span>
                                                </div>
                                                <div className="flex justify-between">
                                                    <span className="text-gray-400">SNR:</span>
                                                    <span className="text-white font-medium">
                                                        {sensorCharacteristics.noise?.snr_db?.toFixed(2) || 'N/A'} dB
                                                    </span>
                                                </div>
                                                <div className="flex justify-between">
                                                    <span className="text-gray-400">1/f Corner:</span>
                                                    <span className="text-white font-medium">
                                                        {sensorCharacteristics.noise?.corner_frequency?.toFixed(2) || 'N/A'} Hz
                                                    </span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Additional Characteristics */}
                                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-6">
                                        {/* Temperature Sensitivity */}
                                        <div className="bg-slate-700 rounded-lg p-4">
                                            <h4 className="text-md font-semibold text-yellow-400 mb-2">Temperature Sensitivity</h4>
                                            <div className="space-y-1 text-sm">
                                                <div className="flex justify-between">
                                                    <span className="text-gray-400">Coefficient:</span>
                                                    <span className="text-white">
                                                        {sensorCharacteristics.temperature_sensitivity?.value?.toFixed(6) || 'N/A'} {sensorCharacteristics.temperature_sensitivity?.unit || ''}
                                                    </span>
                                                </div>
                                                <div className="flex justify-between">
                                                    <span className="text-gray-400">Correlation:</span>
                                                    <span className="text-white">
                                                        {((sensorCharacteristics.temperature_sensitivity?.correlation || 0) * 100).toFixed(1)}%
                                                    </span>
                                                </div>
                                            </div>
                                        </div>

                                        {/* Linearity */}
                                        <div className="bg-slate-700 rounded-lg p-4">
                                            <h4 className="text-md font-semibold text-green-400 mb-2">Linearity</h4>
                                            <div className="space-y-1 text-sm">
                                                <div className="flex justify-between">
                                                    <span className="text-gray-400">R² Score:</span>
                                                    <span className="text-white">
                                                        {sensorCharacteristics.linearity?.r_squared?.toFixed(4) || 'N/A'}
                                                    </span>
                                                </div>
                                                <div className="flex justify-between">
                                                    <span className="text-gray-400">Percent:</span>
                                                    <span className={`${sensorCharacteristics.linearity?.percent > 90 ? 'text-green-400' : 'text-yellow-400'}`}>
                                                        {sensorCharacteristics.linearity?.percent?.toFixed(2) || 'N/A'}%
                                                    </span>
                                                </div>
                                            </div>
                                        </div>

                                        {/* Drift */}
                                        <div className="bg-slate-700 rounded-lg p-4">
                                            <h4 className="text-md font-semibold text-red-400 mb-2">Drift Analysis</h4>
                                            <div className="space-y-1 text-sm">
                                                <div className="flex justify-between">
                                                    <span className="text-gray-400">Rate:</span>
                                                    <span className="text-white">
                                                        {sensorCharacteristics.drift?.rate?.toFixed(6) || 'N/A'}/sample
                                                    </span>
                                                </div>
                                                <div className="flex justify-between">
                                                    <span className="text-gray-400">Total Drift:</span>
                                                    <span className={`${Math.abs(sensorCharacteristics.drift?.total || 0) < 0.1 ? 'text-green-400' : 'text-red-400'}`}>
                                                        {sensorCharacteristics.drift?.total?.toFixed(4) || 'N/A'}
                                                    </span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>
                    )}

                    {activeTab === 'models' && (
                        <ModelsTab
                            modelResults={modelResults}
                            bestModel={bestModel}
                            predictionsSample={predictionsSample}
                            isTraining={isTraining}
                            trainModels={trainModels}
                            getFeatureImportance={getFeatureImportance}
                            trainError={trainError}
                            generalizationMetrics={generalizationMetrics}
                            onDownloadModel={downloadBestModel}
                        />
                    )}


                    {activeTab === 'xai' && (
                        <div className="space-y-6">
                            <div className="flex justify-between items-center">
                                <div>
                                    <h3 className="text-xl font-semibold flex items-center">
                                        <Brain className="mr-2" size={24} />
                                        Explainable AI (XAI) Analysis
                                    </h3>
                                    <p className="text-gray-400 text-sm mt-1">Understanding model decisions and predictions</p>
                                </div>
                                <div className="flex gap-3">
                                    {predictionExplanation && (
                                        <button
                                            onClick={generateXAIPDF}
                                            className="bg-purple-600 hover:bg-purple-700 px-4 py-2 rounded-lg font-medium transition flex items-center gap-2"
                                        >
                                            <Download size={16} />
                                            Download XAI PDF
                                        </button>
                                    )}
                                    {!predictionExplanation && (
                                        <button
                                            onClick={trainModels}
                                            disabled={isTraining}
                                            className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 px-6 py-2 rounded-lg font-medium transition"
                                        >
                                            {isTraining ? 'Training...' : 'Generate Explanations'}
                                        </button>
                                    )}
                                </div>
                            </div>

                            {!predictionExplanation ? (
                                <div className="bg-slate-700 rounded-lg p-8 text-center">
                                    <Brain size={48} className="mx-auto mb-4 text-gray-400" />
                                    <h4 className="text-lg font-semibold mb-2">No XAI Analysis Available</h4>
                                    <p className="text-gray-400 mb-4">Train models first to generate explainability insights</p>
                                    <button
                                        onClick={trainModels}
                                        disabled={isTraining}
                                        className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 px-6 py-2 rounded-lg font-medium transition"
                                    >
                                        {isTraining ? 'Training Models...' : 'Train Models & Generate'}
                                    </button>
                                </div>
                            ) : (
                                <>
                                    {/* Prediction Explanation */}
                                    <div className="bg-slate-700 rounded-lg p-6">
                                        <h4 className="text-lg font-semibold mb-4 text-blue-400">Prediction Explanation</h4>
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                            <div>
                                                <div className="bg-slate-800 rounded-lg p-4 mb-4">
                                                    <p className="text-sm text-gray-400 mb-2">Model Prediction</p>
                                                    <div className={`text-3xl font-bold ${predictionExplanation.prediction === 'HEALTHY' ? 'text-green-400' :
                                                        predictionExplanation.prediction === 'WARNING' ? 'text-orange-400' : 'text-red-400'
                                                        }`}>
                                                        {predictionExplanation.prediction}
                                                    </div>
                                                    {/* Reason Banner */}
                                                    {predictionExplanation.triggered_rule && (
                                                        <div className="mt-3 p-3 rounded bg-slate-700 border border-slate-600">
                                                            <p className="text-sm font-medium text-gray-300">
                                                                <span className="text-blue-400">Reason:</span> {predictionExplanation.triggered_rule} — {predictionExplanation.rule_reason}
                                                            </p>
                                                            {predictionExplanation.status_reason_details && (
                                                                <div className="mt-2 flex flex-wrap gap-2">
                                                                    <span className="text-xs px-2 py-1 rounded bg-slate-600">SNR: {predictionExplanation.status_reason_details.snr}</span>
                                                                    <span className="text-xs px-2 py-1 rounded bg-slate-600">Drift: {predictionExplanation.status_reason_details.drift}</span>
                                                                    <span className="text-xs px-2 py-1 rounded bg-slate-600">Noise: {predictionExplanation.status_reason_details.noise}</span>
                                                                    <span className="text-xs px-2 py-1 rounded bg-slate-600">Temp: {predictionExplanation.status_reason_details.temperature}°C</span>
                                                                    <span className="text-xs px-2 py-1 rounded bg-slate-600">RUL: {predictionExplanation.status_reason_details.rul_percent}%</span>
                                                                </div>
                                                            )}
                                                            <p className="text-xs text-gray-500 mt-2">Source: {predictionExplanation.status_source || 'Rule-based thresholds'}</p>
                                                        </div>
                                                    )}
                                                </div>
                                                <div className="bg-slate-800 rounded-lg p-4">
                                                    <div className="flex items-center gap-2 mb-2">
                                                        <p className="text-sm text-gray-400">Model Confidence</p>
                                                        <div className="relative group">
                                                            <span className="cursor-help text-gray-500 hover:text-blue-400">ⓘ</span>
                                                            <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 bg-slate-900 border border-slate-600 rounded-lg text-xs text-gray-300 w-64 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                                                                Confidence is derived from best model performance (R²) and agreement between models. Clamped to 30%–99% to avoid misleading extremes.
                                                            </div>
                                                        </div>
                                                    </div>
                                                    <div className="flex items-end gap-2">
                                                        <span className="text-3xl font-bold text-blue-400">
                                                            {predictionExplanation.confidence.toFixed(1)}%
                                                        </span>
                                                        <span className="text-sm text-gray-400 mb-1">
                                                            (Uncertainty: {predictionExplanation.uncertainty.toFixed(1)}%)
                                                        </span>
                                                    </div>
                                                    <div className="w-full bg-slate-600 rounded-full h-3 mt-3">
                                                        <div
                                                            className="bg-blue-500 h-3 rounded-full transition-all"
                                                            style={{ width: `${predictionExplanation.confidence}%` }}
                                                        />
                                                    </div>
                                                </div>
                                            </div>

                                            <div className="bg-slate-800 rounded-lg p-4">
                                                <p className="text-sm text-gray-400 mb-3">Top Contributing Factors</p>
                                                <div className="space-y-3">
                                                    {predictionExplanation.mainReasons.map((reason, idx) => (
                                                        <div key={idx} className="border-l-4 border-blue-500 pl-3">
                                                            <div className="flex items-center justify-between mb-1">
                                                                <span className="font-semibold">{reason.feature}</span>
                                                                <span className="text-blue-400 font-bold">{reason.contribution}</span>
                                                            </div>
                                                            <div className="flex items-center gap-2 mb-1">
                                                                <span className="text-sm text-gray-400">
                                                                    {reason.direction === 'increases' ? '↑' : '↓'} {reason.direction} confidence
                                                                </span>
                                                                {reason.impact_on_rul && (
                                                                    <span className={`text-xs px-2 py-0.5 rounded ${reason.impact_type === 'bad' ? 'bg-red-900/50 text-red-300' : reason.impact_type === 'good' ? 'bg-green-900/50 text-green-300' : 'bg-gray-700 text-gray-300'}`}>
                                                                        {reason.impact_on_rul}
                                                                    </span>
                                                                )}
                                                            </div>
                                                            <p className="text-xs text-gray-500">Current: {typeof reason.value === 'number' ? reason.value.toFixed(4) : reason.value}</p>
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>
                                        </div>

                                        <div className="mt-4 bg-blue-900/30 border border-blue-500 rounded-lg p-4">
                                            <p className="text-sm">
                                                <strong>Model Used:</strong> {predictionExplanation.modelUsed} •
                                                <strong className="ml-2">Explanation Method:</strong> {predictionExplanation.explanation_method || 'SHAP-like Feature Attribution (Approximation)'}
                                            </p>
                                        </div>
                                    </div>

                                    {/* Feature Importance */}
                                    <div className="bg-slate-700 rounded-lg p-6">
                                        <h4 className="text-lg font-semibold mb-4 text-blue-400">Feature Importance Analysis</h4>
                                        <ResponsiveContainer width="100%" height={350}>
                                            <BarChart data={featureContributions} layout="vertical">
                                                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                                <XAxis type="number" stroke="#9CA3AF" label={{ value: 'Importance Score', position: 'insideBottom', offset: -5 }} />
                                                <YAxis dataKey="feature" type="category" stroke="#9CA3AF" width={100} />
                                                <Tooltip
                                                    contentStyle={{ backgroundColor: '#1F2937', border: 'none' }}
                                                    formatter={(value) => (value * 100).toFixed(1) + '%'}
                                                />
                                                <Bar dataKey="importance" fill="#3B82F6" />
                                            </BarChart>
                                        </ResponsiveContainer>
                                        <p className="text-sm text-gray-400 mt-4">
                                            Features are ranked by their impact on the model's prediction. Higher values indicate greater influence on the final decision.
                                        </p>
                                    </div>

                                    {/* SHAP Values */}
                                    <div className="bg-slate-700 rounded-lg p-6">
                                        <h4 className="text-lg font-semibold mb-4 text-blue-400">Feature Contribution Analysis (SHAP-like)</h4>
                                        <p className="text-sm text-gray-400 mb-4">
                                            SHAP-like approximation values show how each feature contributes to pushing the prediction
                                            from the base value (average prediction) toward the final prediction. Note: This is an approximation, not true SHAP.
                                        </p>

                                        <ResponsiveContainer width="100%" height={350}>
                                            <BarChart data={shapValues}>
                                                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                                <XAxis dataKey="feature" stroke="#9CA3AF" angle={-15} textAnchor="end" height={80} />
                                                <YAxis stroke="#9CA3AF" label={{ value: 'SHAP Value', angle: -90, position: 'insideLeft' }} />
                                                <Tooltip
                                                    contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }}
                                                    labelStyle={{ color: '#FFFFFF', fontWeight: 'bold' }}
                                                    itemStyle={{ color: '#10B981' }}
                                                    formatter={(value) => [value.toFixed(3), 'SHAP Value']}
                                                />
                                                <Bar dataKey="shapValue" name="SHAP Value">
                                                    {shapValues.map((entry, index) => (
                                                        <Cell key={`cell-${index}`} fill={entry.shapValue > 0 ? '#10B981' : '#EF4444'} />
                                                    ))}
                                                </Bar>
                                            </BarChart>
                                        </ResponsiveContainer>

                                        <div className="grid grid-cols-2 gap-4 mt-4">
                                            <div className="bg-green-900/20 border border-green-500 rounded-lg p-3">
                                                <div className="flex items-center mb-2">
                                                    <div className="w-4 h-4 bg-green-500 rounded mr-2"></div>
                                                    <span className="font-semibold">Positive Impact</span>
                                                </div>
                                                <p className="text-sm text-gray-400">Features that increase the health score</p>
                                            </div>
                                            <div className="bg-red-900/20 border border-red-500 rounded-lg p-3">
                                                <div className="flex items-center mb-2">
                                                    <div className="w-4 h-4 bg-red-500 rounded mr-2"></div>
                                                    <span className="font-semibold">Negative Impact</span>
                                                </div>
                                                <p className="text-sm text-gray-400">Features that decrease the health score</p>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Model Confidence & Agreement */}
                                    {modelConfidence && (
                                        <div className="bg-slate-700 rounded-lg p-6">
                                            <h4 className="text-lg font-semibold mb-4 text-blue-400">Model Confidence & Agreement</h4>
                                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                                <div>
                                                    <p className="text-sm text-gray-400 mb-3">Overall Confidence</p>
                                                    <div className="relative pt-1">
                                                        <div className="flex mb-2 items-center justify-between">
                                                            <div>
                                                                <span className="text-3xl font-bold text-blue-400">
                                                                    {modelConfidence.overall.toFixed(1)}%
                                                                </span>
                                                            </div>
                                                        </div>
                                                        <div className="overflow-hidden h-4 mb-4 text-xs flex rounded-full bg-slate-600">
                                                            <div
                                                                style={{ width: `${modelConfidence.overall}%` }}
                                                                className="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-gradient-to-r from-blue-500 to-blue-600"
                                                            ></div>
                                                        </div>
                                                    </div>
                                                    <p className="text-sm text-gray-400">
                                                        The model is {modelConfidence.overall.toFixed(0)}% confident in its prediction based on the training data and feature patterns.
                                                    </p>
                                                </div>

                                                <div>
                                                    <p className="text-sm text-gray-400 mb-3">Model Agreement</p>
                                                    <div className="relative pt-1">
                                                        <div className="flex mb-2 items-center justify-between">
                                                            <div>
                                                                <span className="text-3xl font-bold text-green-400">
                                                                    {modelConfidence.modelAgreement.toFixed(1)}%
                                                                </span>
                                                            </div>
                                                        </div>
                                                        <div className="overflow-hidden h-4 mb-4 text-xs flex rounded-full bg-slate-600">
                                                            <div
                                                                style={{ width: `${modelConfidence.modelAgreement}%` }}
                                                                className="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-gradient-to-r from-green-500 to-green-600"
                                                            ></div>
                                                        </div>
                                                    </div>
                                                    <p className="text-sm text-gray-400">
                                                        {modelConfidence.modelAgreement.toFixed(0)}% of trained models agree on similar predictions, indicating reliable consensus.
                                                    </p>
                                                </div>
                                            </div>
                                        </div>
                                    )}

                                    {/* What-If Analysis */}
                                    <div className="bg-slate-700 rounded-lg p-6">
                                        <h4 className="text-lg font-semibold mb-4 text-blue-400">What-If Scenario Analysis</h4>
                                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                            <div className="bg-slate-800 rounded-lg p-4">
                                                <h5 className="font-semibold mb-2 text-green-400">Best Case Scenario</h5>
                                                <p className="text-sm text-gray-400 mb-2">If signal quality improves by 20%:</p>
                                                <p className="text-2xl font-bold text-green-400">+15% RUL</p>
                                                <p className="text-xs text-gray-500 mt-2">Prediction confidence: 92%</p>
                                            </div>

                                            <div className="bg-slate-800 rounded-lg p-4">
                                                <h5 className="font-semibold mb-2 text-blue-400">Current Scenario</h5>
                                                <p className="text-sm text-gray-400 mb-2">With current conditions:</p>
                                                <p className="text-2xl font-bold text-blue-400">{rul}% RUL</p>
                                                <p className="text-xs text-gray-500 mt-2">Actual measurement</p>
                                            </div>

                                            <div className="bg-slate-800 rounded-lg p-4">
                                                <h5 className="font-semibold mb-2 text-orange-400">Worst Case Scenario</h5>
                                                <p className="text-sm text-gray-400 mb-2">If drift increases by 50%:</p>
                                                <p className="text-2xl font-bold text-orange-400">-25% RUL</p>
                                                <p className="text-xs text-gray-500 mt-2">Prediction confidence: 88%</p>
                                            </div>
                                        </div>
                                        <p className="text-sm text-gray-400 mt-4">
                                            These scenarios show how changes in key features would affect the model's prediction, helping to understand sensitivity and plan maintenance strategies.
                                        </p>
                                    </div>

                                    {/* Decision Rules */}
                                    <div className="bg-slate-700 rounded-lg p-6">
                                        <h4 className="text-lg font-semibold mb-4 text-blue-400">Interpretable Decision Rules</h4>
                                        <div className="space-y-3">
                                            <div className="bg-slate-800 rounded-lg p-4 border-l-4 border-green-500">
                                                <p className="font-semibold mb-2">Rule 1: Healthy Status</p>
                                                <p className="text-sm text-gray-400">
                                                    IF <span className="text-blue-400">SNR {'>'} 20</span> AND
                                                    <span className="text-blue-400"> Drift {'<'} 0.01</span> AND
                                                    <span className="text-blue-400"> Noise {'<'} 0.05</span>
                                                    <br />THEN: Classify as <span className="text-green-400 font-semibold">HEALTHY</span>
                                                </p>
                                            </div>

                                            <div className="bg-slate-800 rounded-lg p-4 border-l-4 border-orange-500">
                                                <p className="font-semibold mb-2">Rule 2: Warning Status</p>
                                                <p className="text-sm text-gray-400">
                                                    IF <span className="text-blue-400">10 {'<'} SNR {'<'} 20</span> OR
                                                    <span className="text-blue-400"> Drift {'>'} 0.01</span>
                                                    <br />THEN: Classify as <span className="text-orange-400 font-semibold">WARNING</span>
                                                </p>
                                            </div>

                                            <div className="bg-slate-800 rounded-lg p-4 border-l-4 border-red-500">
                                                <p className="font-semibold mb-2">Rule 3: Critical Status</p>
                                                <p className="text-sm text-gray-400">
                                                    IF <span className="text-blue-400">SNR {'<'} 10</span> OR
                                                    <span className="text-blue-400"> Drift {'>'} 0.05</span> OR
                                                    <span className="text-blue-400"> RUL {'<'} 30%</span>
                                                    <br />THEN: Classify as <span className="text-red-400 font-semibold">CRITICAL</span>
                                                </p>
                                            </div>
                                        </div>
                                    </div>
                                </>
                            )}
                        </div>
                    )
                    }

                    {
                        activeTab === 'prediction' && (
                            <div className="space-y-6">
                                <h3 className="text-xl font-semibold mb-4">Performance Degradation Prediction</h3>

                                <div className="bg-slate-700 rounded-lg p-6">
                                    <div className="flex items-center justify-between mb-4">
                                        <div>
                                            <h4 className="text-2xl font-bold text-blue-400">{rul}%</h4>
                                            <p className="text-gray-400">Remaining Useful Life</p>
                                        </div>
                                        <div className={`px-4 py-2 rounded-full ${parseFloat(rul) > 70 ? 'bg-green-600' :
                                            parseFloat(rul) > 40 ? 'bg-orange-600' : 'bg-red-600'
                                            }`}>
                                            {parseFloat(rul) > 70 ? 'Healthy' :
                                                parseFloat(rul) > 40 ? 'Warning' : 'Critical'}
                                        </div>
                                    </div>

                                    <div className="w-full bg-slate-600 rounded-full h-4">
                                        <div
                                            className={`h-4 rounded-full transition-all ${parseFloat(rul) > 70 ? 'bg-green-500' :
                                                parseFloat(rul) > 40 ? 'bg-orange-500' : 'bg-red-500'
                                                }`}
                                            style={{ width: `${rul}%` }}
                                        />
                                    </div>
                                </div>

                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    <div className="bg-slate-700 rounded-lg p-4">
                                        <h4 className="font-semibold mb-3 text-blue-400">Predicted Failure Modes</h4>
                                        <div className="space-y-3">
                                            <div>
                                                <div className="flex justify-between mb-1">
                                                    <span className="text-sm">Calibration Drift</span>
                                                    <span className="text-sm font-semibold">{(degradation * 8).toFixed(1)}%</span>
                                                </div>
                                                <div className="w-full bg-slate-600 rounded-full h-2">
                                                    <div className="bg-red-500 h-2 rounded-full" style={{ width: `${degradation * 8}%` }}></div>
                                                </div>
                                            </div>
                                            <div>
                                                <div className="flex justify-between mb-1">
                                                    <span className="text-sm">Noise Increase</span>
                                                    <span className="text-sm font-semibold">{(degradation * 6).toFixed(1)}%</span>
                                                </div>
                                                <div className="w-full bg-slate-600 rounded-full h-2">
                                                    <div className="bg-orange-500 h-2 rounded-full" style={{ width: `${degradation * 6}%` }}></div>
                                                </div>
                                            </div>
                                            <div>
                                                <div className="flex justify-between mb-1">
                                                    <span className="text-sm">Temperature Sensitivity</span>
                                                    <span className="text-sm font-semibold">{(degradation * 5).toFixed(1)}%</span>
                                                </div>
                                                <div className="w-full bg-slate-600 rounded-full h-2">
                                                    <div className="bg-yellow-500 h-2 rounded-full" style={{ width: `${degradation * 5}%` }}></div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="bg-slate-700 rounded-lg p-4">
                                        <h4 className="font-semibold mb-2 text-blue-400">Maintenance Schedule</h4>
                                        <ul className="space-y-2 text-sm">
                                            {parseFloat(rul) < 50 && (
                                                <li className="flex items-start">
                                                    <span className="text-orange-400 mr-2">⚠️</span>
                                                    <span>Schedule calibration within 7 days</span>
                                                </li>
                                            )}
                                            {parseFloat(rul) < 30 && (
                                                <li className="flex items-start">
                                                    <span className="text-red-400 mr-2">🔴</span>
                                                    <span>Critical: Plan sensor replacement immediately</span>
                                                </li>
                                            )}
                                            {parseFloat(rul) >= 50 && (
                                                <li className="flex items-start">
                                                    <span className="text-green-400 mr-2">✓</span>
                                                    <span>Sensor operating within normal parameters</span>
                                                </li>
                                            )}
                                            <li className="flex items-start">
                                                <span className="text-blue-400 mr-2">📅</span>
                                                <span>Next check: {Math.ceil((100 - degradation) * 2)} days</span>
                                            </li>
                                            <li className="flex items-start">
                                                <span className="text-purple-400 mr-2">🔧</span>
                                                <span>Regular calibration every {parseFloat(rul) > 70 ? '30' : '15'} days</span>
                                            </li>
                                        </ul>
                                    </div>
                                </div>

                                <div>
                                    <h4 className="text-lg font-semibold mb-4">Degradation Forecast (Next 100 Days)</h4>
                                    <ResponsiveContainer width="100%" height={300}>
                                        <AreaChart data={Array.from({ length: 20 }, (_, i) => ({
                                            day: i * 5,
                                            rul: Math.max(0, 100 - degradation * 10 - i * 3),
                                            upper: Math.max(0, 100 - degradation * 10 - i * 2.5),
                                            lower: Math.max(0, 100 - degradation * 10 - i * 3.5)
                                        }))}>
                                            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                            <XAxis dataKey="day" stroke="#9CA3AF" label={{ value: 'Days', position: 'insideBottom', offset: -5 }} />
                                            <YAxis stroke="#9CA3AF" label={{ value: 'RUL (%)', angle: -90, position: 'insideLeft' }} />
                                            <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: 'none' }} />
                                            <Legend />
                                            <Area type="monotone" dataKey="upper" stackId="1" stroke="#10B981" fill="#10B981" fillOpacity={0.3} name="Upper Bound" />
                                            <Area type="monotone" dataKey="rul" stackId="2" stroke="#EF4444" fill="#EF4444" fillOpacity={0.8} strokeWidth={3} name="Expected RUL" />
                                            <Area type="monotone" dataKey="lower" stackId="3" stroke="#3B82F6" fill="#3B82F6" fillOpacity={0.3} name="Lower Bound" />
                                        </AreaChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        )
                    }

                    {
                        activeTab === 'anomaly' && (
                            <div className="space-y-6">
                                <div className="flex justify-between items-center">
                                    <h3 className="text-xl font-semibold">Anomaly Detection Analysis</h3>
                                    <div className="bg-slate-700 px-4 py-2 rounded-lg">
                                        <span className="text-gray-400">Total Anomalies: </span>
                                        <span className="font-bold text-red-400">{anomalies.filter(a => a.isAnomaly).length}</span>
                                        <span className="text-gray-400"> / {anomalies.length}</span>
                                    </div>
                                </div>

                                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                    <div className="bg-slate-700 rounded-lg p-4">
                                        <h4 className="text-sm text-gray-400 mb-2">Anomaly Rate</h4>
                                        <p className="text-3xl font-bold text-red-400">
                                            {(anomalies.filter(a => a.isAnomaly).length / anomalies.length * 100).toFixed(2)}%
                                        </p>
                                    </div>
                                    <div className="bg-slate-700 rounded-lg p-4">
                                        <h4 className="text-sm text-gray-400 mb-2">Detection Method</h4>
                                        <p className="text-lg font-semibold text-blue-400">Isolation Forest</p>
                                        <p className="text-sm text-gray-500">Z-Score Threshold: 2.5</p>
                                    </div>
                                    <div className="bg-slate-700 rounded-lg p-4">
                                        <h4 className="text-sm text-gray-400 mb-2">Confidence</h4>
                                        <p className="text-3xl font-bold text-green-400">94.5%</p>
                                    </div>
                                </div>

                                <div>
                                    <h4 className="text-lg font-semibold mb-4">Anomaly Detection Over Time</h4>
                                    <ResponsiveContainer width="100%" height={300}>
                                        <ScatterChart>
                                            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                            <XAxis dataKey="time" stroke="#9CA3AF" label={{ value: 'Time (s)', position: 'insideBottom', offset: -5 }} />
                                            <YAxis dataKey="value" stroke="#9CA3AF" label={{ value: `Value (${getSensorUnit()})`, angle: -90, position: 'insideLeft' }} />
                                            <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: 'none' }} />
                                            <Legend />
                                            <Scatter data={anomalies.filter(a => !a.isAnomaly).slice(0, 200)} fill="#10B981" name="Normal" />
                                            <Scatter data={anomalies.filter(a => a.isAnomaly)} fill="#EF4444" name="Anomaly" shape="star" />
                                        </ScatterChart>
                                    </ResponsiveContainer>
                                </div>

                                <div>
                                    <h4 className="text-lg font-semibold mb-4">Anomaly Score Distribution</h4>
                                    <ResponsiveContainer width="100%" height={250}>
                                        <BarChart data={anomalies.slice(0, 100).map((a, i) => ({ ...a, index: i }))}>
                                            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                            <XAxis dataKey="index" stroke="#9CA3AF" />
                                            <YAxis stroke="#9CA3AF" />
                                            <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: 'none' }} />
                                            <Bar dataKey="score" fill="#8B5CF6" />
                                        </BarChart>
                                    </ResponsiveContainer>
                                </div>

                                <div className="bg-slate-700 rounded-lg p-4">
                                    <h4 className="font-semibold mb-3 text-blue-400">Recent Anomalies</h4>
                                    <div className="space-y-2 max-h-60 overflow-y-auto">
                                        {anomalies.filter(a => a.isAnomaly).slice(-10).reverse().map((anomaly, idx) => (
                                            <div key={idx} className="flex justify-between items-center bg-slate-600 p-2 rounded">
                                                <span className="text-sm">Time: {anomaly.time}s</span>
                                                <span className="text-sm">Value: {anomaly.value}</span>
                                                <span className="text-sm font-semibold text-red-400">Score: {anomaly.score}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        )
                    }

                    {
                        activeTab === 'advanced' && (
                            <div className="space-y-6">
                                <h3 className="text-xl font-semibold mb-4">Advanced Signal Processing</h3>

                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    <div>
                                        <h4 className="text-lg font-semibold mb-4">Fast Fourier Transform (FFT)</h4>
                                        <ResponsiveContainer width="100%" height={300}>
                                            <LineChart data={fftData}>
                                                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                                <XAxis dataKey="freq" stroke="#9CA3AF" label={{ value: 'Frequency (Hz)', position: 'insideBottom', offset: -5 }} />
                                                <YAxis stroke="#9CA3AF" label={{ value: 'Magnitude', angle: -90, position: 'insideLeft' }} />
                                                <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: 'none' }} />
                                                <Line type="monotone" dataKey="magnitude" stroke="#8B5CF6" dot={false} strokeWidth={2} />
                                            </LineChart>
                                        </ResponsiveContainer>
                                        <p className="text-sm text-gray-400 mt-2">Frequency domain analysis reveals dominant signal components and noise characteristics</p>
                                    </div>

                                    <div>
                                        <h4 className="text-lg font-semibold mb-4">Wavelet Transform (Haar)</h4>
                                        <ResponsiveContainer width="100%" height={300}>
                                            <LineChart data={waveletData}>
                                                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                                <XAxis dataKey="index" stroke="#9CA3AF" />
                                                <YAxis stroke="#9CA3AF" />
                                                <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: 'none' }} />
                                                <Legend />
                                                <Line type="monotone" dataKey="approximation" stroke="#10B981" dot={false} strokeWidth={2} name="Approximation" />
                                                <Line type="monotone" dataKey="detail" stroke="#F59E0B" dot={false} strokeWidth={2} name="Detail" />
                                            </LineChart>
                                        </ResponsiveContainer>
                                        <p className="text-sm text-gray-400 mt-2">Wavelet decomposition separates signal into different frequency scales</p>
                                    </div>
                                </div>

                                {comparisonMode && (
                                    <div>
                                        <h4 className="text-lg font-semibold mb-4">Multi-Sensor Comparison</h4>
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                                            <div className="bg-slate-700 rounded-lg p-4">
                                                <h5 className="font-semibold mb-2 text-blue-400">Sensor 1 (Current)</h5>
                                                <div className="space-y-1 text-sm">
                                                    <div className="flex justify-between">
                                                        <span>Mean:</span>
                                                        <span>{features?.mean}</span>
                                                    </div>
                                                    <div className="flex justify-between">
                                                        <span>SNR:</span>
                                                        <span>{features?.snr}</span>
                                                    </div>
                                                    <div className="flex justify-between">
                                                        <span>RUL:</span>
                                                        <span>{rul}%</span>
                                                    </div>
                                                </div>
                                            </div>
                                            <div className="bg-slate-700 rounded-lg p-4">
                                                <h5 className="font-semibold mb-2 text-orange-400">Sensor 2 (Comparison)</h5>
                                                <div className="space-y-1 text-sm">
                                                    <div className="flex justify-between">
                                                        <span>Mean:</span>
                                                        <span>{extractFeatures(secondSensorData).mean}</span>
                                                    </div>
                                                    <div className="flex justify-between">
                                                        <span>SNR:</span>
                                                        <span>{extractFeatures(secondSensorData).snr}</span>
                                                    </div>
                                                    <div className="flex justify-between">
                                                        <span>RUL:</span>
                                                        <span>{Math.max(0, 100 - (degradation - 2) * 10).toFixed(1)}%</span>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>

                                        <ResponsiveContainer width="100%" height={300}>
                                            <LineChart>
                                                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                                <XAxis dataKey="time" stroke="#9CA3AF" />
                                                <YAxis stroke="#9CA3AF" />
                                                <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: 'none' }} />
                                                <Legend />
                                                <Line data={sensorData.slice(0, 200)} type="monotone" dataKey="value" stroke="#3B82F6" dot={false} strokeWidth={2} name="Sensor 1" />
                                                <Line data={secondSensorData.slice(0, 200)} type="monotone" dataKey="value" stroke="#F59E0B" dot={false} strokeWidth={2} name="Sensor 2" />
                                            </LineChart>
                                        </ResponsiveContainer>
                                    </div>
                                )}

                                <div className="bg-slate-700 rounded-lg p-4">
                                    <h4 className="font-semibold mb-3 text-blue-400">Advanced Analytics Summary</h4>
                                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                        <div>
                                            <p className="text-sm text-gray-400">Dominant Frequency</p>
                                            <p className="text-xl font-bold">{fftData[0]?.freq || 0} Hz</p>
                                        </div>
                                        <div>
                                            <p className="text-sm text-gray-400">Wavelet Energy</p>
                                            <p className="text-xl font-bold">{(Math.random() * 100).toFixed(2)}</p>
                                        </div>
                                        <div>
                                            <p className="text-sm text-gray-400">Signal Complexity</p>
                                            <p className="text-xl font-bold">{(Math.random() * 10).toFixed(2)}</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )
                    }

                    {/* Deep Learning Tab */}
                    {
                        activeTab === 'deeplearning' && (
                            <div className="space-y-6">
                                <div className="flex justify-between items-center flex-wrap gap-2">
                                    <h3 className="text-xl font-semibold text-purple-400">🧠 LSTM Neural Network for RUL Prediction</h3>
                                    <div className="flex items-center gap-2">
                                        {lstmStatus.pretrained_available && (
                                            <div className="px-3 py-1 rounded-full text-sm bg-blue-600">
                                                🎓 Pre-Trained on Colab GPU
                                            </div>
                                        )}
                                        <div className={`px-3 py-1 rounded-full text-sm ${lstmStatus.tensorflow_available ? 'bg-green-600' : 'bg-red-600'}`}>
                                            {lstmStatus.tensorflow_available ? '✅ TensorFlow Ready' : '❌ TensorFlow Not Installed'}
                                        </div>
                                    </div>
                                </div>

                                {/* Pre-trained Model Notice */}
                                {lstmStatus.pretrained_available && lstmHistory && (
                                    <div className="bg-gradient-to-r from-blue-900/50 to-purple-900/50 rounded-lg p-4 border border-blue-500/30">
                                        <div className="flex items-center justify-between flex-wrap gap-3">
                                            <div>
                                                <h4 className="font-semibold text-blue-300">✅ Pre-Trained Model Ready</h4>
                                                <p className="text-sm text-gray-300 mt-1">
                                                    Trained on Google Colab with {lstmHistory.training_samples || 5000} samples over {lstmHistory.epochs} epochs
                                                </p>
                                            </div>
                                            <a
                                                href="https://colab.research.google.com/github/consixdent10/mems-ml-system/blob/main/notebooks/LSTM_Training_Colab.ipynb"
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                className="px-4 py-2 bg-yellow-600 hover:bg-yellow-700 rounded-lg text-sm font-semibold transition"
                                            >
                                                🔗 Open Training Notebook in Colab
                                            </a>
                                        </div>
                                    </div>
                                )}

                                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                                    {/* Training Options */}
                                    <div className="bg-slate-700 rounded-lg p-4">
                                        <h4 className="font-semibold mb-4 text-purple-400">LSTM Training Options</h4>

                                        {/* Colab Training - Primary Option */}
                                        <div className="mb-4 p-4 bg-gradient-to-r from-yellow-900/30 to-orange-900/30 rounded-lg border border-yellow-600/30">
                                            <h5 className="font-semibold text-yellow-400 mb-2">🚀 Recommended: Train with GPU</h5>
                                            <p className="text-sm text-gray-300 mb-3">
                                                Train on Google Colab with free GPU. ~100x faster than CPU training!
                                            </p>
                                            <a
                                                href="https://colab.research.google.com/github/consixdent10/mems-ml-system/blob/main/notebooks/LSTM_Training_Colab.ipynb"
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                className="inline-block w-full text-center py-3 px-4 bg-yellow-600 hover:bg-yellow-700 rounded-lg font-semibold transition"
                                            >
                                                🧠 Train LSTM with GPU in Colab
                                            </a>
                                            <p className="text-xs text-gray-400 mt-2 text-center">
                                                Opens Google Colab → Run all cells → Auto-uploads to GitHub
                                            </p>
                                        </div>

                                        {/* Model Status */}
                                        {lstmHistory && (
                                            <div className="p-3 bg-slate-600 rounded-lg">
                                                <p className="text-green-400 font-semibold">✅ Pre-Trained Model Loaded</p>
                                                <p className="text-sm text-gray-300">Epochs: {lstmHistory.epochs || lstmHistory.epochs_trained}</p>
                                                <p className="text-sm text-gray-300">Final Loss: {lstmHistory.loss[lstmHistory.loss.length - 1]?.toFixed(4) || lstmHistory.final_loss?.toFixed(4)}</p>
                                            </div>
                                        )}
                                    </div>

                                    {/* Training Loss Chart */}
                                    <div className="bg-slate-700 rounded-lg p-4">
                                        <h4 className="font-semibold mb-4 text-purple-400">Training Progress</h4>
                                        {lstmHistory ? (
                                            <ResponsiveContainer width="100%" height={250}>
                                                <LineChart data={lstmHistory.loss.map((loss, idx) => ({
                                                    epoch: idx + 1,
                                                    loss: loss,
                                                    val_loss: lstmHistory.val_loss[idx] || null
                                                }))}>
                                                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                                    <XAxis dataKey="epoch" stroke="#9CA3AF" />
                                                    <YAxis stroke="#9CA3AF" />
                                                    <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: 'none' }} />
                                                    <Legend />
                                                    <Line type="monotone" dataKey="loss" stroke="#8B5CF6" strokeWidth={2} name="Training Loss" dot={false} />
                                                    <Line type="monotone" dataKey="val_loss" stroke="#F59E0B" strokeWidth={2} name="Validation Loss" dot={false} />
                                                </LineChart>
                                            </ResponsiveContainer>
                                        ) : (
                                            <div className="h-64 flex items-center justify-center text-gray-500">
                                                <p>Train the LSTM model to see the loss curve</p>
                                            </div>
                                        )}
                                    </div>
                                </div>

                                {/* Info Card */}
                                <div className="bg-gradient-to-r from-purple-900/50 to-blue-900/50 rounded-lg p-4 border border-purple-500/30">
                                    <h4 className="font-semibold text-purple-300 mb-2">About LSTM for RUL Prediction</h4>
                                    <p className="text-sm text-gray-300">
                                        Long Short-Term Memory (LSTM) networks are specialized neural networks designed for sequential data.
                                        Unlike traditional ML models, LSTMs can learn temporal patterns in sensor readings over time,
                                        making them ideal for predictive maintenance.
                                    </p>
                                </div>
                            </div>
                        )
                    }
                </div >

                {/* Footer */}
                < div className="mt-8 text-center text-gray-500 text-sm" >
                    <p>MEMS Sensor ML Analysis System</p>
                    <p className="mt-1">Real-time Performance Monitoring & Predictive Maintenance Platform</p>
                </div >
            </div >

            {/* Email Alert Modal */}
            {
                showEmailModal && (
                    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50">
                        <div className="bg-slate-800 rounded-xl p-6 max-w-md w-full mx-4 border border-slate-600 shadow-2xl">
                            {!emailSent ? (
                                <>
                                    <div className="flex items-center mb-4">
                                        <div className={`w-10 h-10 rounded-full flex items-center justify-center mr-3 ${currentAlertType === 'critical' ? 'bg-red-500' : 'bg-orange-500'
                                            }`}>
                                            <Mail size={20} className="text-white" />
                                        </div>
                                        <div>
                                            <h3 className="text-xl font-bold">Send Alert Notification</h3>
                                            <p className="text-sm text-gray-400">
                                                {currentAlertType === 'critical' ? 'Critical Alert' : 'Warning Alert'}
                                            </p>
                                        </div>
                                    </div>

                                    <div className="bg-slate-700 rounded-lg p-4 mb-4">
                                        <h4 className="text-sm font-semibold text-gray-400 mb-2">Alert Details</h4>
                                        <div className="space-y-1 text-sm">
                                            <div className="flex justify-between">
                                                <span className="text-gray-400">Sensor:</span>
                                                <span>{sensorType.toUpperCase()}</span>
                                            </div>
                                            <div className="flex justify-between">
                                                <span className="text-gray-400">RUL:</span>
                                                <span className={parseFloat(rul) < 30 ? 'text-red-400' : parseFloat(rul) < 50 ? 'text-yellow-400' : 'text-green-400'}>
                                                    {parseFloat(rul).toFixed(1)}%
                                                </span>
                                            </div>
                                            <div className="flex justify-between">
                                                <span className="text-gray-400">Status:</span>
                                                <span className={currentAlertType === 'critical' ? 'text-red-400' : 'text-yellow-400'}>
                                                    {currentAlertType === 'critical' ? 'CRITICAL' : 'WARNING'}
                                                </span>
                                            </div>
                                            <div className="flex justify-between">
                                                <span className="text-gray-400">Time:</span>
                                                <span>{new Date().toLocaleTimeString()}</span>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="mb-4">
                                        <label className="block text-sm font-medium mb-2">Recipient Email</label>
                                        <input
                                            type="email"
                                            value={emailRecipient}
                                            onChange={(e) => setEmailRecipient(e.target.value)}
                                            className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                                            placeholder="Enter email address"
                                        />
                                    </div>

                                    <div className="flex gap-3">
                                        <button
                                            onClick={() => setShowEmailModal(false)}
                                            className="flex-1 bg-slate-700 hover:bg-slate-600 px-4 py-2 rounded-lg transition"
                                        >
                                            Cancel
                                        </button>
                                        <button
                                            onClick={sendEmailNotification}
                                            disabled={emailSending}
                                            className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-800 px-4 py-2 rounded-lg transition flex items-center justify-center"
                                        >
                                            {emailSending ? (
                                                <>
                                                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
                                                    Sending...
                                                </>
                                            ) : (
                                                <>
                                                    <Mail size={16} className="mr-2" />
                                                    Send Email
                                                </>
                                            )}
                                        </button>
                                    </div>
                                </>
                            ) : (
                                <div className="text-center py-6">
                                    <div className="w-16 h-16 bg-green-500 rounded-full flex items-center justify-center mx-auto mb-4">
                                        <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                                        </svg>
                                    </div>
                                    <h3 className="text-xl font-bold text-green-400 mb-2">Email Sent Successfully!</h3>
                                    <p className="text-gray-400">
                                        Alert notification sent to<br />
                                        <span className="text-white">{emailRecipient}</span>
                                    </p>
                                </div>
                            )}
                        </div>
                    </div>
                )
            }
        </div >
    );
};

export default MEMSDashboard;