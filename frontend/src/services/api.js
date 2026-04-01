// API Configuration - Use environment variable for production, localhost for development
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Dev-only logging and production warning
if (import.meta.env.DEV) {
    console.log('[API] Base URL:', API_BASE_URL);
}
if (!import.meta.env.VITE_API_URL && typeof window !== 'undefined' && window.location.hostname !== 'localhost') {
    console.warn('[API] VITE_API_URL is missing in production. API calls will fail.');
}


// API Helper Functions
export const api = {
    generateData: async (sensorType, degradationLevel) => {
        const response = await fetch(`${API_BASE_URL}/api/generate-data`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                sensor_type: sensorType,
                num_samples: 1000,
                degradation_level: degradationLevel
            })
        });
        if (!response.ok) throw new Error('Failed to generate data');
        return response.json();
    },

    uploadData: async (file) => {
        const formData = new FormData();
        formData.append('file', file);
        const response = await fetch(`${API_BASE_URL}/api/upload-data`, {
            method: 'POST',
            body: formData
        });
        if (!response.ok) throw new Error('Failed to upload data');
        return response.json();
    },

    trainModels: async (sensorData) => {
        const response = await fetch(`${API_BASE_URL}/api/train-models`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sensor_data: sensorData })
        });
        if (!response.ok) {
            const errText = await response.text();
            throw new Error(`trainModels failed: ${response.status} ${errText}`);
        }
        return response.json();
    },

    getXAIAnalysis: async (sensorData) => {
        const response = await fetch(`${API_BASE_URL}/api/xai-analysis`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sensor_data: sensorData })
        });
        if (!response.ok) {
            const errText = await response.text();
            throw new Error(`getXAIAnalysis failed: ${response.status} ${errText}`);
        }
        return response.json();
    },

    // Real Datasets API
    listDatasets: async () => {
        const response = await fetch(`${API_BASE_URL}/api/datasets`);
        if (!response.ok) throw new Error('Failed to list datasets');
        return response.json();
    },

    loadDataset: async (datasetId, options = {}) => {
        const response = await fetch(`${API_BASE_URL}/api/datasets/load`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                dataset_id: datasetId,
                degradation_stage: options.degradation_stage || 0,
                fault_type: options.fault_type || 'normal',
                scenario: options.scenario || 'normal'
            })
        });
        if (!response.ok) throw new Error('Failed to load dataset');
        return response.json();
    },


    // Download Best Model
    downloadBestModel: async () => {
        const response = await fetch(`${API_BASE_URL}/api/download-best-model`);
        if (!response.ok) {
            if (response.status === 404) {
                throw new Error('No model found. Train models first.');
            }
            throw new Error('Failed to download model');
        }
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'best_model.joblib';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        return { success: true };
    },

    // Health Report (unified RUL, status, risks, forecast)
    healthReport: async (sensorData = null, degradationLevel = null) => {
        const response = await fetch(`${API_BASE_URL}/api/health-report`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                sensor_data: sensorData,
                degradation_level: degradationLevel
            })
        });
        if (!response.ok) throw new Error('Failed to get health report');
        return response.json();
    },

    // FFT Analysis (using numpy.fft on backend for O(N log N) performance)
    computeFFT: async (sensorData, sampleRate = null) => {
        const response = await fetch(`${API_BASE_URL}/api/signal/fft`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                sensor_data: sensorData,
                sample_rate: sampleRate
            })
        });
        if (!response.ok) throw new Error('Failed to compute FFT');
        return response.json();
    }
};

export { API_BASE_URL };
export default api;
