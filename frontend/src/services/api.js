// API Configuration - Use environment variable for production, localhost for development
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

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
        if (!response.ok) throw new Error('Failed to train models');
        return response.json();
    },

    getXAIAnalysis: async (sensorData) => {
        const response = await fetch(`${API_BASE_URL}/api/xai-analysis`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sensor_data: sensorData })
        });
        if (!response.ok) throw new Error('Failed to get XAI analysis');
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

    // LSTM Status
    getLstmStatus: async () => {
        const response = await fetch(`${API_BASE_URL}/api/lstm-status`);
        if (!response.ok) throw new Error('Failed to get LSTM status');
        return response.json();
    },

    // LSTM Predict
    lstmPredict: async (features) => {
        const response = await fetch(`${API_BASE_URL}/api/lstm-predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features })
        });
        if (!response.ok) throw new Error('Failed to get LSTM prediction');
        return response.json();
    }
};

export { API_BASE_URL };
export default api;
