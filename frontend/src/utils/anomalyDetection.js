/**
 * Anomaly Detection Utilities
 * Z-score based anomaly detection (Isolation Forest approach)
 */

/**
 * Detect anomalies in sensor data using z-score method
 * @param {Array} data - Array of sensor data points with 'value' and 'time' properties
 * @param {number} threshold - Z-score threshold for anomaly detection (default: 2.5)
 * @returns {Array} - Array of data points with anomaly flags
 */
export const detectAnomalies = (data, threshold = 2.5) => {
    const values = data.map(d => parseFloat(d.value));
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const std = Math.sqrt(values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length);

    const anomalies = data.map((d, i) => {
        const zScore = Math.abs((parseFloat(d.value) - mean) / std);
        return {
            time: d.time,
            value: d.value,
            isAnomaly: zScore > threshold,
            score: zScore.toFixed(2)
        };
    });

    return anomalies;
};

/**
 * Get anomaly statistics from detected anomalies
 * @param {Array} anomalies - Array of anomaly detection results
 * @returns {Object} - Statistics about anomalies
 */
export const getAnomalyStats = (anomalies) => {
    const anomalyList = anomalies.filter(a => a.isAnomaly);
    return {
        total: anomalies.length,
        anomalyCount: anomalyList.length,
        anomalyRate: ((anomalyList.length / anomalies.length) * 100).toFixed(2),
        recentAnomalies: anomalyList.slice(-5).reverse()
    };
};
