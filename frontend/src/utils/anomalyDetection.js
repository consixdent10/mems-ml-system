/**
 * Anomaly Detection Utilities
 * Rolling Z-score based anomaly detection with configurable parameters
 */

/**
 * Detect anomalies using rolling window Z-score method
 * @param {Array} data - Array of sensor data points with 'value' and 'time' properties
 * @param {number} windowSize - Rolling window size for computing local statistics (default: 50)
 * @param {number} threshold - Z-score threshold for anomaly detection (default: 2.5)
 * @returns {Array} - Array of data points with anomaly flags and scores
 */
export const detectAnomalies = (data, windowSize = 50, threshold = 2.5) => {
    if (!data || data.length < 2) {
        return [];
    }

    const values = data.map(d => parseFloat(d.value) || 0);
    const anomalies = [];

    for (let i = 0; i < values.length; i++) {
        // Determine window range (use available points if less than windowSize)
        const windowStart = Math.max(0, i - windowSize + 1);
        const windowEnd = i + 1;
        const window = values.slice(windowStart, windowEnd);

        // Compute rolling mean
        const rollingMean = window.reduce((a, b) => a + b, 0) / window.length;

        // Compute rolling standard deviation
        const rollingStd = Math.sqrt(
            window.reduce((a, b) => a + Math.pow(b - rollingMean, 2), 0) / window.length
        );

        // Compute z-score (handle zero std)
        const zScore = rollingStd > 1e-8
            ? Math.abs((values[i] - rollingMean) / rollingStd)
            : 0;

        const isAnomaly = zScore > threshold;

        anomalies.push({
            time: data[i].time,
            value: data[i].value,
            isAnomaly,
            score: zScore.toFixed(2),
            anomalyScore: parseFloat(zScore.toFixed(4)),
            rollingMean: rollingMean.toFixed(4),
            rollingStd: rollingStd.toFixed(4),
            threshold: threshold
        });
    }

    return anomalies;
};

/**
 * Get anomaly statistics from detected anomalies
 * @param {Array} anomalies - Array of anomaly detection results
 * @returns {Object} - Statistics about anomalies
 */
export const getAnomalyStats = (anomalies) => {
    if (!anomalies || anomalies.length === 0) {
        return {
            total: 0,
            anomalyCount: 0,
            anomalyRate: '0.00',
            recentAnomalies: [],
            maxScore: 0,
            avgScore: 0
        };
    }

    const anomalyList = anomalies.filter(a => a.isAnomaly);
    const scores = anomalies.map(a => parseFloat(a.score) || 0);
    const maxScore = Math.max(...scores);
    const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;

    return {
        total: anomalies.length,
        anomalyCount: anomalyList.length,
        anomalyRate: ((anomalyList.length / anomalies.length) * 100).toFixed(2),
        recentAnomalies: anomalyList.slice(-5).reverse(),
        maxScore: maxScore.toFixed(2),
        avgScore: avgScore.toFixed(2)
    };
};
