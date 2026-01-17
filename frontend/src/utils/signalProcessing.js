/**
 * Signal Processing Utilities
 * FFT and Wavelet Transform implementations
 */

/**
 * Perform FFT on sensor data
 * @param {Array} data - Array of sensor data points with 'value' property
 * @returns {Object} - frequencies and magnitudes arrays
 */
export const performFFT = (data) => {
    const values = data.map(d => parseFloat(d.value));
    const N = values.length;
    const frequencies = [];
    const magnitudes = [];

    for (let k = 0; k < N / 2; k++) {
        let real = 0, imag = 0;
        for (let n = 0; n < N; n++) {
            const angle = (2 * Math.PI * k * n) / N;
            real += values[n] * Math.cos(angle);
            imag -= values[n] * Math.sin(angle);
        }
        const magnitude = Math.sqrt(real * real + imag * imag) / N;
        frequencies.push({ freq: (k / N * 100).toFixed(2), magnitude: magnitude.toFixed(4) });
        magnitudes.push(magnitude);
    }

    return { frequencies: frequencies.slice(0, 50), magnitudes };
};

/**
 * Perform Haar Wavelet Transform on sensor data
 * @param {Array} data - Array of sensor data points with 'value' property
 * @returns {Array} - Wavelet coefficients with approximation and detail
 */
export const waveletTransform = (data) => {
    // Simplified Haar wavelet transform
    const values = data.map(d => parseFloat(d.value));
    const coefficients = [];

    for (let i = 0; i < values.length - 1; i += 2) {
        const avg = (values[i] + values[i + 1]) / 2;
        const diff = (values[i] - values[i + 1]) / 2;
        coefficients.push({ index: i / 2, approximation: avg.toFixed(4), detail: diff.toFixed(4) });
    }

    return coefficients.slice(0, 50);
};
