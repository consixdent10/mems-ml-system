/**
 * Signal Processing Utilities
 * FFT and Wavelet Transform implementations with proper DC removal and windowing
 */

/**
 * Apply Hann window to reduce spectral leakage
 * @param {number} i - Sample index
 * @param {number} N - Total samples
 * @returns {number} - Window coefficient
 */
const hannWindow = (i, N) => {
    return 0.5 * (1 - Math.cos((2 * Math.PI * i) / (N - 1)));
};

/**
 * Perform FFT on sensor data with proper preprocessing
 * Includes: DC removal, Hann window, correct frequency axis
 * 
 * @param {Array} data - Array of sensor data points with 'value' and optional 'time' property
 * @param {number} sampleRate - Optional sample rate in Hz (default: inferred or 100 Hz)
 * @returns {Object} - frequencies array, magnitudes, dominantFrequency, sampleRate
 */
export const performFFT = (data, sampleRate = null) => {
    if (!data || data.length < 4) {
        return {
            frequencies: [],
            magnitudes: [],
            dominantFrequency: 0,
            sampleRate: 100
        };
    }

    const values = data.map(d => parseFloat(d.value) || 0);
    const N = values.length;

    // Infer sample rate from time column if available
    let fs = sampleRate;
    if (!fs && data[0]?.time !== undefined && data[1]?.time !== undefined) {
        const dt = parseFloat(data[1].time) - parseFloat(data[0].time);
        if (dt > 0) {
            fs = 1 / dt;
        }
    }
    // Fallback to reasonable default (100 Hz)
    if (!fs || fs <= 0 || !isFinite(fs)) {
        fs = 100;
    }

    // 1) Remove DC offset (mean subtraction)
    const mean = values.reduce((sum, v) => sum + v, 0) / N;
    const dcRemoved = values.map(v => v - mean);

    // 2) Apply Hann window to reduce spectral leakage
    const windowed = dcRemoved.map((v, i) => v * hannWindow(i, N));

    // 3) Compute FFT (DFT for simplicity, only first half - Nyquist)
    const frequencies = [];
    const magnitudes = [];
    const halfN = Math.floor(N / 2);

    for (let k = 0; k < halfN; k++) {
        let real = 0, imag = 0;
        for (let n = 0; n < N; n++) {
            const angle = (2 * Math.PI * k * n) / N;
            real += windowed[n] * Math.cos(angle);
            imag -= windowed[n] * Math.sin(angle);
        }

        // Magnitude (single-sided, multiply by 2 except DC/Nyquist)
        let magnitude = Math.sqrt(real * real + imag * imag) / N;
        if (k > 0 && k < halfN - 1) {
            magnitude *= 2;
        }

        // Correct frequency axis: freq = k * fs / N
        const freq = (k * fs) / N;

        frequencies.push({
            freq: freq.toFixed(2),
            magnitude: magnitude.toFixed(6),
            freqHz: freq
        });
        magnitudes.push(magnitude);
    }

    // 4) Find dominant frequency (IGNORE DC bin at index 0)
    let maxMag = 0;
    let maxIdx = 1;  // Start from index 1, not 0
    for (let k = 1; k < magnitudes.length; k++) {
        if (magnitudes[k] > maxMag) {
            maxMag = magnitudes[k];
            maxIdx = k;
        }
    }
    const dominantFrequency = (maxIdx * fs) / N;

    // 5) Normalize magnitudes for chart display
    const maxMagnitude = Math.max(...magnitudes.slice(1)) || 1;  // Ignore DC
    const normalizedFreqs = frequencies.map((f, i) => ({
        ...f,
        magnitudeNorm: (magnitudes[i] / maxMagnitude).toFixed(4)
    }));

    // Return only meaningful frequencies (limit to 50 bins for performance)
    return {
        frequencies: normalizedFreqs.slice(0, Math.min(50, halfN)),
        magnitudes: magnitudes.slice(0, Math.min(50, halfN)),
        dominantFrequency: dominantFrequency.toFixed(2),
        sampleRate: fs,
        nyquistFrequency: fs / 2
    };
};

/**
 * Perform Haar Wavelet Transform on sensor data
 * @param {Array} data - Array of sensor data points with 'value' property
 * @returns {Array} - Wavelet coefficients with approximation and detail
 */
export const waveletTransform = (data) => {
    if (!data || data.length < 2) {
        return [];
    }

    // Simplified Haar wavelet transform
    const values = data.map(d => parseFloat(d.value) || 0);
    const coefficients = [];

    for (let i = 0; i < values.length - 1; i += 2) {
        const avg = (values[i] + values[i + 1]) / 2;
        const diff = (values[i] - values[i + 1]) / 2;
        coefficients.push({ index: i / 2, approximation: avg.toFixed(4), detail: diff.toFixed(4) });
    }

    return coefficients.slice(0, 50);
};
