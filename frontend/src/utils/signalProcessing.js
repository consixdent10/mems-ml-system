/**
 * Signal Processing Utilities
 * FFT and Wavelet Transform implementations with proper DC removal and windowing
 * Viva-ready with comprehensive metrics
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
 * @returns {Object} - Complete FFT analysis with metrics
 */
export const performFFT = (data, sampleRate = null) => {
    if (!data || data.length < 4) {
        return {
            frequencies: [],
            magnitudes: [],
            dominantFrequency: '0.00',
            peakMagnitude: 0,
            noiseFloor: 0,
            sampleRate: 100,
            nyquistFrequency: 50,
            bandwidth: 0
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

    // 1) Remove DC offset (mean subtraction) - PREVENTS FALSE 0 Hz SPIKE
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

        // Magnitude with proper scaling: (2/N) for single-sided spectrum
        let magnitude = Math.sqrt(real * real + imag * imag) / N;
        if (k > 0 && k < halfN - 1) {
            magnitude *= 2;  // Single-sided spectrum scaling
        }

        // Correct frequency axis: freq = k * fs / N
        const freq = (k * fs) / N;

        frequencies.push({
            freq: freq.toFixed(2),
            magnitude: magnitude.toFixed(6),
            freqHz: freq,
            magnitudeRaw: magnitude
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
    const peakMagnitude = maxMag;

    // 5) Calculate noise floor (median of upper 30% frequency magnitudes)
    const upperBins = magnitudes.slice(Math.floor(magnitudes.length * 0.7));
    const sortedUpper = [...upperBins].sort((a, b) => a - b);
    const noiseFloor = sortedUpper[Math.floor(sortedUpper.length / 2)] || 0;

    // 6) Calculate 3dB bandwidth (frequencies where magnitude > peakMag/sqrt(2))
    const threshold = peakMagnitude / Math.sqrt(2);
    let bandwidthLow = dominantFrequency;
    let bandwidthHigh = dominantFrequency;
    for (let k = maxIdx; k >= 1; k--) {
        if (magnitudes[k] >= threshold) {
            bandwidthLow = (k * fs) / N;
        } else break;
    }
    for (let k = maxIdx; k < magnitudes.length; k++) {
        if (magnitudes[k] >= threshold) {
            bandwidthHigh = (k * fs) / N;
        } else break;
    }
    const bandwidth = bandwidthHigh - bandwidthLow;

    // 7) Normalize magnitudes for chart display
    const maxMagnitude = Math.max(...magnitudes.slice(1)) || 1;
    const normalizedFreqs = frequencies.map((f, i) => ({
        ...f,
        magnitudeNorm: (magnitudes[i] / maxMagnitude).toFixed(4),
        magnitudeLog: Math.log10(magnitudes[i] + 1e-8).toFixed(4)
    }));

    // Return comprehensive FFT analysis
    return {
        frequencies: normalizedFreqs.slice(0, Math.min(50, halfN)),
        magnitudes: magnitudes.slice(0, Math.min(50, halfN)),
        dominantFrequency: dominantFrequency.toFixed(2),
        peakMagnitude: peakMagnitude.toFixed(4),
        noiseFloor: noiseFloor.toFixed(6),
        bandwidth: bandwidth.toFixed(2),
        sampleRate: fs,
        nyquistFrequency: fs / 2,
        dcRemoved: true,
        windowApplied: 'Hann'
    };
};

/**
 * Perform Haar Wavelet Transform on sensor data with energy metrics
 * @param {Array} data - Array of sensor data points with 'value' property
 * @returns {Object} - Wavelet coefficients and energy analysis
 */
export const waveletTransform = (data) => {
    if (!data || data.length < 2) {
        return {
            coefficients: [],
            approxEnergy: 0,
            detailEnergy: 0,
            energyRatio: 0,
            interpretation: 'Insufficient data'
        };
    }

    // Simplified Haar wavelet transform
    const values = data.map(d => parseFloat(d.value) || 0);
    const coefficients = [];

    // Normalize input for stable visualization
    const valMin = Math.min(...values);
    const valMax = Math.max(...values);
    const valRange = valMax - valMin || 1;
    const normalized = values.map(v => (v - valMin) / valRange * 10);

    let approxEnergy = 0;
    let detailEnergy = 0;

    for (let i = 0; i < normalized.length - 1; i += 2) {
        const a = normalized[i];
        const b = normalized[i + 1];
        const avg = (a + b) / 2;
        const diff = (a - b) / 2;

        coefficients.push({
            index: i / 2,
            approximation: avg.toFixed(4),
            detail: diff.toFixed(4),
            approxRaw: avg,
            detailRaw: diff
        });

        approxEnergy += avg * avg;
        detailEnergy += diff * diff;
    }

    const totalEnergy = approxEnergy + detailEnergy || 1;
    const energyRatio = detailEnergy / totalEnergy;

    // Interpretation based on energy ratio
    let interpretation = '';
    if (energyRatio < 0.1) {
        interpretation = 'Low-frequency components dominate (smooth signal)';
    } else if (energyRatio < 0.3) {
        interpretation = 'Balanced frequency content (normal operation)';
    } else if (energyRatio < 0.5) {
        interpretation = 'Elevated high-frequency content (increased noise)';
    } else {
        interpretation = 'High-frequency noise dominates (possible degradation)';
    }

    return {
        coefficients: coefficients.slice(0, 50),
        approxEnergy: approxEnergy.toFixed(2),
        detailEnergy: detailEnergy.toFixed(2),
        totalEnergy: totalEnergy.toFixed(2),
        energyRatio: (energyRatio * 100).toFixed(1),
        interpretation
    };
};
