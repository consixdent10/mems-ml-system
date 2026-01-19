"""
RUL (Remaining Useful Life) Utility Functions

Single source of truth for RUL computation across the entire backend.
All API endpoints should use these functions for consistent RUL values.
"""

import numpy as np


def clamp(value: float, low: float, high: float) -> float:
    """Clamp value between low and high bounds."""
    return max(low, min(high, value))


def compute_degradation_score(drift: float, noise: float, temperature: float = 25.0, 
                               sensor_type: str = 'accelerometer') -> float:
    """
    Compute a degradation score (0-100) from sensor characteristics.
    
    Higher drift/noise = higher degradation score
    
    Args:
        drift: Mean drift value (typical range: 0.002 - 0.06)
        noise: Mean noise value (typical range: 0.01 - 0.15)
        temperature: Mean temperature (optional, for future use)
        sensor_type: Type of sensor
    
    Returns:
        Degradation score from 0 (healthy) to 100 (critical)
    """
    # Normalize drift to 0-100 scale (0.002 = healthy, 0.06 = critical)
    drift_score = ((drift - 0.002) / (0.06 - 0.002)) * 100
    drift_score = clamp(drift_score, 0, 100)
    
    # Normalize noise to 0-100 scale (0.01 = healthy, 0.15 = critical)
    noise_score = ((noise - 0.01) / (0.15 - 0.01)) * 100
    noise_score = clamp(noise_score, 0, 100)
    
    # Temperature contribution (mild effect)
    temp_deviation = abs(temperature - 25.0)
    temp_score = min(temp_deviation * 2, 20)  # Max 20% contribution
    
    # Weighted combination
    degradation_score = 0.45 * drift_score + 0.45 * noise_score + 0.10 * temp_score
    
    # Add small random noise to prevent perfect ML prediction
    # This makes the RUL target slightly unpredictable for realism
    random_noise = np.random.normal(0, 3)  # ±3% noise
    degradation_score += random_noise
    
    return clamp(degradation_score, 0, 100)


def compute_rul_from_degradation_level(degradation_level: int) -> float:
    """
    Compute RUL% from the degradation slider value (0-100).
    
    This is the PRIMARY function for synthetic data generation.
    Ensures monotonic trend: higher degradation => lower RUL.
    
    Args:
        degradation_level: Degradation percentage from slider (0-100)
    
    Returns:
        RUL percentage (0-100)
    
    Expected behavior:
        1% degradation => RUL ~99-100%
        50% degradation => RUL ~45-60%
        90% degradation => RUL ~5-20%
    """
    # Base RUL is inverse of degradation
    base_rul = 100 - degradation_level
    
    # Add small random variation (±3%) for realism
    jitter = np.random.uniform(-3, 3)
    
    # Apply jitter but maintain monotonic trend
    rul = base_rul + jitter
    
    # Clamp to valid range
    return clamp(rul, 0, 100)


def compute_rul_from_sensor_data(drift: float, noise: float, 
                                  temperature: float = 25.0,
                                  sensor_type: str = 'accelerometer') -> float:
    """
    Compute RUL% from actual sensor measurements.
    
    This is used when degradation level is not available
    (e.g., uploaded data).
    
    Args:
        drift: Mean drift value
        noise: Mean noise value
        temperature: Mean temperature
        sensor_type: Type of sensor
    
    Returns:
        RUL percentage (0-100)
    """
    degradation_score = compute_degradation_score(drift, noise, temperature, sensor_type)
    
    # RUL = 100 - degradation_score with jitter
    base_rul = 100 - degradation_score
    jitter = np.random.uniform(-2, 2)
    rul = base_rul + jitter
    
    return clamp(rul, 0, 100)


def get_health_status(rul_percent: float) -> str:
    """
    Get health status string from RUL percentage.
    
    Returns:
        'HEALTHY', 'WARNING', or 'CRITICAL'
    """
    if rul_percent >= 70:
        return 'HEALTHY'
    elif rul_percent >= 30:
        return 'WARNING'
    else:
        return 'CRITICAL'


def forecast_rul_curve(rul_start: float, drift: float, noise: float, 
                       degradation_level: int, horizon: int = 100) -> dict:
    """
    Generate RUL forecast curve over given horizon (days).
    
    The curve starts at rul_start and decreases over time based on
    degradation level. Higher degradation = faster decline.
    
    Args:
        rul_start: Current RUL percentage (0-100)
        drift: Current drift value
        noise: Current noise value
        degradation_level: Current degradation level (0-100)
        horizon: Forecast horizon in days (default 100)
    
    Returns:
        Dictionary with days, expected, upper, lower arrays
    """
    days = list(range(horizon))
    
    # Degradation rate: higher degradation = faster decline
    # At 0% deg: decline ~0.2% per day
    # At 50% deg: decline ~0.5% per day
    # At 100% deg: decline ~1.0% per day
    daily_decline = 0.2 + (degradation_level / 100) * 0.8
    
    # Add contribution from drift/noise
    drift_factor = (drift / 0.06) * 0.3 if drift > 0 else 0
    noise_factor = (noise / 0.15) * 0.2 if noise > 0 else 0
    daily_decline += drift_factor + noise_factor
    
    expected = []
    upper = []
    lower = []
    
    for day in days:
        # Exponential decay for realism
        decay_factor = np.exp(-daily_decline * day / 100)
        predicted_rul = rul_start * decay_factor
        
        # Add slight randomness
        jitter = np.random.uniform(-0.5, 0.5)
        predicted_rul = clamp(predicted_rul + jitter, 0, 100)
        
        # Confidence bounds (±5-10%)
        uncertainty = 3 + (day / horizon) * 5  # Grows with time
        upper_bound = clamp(predicted_rul + uncertainty, 0, 100)
        lower_bound = clamp(predicted_rul - uncertainty, 0, 100)
        
        expected.append(round(predicted_rul, 2))
        upper.append(round(upper_bound, 2))
        lower.append(round(lower_bound, 2))
    
    return {
        'days': days,
        'expected': expected,
        'upper': upper,
        'lower': lower
    }


def compute_risk_scores(drift: float, noise: float, temperature: float = 25.0) -> dict:
    """
    Compute failure mode risk scores (0-100 for each).
    
    Args:
        drift: Mean drift value
        noise: Mean noise value
        temperature: Mean temperature
    
    Returns:
        Dictionary with calibration_drift, noise_increase, temperature_sensitivity
        Each value is clamped to 0-100.
    """
    # Critical thresholds (values at which risk = 100%)
    DRIFT_CRITICAL = 0.06
    NOISE_CRITICAL = 0.15
    TEMP_SPAN = 30  # ±15°C from optimal
    
    # Calibration drift risk
    drift_risk = (abs(drift) / DRIFT_CRITICAL) * 100
    drift_risk = clamp(drift_risk, 0, 100)
    
    # Noise increase risk
    noise_risk = (noise / NOISE_CRITICAL) * 100
    noise_risk = clamp(noise_risk, 0, 100)
    
    # Temperature sensitivity risk
    temp_deviation = abs(temperature - 25.0)
    temp_risk = (temp_deviation / TEMP_SPAN) * 100
    temp_risk = clamp(temp_risk, 0, 100)
    
    return {
        'calibration_drift': round(drift_risk, 1),
        'noise_increase': round(noise_risk, 1),
        'temperature_sensitivity': round(temp_risk, 1)
    }


def get_maintenance_schedule(rul_percent: float) -> dict:
    """
    Get maintenance recommendations based on RUL.
    
    Returns:
        Dictionary with next_check_days, recommended_interval_days, notes
    """
    if rul_percent >= 70:
        return {
            'next_check_days': 30,
            'recommended_interval_days': 90,
            'notes': ['Routine maintenance schedule', 'Continue normal monitoring']
        }
    elif rul_percent >= 50:
        return {
            'next_check_days': 14,
            'recommended_interval_days': 60,
            'notes': ['Increase monitoring frequency', 'Schedule calibration check']
        }
    elif rul_percent >= 30:
        return {
            'next_check_days': 7,
            'recommended_interval_days': 30,
            'notes': ['Urgent: Plan sensor replacement', 'Daily monitoring recommended']
        }
    else:
        return {
            'next_check_days': 1,
            'recommended_interval_days': 7,
            'notes': ['CRITICAL: Immediate replacement required', 'Backup systems recommended']
        }
