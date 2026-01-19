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
