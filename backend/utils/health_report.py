"""
Health Report Utility

Single source of truth for unified sensor health report.
Combines RUL, status, risks, forecast, and maintenance schedule.
All APIs should use this function for consistent data.
"""

import numpy as np
import pandas as pd

from utils.rul_utils import (
    compute_rul_from_degradation_level,
    compute_rul_from_sensor_data,
    forecast_rul_curve,
    compute_risk_scores,
    get_maintenance_schedule,
    clamp
)
from utils.status_utils import get_status_from_features


def build_health_report(sensor_data=None, degradation_level=None, ml_rul=None):
    """
    Build a complete unified health report.
    
    Args:
        sensor_data: DataFrame or list of dicts with sensor readings
                    (must have 'value', 'temperature', 'drift', 'noise')
        degradation_level: Degradation slider value (0-100)
    
    Returns:
        Complete health report dictionary with:
        - rul_percent
        - status, triggered_rule, rule_reason, status_reason_details
        - failure_risks (drift_risk, noise_risk, temp_risk)
        - maintenance_schedule
        - forecast (days, expected, upper, lower)
    """
    # Convert to DataFrame if needed
    if sensor_data is not None and not isinstance(sensor_data, pd.DataFrame):
        sensor_data = pd.DataFrame(sensor_data)
    
    # Extract features from sensor data
    if sensor_data is not None and len(sensor_data) > 0:
        mean_value = float(np.mean(sensor_data['value'].values))
        # Handle real datasets that may not have temperature/drift/noise columns
        mean_temp = 25.0  # Default room temp; vibration datasets don't have real temperature sensors
        mean_drift = float(np.mean(sensor_data['drift'].values)) if 'drift' in sensor_data.columns else 0.002
        mean_noise = float(np.mean(sensor_data['noise'].values)) if 'noise' in sensor_data.columns else abs(float(np.std(sensor_data['value'].values)))
        
        # Calculate SNR
        signal_power = np.var(sensor_data['value'].values)
        if 'noise' in sensor_data.columns:
            noise_power = np.var(sensor_data['noise'].values) + 0.0001
        else:
            # For real datasets: estimate noise as high-frequency component
            values = sensor_data['value'].values
            smoothed = np.convolve(values, np.ones(10)/10, mode='same')
            noise_power = np.var(values - smoothed) + 0.0001
        snr = float(10 * np.log10(signal_power / noise_power))
        
        # Wait for ML model training for RUL estimation
        rul_percent = 100.0
    elif degradation_level is not None:
        # Compute from degradation level (synthetic data mode)
        rul_percent = compute_rul_from_degradation_level(degradation_level)
        
        # Estimate features from degradation level
        mean_drift = 0.002 + (degradation_level / 100) * 0.058  # 0.002 to 0.06
        mean_noise = 0.01 + (degradation_level / 100) * 0.14   # 0.01 to 0.15
        mean_temp = 25.0 + (degradation_level / 100) * 10      # 25 to 35
        mean_value = 9.81  # Default accelerometer
        
        # Estimate SNR
        snr = max(5, 30 - (degradation_level / 100) * 25)  # 30 to 5
    else:
        # Default fallback
        rul_percent = 100.0
        mean_drift = 0.002
        mean_noise = 0.01
        mean_temp = 25.0
        mean_value = 9.81
        snr = 30.0
    
    # Override with ML RUL if provided
    if ml_rul is not None:
        rul_percent = ml_rul
        
    # Ensure RUL is clamped
    rul_percent = clamp(rul_percent, 0, 100)
    
    # Get unified status
    status_result = get_status_from_features(snr, mean_drift, mean_noise, rul_percent, mean_temp)
    
    # Get failure risk scores
    failure_risks = compute_risk_scores(mean_drift, mean_noise, mean_temp)
    
    # Get maintenance schedule
    maintenance = get_maintenance_schedule(rul_percent)
    
    # Generate forecast
    deg_level = degradation_level if degradation_level is not None else int(100 - rul_percent)
    forecast = forecast_rul_curve(rul_percent, mean_drift, mean_noise, deg_level, horizon=100)
    
    # Build complete report
    return {
        'rul_percent': round(rul_percent, 2),
        'status': status_result['status'],
        'triggered_rule': status_result['triggered_rule'],
        'rule_reason': status_result['rule_reason'],
        'status_reason_details': status_result['status_reason_details'],
        'failure_risks': {
            'drift_risk': failure_risks['calibration_drift'],
            'noise_risk': failure_risks['noise_increase'],
            'temp_risk': failure_risks['temperature_sensitivity']
        },
        'maintenance_schedule': {
            'summary': maintenance['notes'][0] if maintenance['notes'] else 'Continue monitoring',
            'next_check_days': maintenance['next_check_days'],
            'calibration_interval_days': maintenance['recommended_interval_days'],
            'notes': maintenance['notes']
        },
        'forecast': forecast,
        'sensor_stats': {
            'mean_value': round(mean_value, 4),
            'mean_drift': round(mean_drift, 4),
            'mean_noise': round(mean_noise, 4),
            'mean_temp': round(mean_temp, 2),
            'snr': round(snr, 2)
        }
    }
