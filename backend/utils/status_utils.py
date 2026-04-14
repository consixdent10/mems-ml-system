"""
Status Utility Functions

Single source of truth for sensor status determination.
All API endpoints must use these functions for consistent status values.
"""


def get_status_from_features(snr: float, drift: float, noise: float, 
                              rul_percent: float, temperature: float = 25.0) -> dict:
    """
    Determine sensor status from features using consistent rules.
    
    Rules (matching XAI thresholds):
    - HEALTHY: SNR > 20 AND Drift < 0.02 AND Noise < 0.08
    - WARNING: (10 < SNR <= 20) OR (0.02 <= Drift < 0.05) OR (0.08 <= Noise < 0.12)
    - CRITICAL: SNR <= 10 OR Drift >= 0.05 OR Noise >= 0.12 OR RUL < 30
    
    Args:
        snr: Signal-to-Noise Ratio
        drift: Mean drift value
        noise: Mean noise value
        rul_percent: Remaining Useful Life percentage (0-100)
        temperature: Temperature (optional, for details)
    
    Returns:
        Dictionary with status, triggered_rule, rule_reason, status_reason_details
    """
    # Initialize details
    details = {
        'snr': round(snr, 2),
        'drift': round(drift, 4),
        'noise': round(noise, 4),
        'temperature': round(temperature, 2),
        'rul_percent': round(rul_percent, 2)
    }
    
    # Check CRITICAL conditions
    # RUL < 30% is the primary CRITICAL trigger
    # Secondary indicators (SNR/drift/noise) only trigger CRITICAL
    # if RUL is already in warning range (<70%)
    critical_reasons = []
    if rul_percent < 30:
        critical_reasons.append(f"RUL={rul_percent:.1f}% (<30%)")
    if rul_percent < 70:  # Only check secondary if RUL is concerning
        if drift >= 0.05:
            critical_reasons.append(f"Drift={drift:.4f} (>=0.05)")
        if noise >= 0.12:
            critical_reasons.append(f"Noise={noise:.4f} (>=0.12)")
    
    if critical_reasons:
        return {
            'status': 'CRITICAL',
            'triggered_rule': 'Rule 3 (Critical Threshold)',
            'rule_reason': ' | '.join(critical_reasons),
            'status_reason_details': details
        }
    
    # Check WARNING conditions
    # RUL is primary; SNR/drift/noise are secondary
    # Only warn if RUL is in warning range OR multiple secondary indicators trigger
    warning_reasons = []
    secondary_warnings = 0
    if 10 < snr <= 20:
        secondary_warnings += 1
    if 0.02 <= drift < 0.05:
        warning_reasons.append(f"Drift={drift:.4f} (0.02-0.05)")
        secondary_warnings += 1
    if 0.08 <= noise < 0.12:
        warning_reasons.append(f"Noise={noise:.4f} (0.08-0.12)")
        secondary_warnings += 1
    if 30 <= rul_percent < 70:
        warning_reasons.append(f"RUL={rul_percent:.1f}% (30-70%)")
    
    # Trigger WARNING only if RUL is in warning range OR 2+ secondary indicators
    if warning_reasons or secondary_warnings >= 2:
        if not warning_reasons and secondary_warnings >= 2:
            warning_reasons.append("Multiple degradation indicators detected")
        return {
            'status': 'WARNING',
            'triggered_rule': 'Rule 2 (Warning Threshold)',
            'rule_reason': ' | '.join(warning_reasons),
            'status_reason_details': details
        }
    
    # All conditions healthy
    return {
        'status': 'HEALTHY',
        'triggered_rule': 'Rule 1 (All Normal)',
        'rule_reason': f"SNR={snr:.1f} (>20), Drift={drift:.4f} (<0.02), Noise={noise:.4f} (<0.08), RUL={rul_percent:.1f}% (≥70%)",
        'status_reason_details': details
    }


def get_status_color(status: str) -> str:
    """Get CSS/Tailwind color class for status."""
    colors = {
        'HEALTHY': 'green',
        'WARNING': 'orange',
        'CRITICAL': 'red'
    }
    return colors.get(status, 'gray')


def get_status_emoji(status: str) -> str:
    """Get emoji for status."""
    emojis = {
        'HEALTHY': '✅',
        'WARNING': '⚠️',
        'CRITICAL': '🔴'
    }
    return emojis.get(status, '❓')
