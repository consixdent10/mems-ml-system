import sys

f = r'c:\Users\WIN 10\Desktop\Downloads\VS Code Praval\mems-ml-system\backend\main.py'
with open(f, 'r', encoding='utf-8') as fh:
    c = fh.read()

old_block = """        report = build_health_report(
            sensor_data=request.sensor_data,
            degradation_level=request.degradation_level
        )
        
        # Override RUL with ML model prediction if available
        if ml_rul is not None:
            report['rul_percent'] = round(ml_rul, 2)
            report['rul_source'] = f'ML Model ({ml_trainer.best_model_name})'
            # Recompute status based on ML-predicted RUL
            from utils.status_utils import get_status_from_features
            snr = report.get('sensor_stats', {}).get('snr', 30)
            drift = report.get('sensor_stats', {}).get('mean_drift', 0.002)
            noise = report.get('sensor_stats', {}).get('mean_noise', 0.01)
            temp = report.get('sensor_stats', {}).get('mean_temp', 25.0)
            status_result = get_status_from_features(snr, drift, noise, ml_rul, temp)
            report['status'] = status_result['status']
            report['triggered_rule'] = status_result['triggered_rule']
            report['rule_reason'] = status_result['rule_reason']
        else:
            report['rul_source'] = 'Rule-based estimation'"""

new_block = """        # Pass ml_rul into health report so it starts the forecast correctly
        report = build_health_report(
            sensor_data=request.sensor_data,
            degradation_level=request.degradation_level,
            ml_rul=ml_rul
        )
        
        if ml_rul is not None:
            report['rul_source'] = f'ML Model ({ml_trainer.best_model_name})'
        else:
            report['rul_source'] = 'Rule-based estimation'"""

if old_block in c:
    c = c.replace(old_block, new_block)
    with open(f, 'w', encoding='utf-8') as fh:
        fh.write(c)
    print("main.py modified successfully.")
else:
    print("ERROR: Target text in main.py not found.")
