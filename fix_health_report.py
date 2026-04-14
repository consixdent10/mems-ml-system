import sys

f = r'c:\Users\WIN 10\Desktop\Downloads\VS Code Praval\mems-ml-system\backend\utils\health_report.py'
with open(f, 'r', encoding='utf-8') as fh:
    c = fh.read()

old1 = "def build_health_report(sensor_data=None, degradation_level=None):"
new1 = "def build_health_report(sensor_data=None, degradation_level=None, ml_rul=None):"

old2 = """    # Ensure RUL is clamped
    rul_percent = clamp(rul_percent, 0, 100)"""

new2 = """    # Override with ML RUL if provided
    if ml_rul is not None:
        rul_percent = ml_rul
        
    # Ensure RUL is clamped
    rul_percent = clamp(rul_percent, 0, 100)"""

if old1 in c and old2 in c:
    c = c.replace(old1, new1)
    c = c.replace(old2, new2)
    with open(f, 'w', encoding='utf-8') as fh:
        fh.write(c)
    print("health_report.py modified successfully.")
else:
    print("ERROR: Target text in health_report.py not found.")
