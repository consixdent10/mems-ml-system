import pandas as pd
import numpy as np
import os

def generate_critical_csv():
    print("Generating extreme degradation CSV for Critical Banner...")
    
    # 2000 samples (~0.1 seconds at 20kHz)
    n_samples = 2000
    time = np.linspace(0, 0.1, n_samples)
    
    # 1. Base severely degraded vibration
    # High fundamental amplitude
    base_vibration = 6.0 * np.sin(2 * np.pi * 50 * time)
    
    # 2. Extreme impulses (representing severe outer race fault impacts)
    # Adding brutal spikes every 0.01s
    impulses = np.zeros(n_samples)
    for i in range(0, n_samples, 200):
        impulses[i:min(i+5, n_samples)] = np.random.uniform(15.0, 25.0, min(5, n_samples-i))
    
    # 3. High broadband noise (severe bearing wear)
    noise = np.random.normal(0, 3.5, n_samples)
    
    # 4. Calibration Drift (wandering baseline)
    drift = np.linspace(0, 2.5, n_samples)
    
    # Combine signals for the overall 'value' to look visually brutal
    values = base_vibration + impulses + noise + drift
    
    # But for the ML Pipeline, we must explicitely supply the severe columns
    # because the custom upload logic uses 25.0 as default temperature if missing
    df = pd.DataFrame({
        'time': time,
        'value': values,
        'temperature': np.random.uniform(55.0, 65.0, n_samples),
        'drift': np.linspace(0.5, 0.8, n_samples), # Very high drift
        'noise': np.random.uniform(0.12, 0.20, n_samples) # Huge noise
    })
    
    # Save to desktop
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "Severe_Fault_Critical.csv")
    df.to_csv(desktop_path, index=False)
    print("✅ Successfully created: " + desktop_path.encode('utf-8', 'ignore').decode('utf-8'))
    print("You can now upload this custom CSV to the dashboard!")

if __name__ == "__main__":
    generate_critical_csv()
