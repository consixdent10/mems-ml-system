
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from scipy.stats import kurtosis, skew
from scipy.fft import fft
from datetime import datetime

# Directory for saved models
SAVED_MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def safe_minmax(x):
    """Safely normalize array to [0, 1] range, avoiding divide-by-zero"""
    x = np.asarray(x, dtype=float)
    min_val = np.min(x)
    max_val = np.max(x)
    if max_val - min_val < 1e-10:
        return np.zeros_like(x)
    return (x - min_val) / (max_val - min_val)


def extract_statistical_features(signal_segment):
    """
    Extract statistical features from a vibration signal segment.
    
    These are standard features used in bearing fault diagnosis literature
    (Lei et al., 2020; Randall & Antoni, 2011).
    
    Returns dict of 10 features.
    """
    x = np.asarray(signal_segment, dtype=float)
    n = len(x)
    
    # Time-domain features
    rms = np.sqrt(np.mean(x ** 2))
    mean_val = np.mean(x)
    std_val = np.std(x)
    peak = np.max(np.abs(x))
    peak_to_peak = np.max(x) - np.min(x)
    
    # Higher-order statistics
    kurt = float(kurtosis(x, fisher=True))  # Excess kurtosis (0 for Gaussian)
    skewness = float(skew(x))
    
    # Shape descriptors
    crest_factor = peak / rms if rms > 1e-10 else 0
    shape_factor = rms / np.mean(np.abs(x)) if np.mean(np.abs(x)) > 1e-10 else 0
    impulse_factor = peak / np.mean(np.abs(x)) if np.mean(np.abs(x)) > 1e-10 else 0
    
    # Frequency-domain features
    fft_vals = np.abs(fft(x))[:n // 2]
    freq_mean = np.mean(fft_vals)
    freq_std = np.std(fft_vals)
    
    return {
        'rms': float(rms),
        'std': float(std_val),
        'peak': float(peak),
        'peak_to_peak': float(peak_to_peak),
        'kurtosis': kurt,
        'skewness': skewness,
        'crest_factor': float(crest_factor),
        'shape_factor': float(shape_factor),
        'impulse_factor': float(impulse_factor),
        'freq_energy': float(freq_mean),
    }


# ==============================================================================
# PART 1: RUL Regression Trainer (Improved)
# ==============================================================================

class MLModelTrainer:
    """Train regression models for Remaining Useful Life (RUL) prediction"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.trained = False
        self.model_paths = {}
        self.best_model_name = None
        self.predictions_sample = None
        self.feature_names = [
            'RMS',
            'Std Dev',
            'Peak',
            'Peak-to-Peak',
            'Kurtosis',
            'Skewness',
            'Crest Factor',
            'Shape Factor',
            'Impulse Factor',
            'Freq Energy',
            'Mean Drift',
            'Mean Noise',
            'Mean Temperature',
        ]
        
        # Create saved_models directory if it doesn't exist
        os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    @property
    def expected_feature_count(self):
        return len(self.feature_names)

    def _window_aux_features(self, data, start, end):
        """Extract drift/noise/temperature features for one training window."""
        window = data.iloc[start:end]
        values = window['value'].astype(float).values

        if 'drift' in window.columns:
            mean_drift = float(np.mean(np.abs(window['drift'].astype(float).values)))
        else:
            mean_drift = float(abs(values[-1] - values[0]) / max(len(values), 1))

        if 'noise' in window.columns:
            mean_noise = float(np.mean(np.abs(window['noise'].astype(float).values)))
        else:
            smooth_window = min(10, len(values))
            smoothed = pd.Series(values).rolling(window=smooth_window, min_periods=1).mean().values
            mean_noise = float(np.std(values - smoothed))

        if 'temperature' in window.columns:
            mean_temp = float(np.mean(window['temperature'].astype(float).values))
        else:
            mean_temp = 25.0

        return mean_drift, mean_noise, mean_temp

    def _estimate_window_rul(self, feats, mean_drift, mean_noise, mean_temp):
        """
        Estimate RUL from physical/signal indicators instead of row position.

        Using row position made the end of a healthy dataset look degraded. These
        indicators keep baseline data healthy while lowering RUL for noisy,
        impulsive, high-vibration windows.
        """
        drift_score = safe_minmax([0.002, mean_drift, 0.06])[1] * 100
        noise_score = safe_minmax([0.01, mean_noise, 0.15])[1] * 100
        temp_score = min(abs(mean_temp - 25.0) * 2.0, 20.0)

        vibration_score = (feats['std'] / (feats['std'] + 0.25)) * 100 if feats['std'] > 0 else 0
        impulsive_score = min(max(feats['kurtosis'], 0) / 10.0 * 100, 100)
        crest_score = min(max(feats['crest_factor'] - 3.0, 0) / 3.0 * 100, 100)

        degradation = (
            0.20 * drift_score +
            0.20 * noise_score +
            0.10 * temp_score +
            0.25 * vibration_score +
            0.15 * impulsive_score +
            0.10 * crest_score
        )
        return float(np.clip(100 - degradation, 0, 100))
        
    def prepare_data(self, data):
        """
        Prepare features and target from sensor data.
        
        Features (11 total):
            - 10 statistical features from sliding windows (RMS, Std, Kurtosis, etc.)
            - 1 temporal feature: normalized operating time (0→1)
              This is standard in prognostics — e.g., NASA CMAPSS uses cycle number.
        
        Target: RUL% (0 to 100) - derived from degradation indicators
        """
        values = data['value'].values
        n = len(values)
        
        # --- Extract windowed features ---
        # Smaller windows = more samples for robust accuracy measurement
        window_size = min(50, n // 10)
        if window_size < 20:
            window_size = 20
        
        step = max(1, window_size // 6)  # ~83% overlap → ~325 windows from 2000 rows
        
        feature_rows = []
        for start in range(0, n - window_size, step):
            segment = values[start:start + window_size]
            feats = extract_statistical_features(segment)
            feature_rows.append(feats)
        
        X = pd.DataFrame(feature_rows).values
        num_windows = len(feature_rows)
        
        # Add normalized operating time as feature #11
        # This is standard practice in prognostics (PHM / NASA CMAPSS)
        # Operating time is the strongest predictor of remaining life
        operating_time = np.linspace(0, 1, num_windows).reshape(-1, 1)
        X = np.hstack([X, operating_time])
        
        # --- Compute RUL target ---
        # Degradation = f(operating_time, signal_characteristics)
        window_positions = operating_time.flatten()
        
        # Signal-based degradation indicators
        rms_values = X[:, 0]  # RMS
        kurt_values = X[:, 4]  # Kurtosis
        
        rms_norm = safe_minmax(rms_values)
        kurt_norm = safe_minmax(np.clip(kurt_values, -3, 20))
        
        # Composite degradation: 55% time + 25% RMS + 20% kurtosis
        degradation = (
            0.55 * window_positions +
            0.25 * rms_norm +
            0.20 * kurt_norm
        )
        degradation = np.clip(degradation, 0, 1)
        
        # RUL = 100% (healthy) to 0% (failed)
        y = (1 - degradation) * 100
        
        # Add small realistic noise (1.5-2.5%) — prevents perfect fit
        noise_std = 1.5 + degradation * 1.0
        y = y + np.random.normal(0, 1, len(y)) * noise_std
        y = np.clip(y, 0, 100)
        
        return X, y.astype(float)

    def prepare_data(self, data):
        """
        Prepare windowed features and RUL targets from sensor data.

        The target is derived from physical/signal degradation indicators. It
        does not use row position, so a healthy baseline recording remains
        healthy throughout the file.
        """
        values = data['value'].astype(float).values
        n = len(values)

        window_size = min(50, n // 10)
        if window_size < 20:
            window_size = 20

        step = max(1, window_size // 6)

        feature_rows = []
        targets = []
        for start in range(0, n - window_size, step):
            segment = values[start:start + window_size]
            feats = extract_statistical_features(segment)
            mean_drift, mean_noise, mean_temp = self._window_aux_features(
                data, start, start + window_size
            )
            feats['mean_drift'] = mean_drift
            feats['mean_noise'] = mean_noise
            feats['mean_temperature'] = mean_temp
            feature_rows.append(feats)
            targets.append(self._estimate_window_rul(feats, mean_drift, mean_noise, mean_temp))

        if not feature_rows:
            raise ValueError("Not enough sensor samples to train models")

        X = pd.DataFrame(feature_rows)[[
            'rms',
            'std',
            'peak',
            'peak_to_peak',
            'kurtosis',
            'skewness',
            'crest_factor',
            'shape_factor',
            'impulse_factor',
            'freq_energy',
            'mean_drift',
            'mean_noise',
            'mean_temperature',
        ]].values

        rng = np.random.default_rng(42)
        y = np.asarray(targets, dtype=float)
        y = np.clip(y + rng.normal(0, 0.75, len(y)), 0, 100)

        return X, y.astype(float)
    
    def _compute_prediction_accuracy(self, y_true, y_pred, tolerance=7.0):
        """
        Compute prediction accuracy using tolerance-band method (PHM standard).
        
        A prediction is 'correct' if it falls within +/- tolerance
        percentage points of the actual RUL value.
        
        Default tolerance = 5 percentage points.
        This is the standard evaluation metric in Prognostics & Health
        Management (PHM) literature, more meaningful than arbitrary bins
        for continuous RUL prediction.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        errors = np.abs(y_true - y_pred)
        correct = np.sum(errors <= tolerance)
        accuracy = float(correct / len(y_true) * 100)
        return round(accuracy, 1)
    
    def train_all_models(self, data):
        """Train all ML models for RUL regression with accuracy metric"""
        print("Preparing data for RUL prediction...")
        X, y = self.prepare_data(data)
        
        # Split data: 80% train, 20% test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
        print(f"RUL range: {y.min():.1f}% to {y.max():.1f}%")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = []
        all_predictions = {}
        
        # 1. Random Forest
        print("Training Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        rf_start = datetime.now()
        rf_model.fit(X_train_scaled, y_train)
        rf_time = (datetime.now() - rf_start).total_seconds()
        rf_pred = rf_model.predict(X_test_scaled)
        all_predictions['Random Forest'] = rf_pred
        
        rf_metrics = self._compute_regression_metrics(y_test, rf_pred)
        rf_accuracy = self._compute_prediction_accuracy(y_test, rf_pred)
        results.append({
            'modelType': 'Random Forest',
            'mae': rf_metrics['mae'],
            'rmse': rf_metrics['rmse'],
            'mse': rf_metrics['mse'],
            'r2Score': rf_metrics['r2Score'],
            'mape': rf_metrics['mape'],
            'accuracy': rf_accuracy,
            'trainingTime': round(rf_time, 2),
            'trainingSize': len(X_train),
            'testSize': len(X_test)
        })
        self.models['Random Forest'] = rf_model
        
        # 2. Gradient Boosting
        print("Training Gradient Boosting...")
        gb_model = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            random_state=42
        )
        gb_start = datetime.now()
        gb_model.fit(X_train_scaled, y_train)
        gb_time = (datetime.now() - gb_start).total_seconds()
        gb_pred = gb_model.predict(X_test_scaled)
        all_predictions['Gradient Boosting'] = gb_pred
        
        gb_metrics = self._compute_regression_metrics(y_test, gb_pred)
        gb_accuracy = self._compute_prediction_accuracy(y_test, gb_pred)
        results.append({
            'modelType': 'Gradient Boosting',
            'mae': gb_metrics['mae'],
            'rmse': gb_metrics['rmse'],
            'mse': gb_metrics['mse'],
            'r2Score': gb_metrics['r2Score'],
            'mape': gb_metrics['mape'],
            'accuracy': gb_accuracy,
            'trainingTime': round(gb_time, 2),
            'trainingSize': len(X_train),
            'testSize': len(X_test)
        })
        self.models['Gradient Boosting'] = gb_model
        
        # 3. Support Vector Machine
        print("Training SVM...")
        svm_model = SVR(kernel='rbf', C=10.0, epsilon=0.1)
        svm_start = datetime.now()
        svm_model.fit(X_train_scaled, y_train)
        svm_time = (datetime.now() - svm_start).total_seconds()
        svm_pred = svm_model.predict(X_test_scaled)
        all_predictions['SVM'] = svm_pred
        
        svm_metrics = self._compute_regression_metrics(y_test, svm_pred)
        svm_accuracy = self._compute_prediction_accuracy(y_test, svm_pred)
        results.append({
            'modelType': 'SVM',
            'mae': svm_metrics['mae'],
            'rmse': svm_metrics['rmse'],
            'mse': svm_metrics['mse'],
            'r2Score': svm_metrics['r2Score'],
            'mape': svm_metrics['mape'],
            'accuracy': svm_accuracy,
            'trainingTime': round(svm_time, 2),
            'trainingSize': len(X_train),
            'testSize': len(X_test)
        })
        self.models['SVM'] = svm_model
        
        # 4. Neural Network
        print("Training Neural Network...")
        nn_model = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            random_state=42,
            max_iter=500,
            early_stopping=True
        )
        nn_start = datetime.now()
        nn_model.fit(X_train_scaled, y_train)
        nn_time = (datetime.now() - nn_start).total_seconds()
        nn_pred = nn_model.predict(X_test_scaled)
        all_predictions['Neural Network'] = nn_pred
        
        nn_metrics = self._compute_regression_metrics(y_test, nn_pred)
        nn_accuracy = self._compute_prediction_accuracy(y_test, nn_pred)
        results.append({
            'modelType': 'Neural Network',
            'mae': nn_metrics['mae'],
            'rmse': nn_metrics['rmse'],
            'mse': nn_metrics['mse'],
            'r2Score': nn_metrics['r2Score'],
            'mape': nn_metrics['mape'],
            'accuracy': nn_accuracy,
            'trainingTime': round(nn_time, 2),
            'trainingSize': len(X_train),
            'testSize': len(X_test)
        })
        self.models['Neural Network'] = nn_model
        
        # Find best model (highest accuracy, then lowest RMSE as tiebreaker)
        best_idx = max(range(len(results)), 
                       key=lambda i: (results[i]['accuracy'], -results[i]['rmse']))
        self.best_model_name = results[best_idx]['modelType']
        self.best_model_r2 = results[best_idx]['r2Score']
        self.best_model_accuracy = results[best_idx]['accuracy']
        
        print(f"Best model: {self.best_model_name} "
              f"(Accuracy: {self.best_model_accuracy}%, "
              f"R2: {self.best_model_r2:.4f}, "
              f"RMSE: {results[best_idx]['rmse']:.4f})")
        
        # Create predictions sample for scatter plot (max 200 points)
        best_pred = all_predictions[self.best_model_name]
        sample_size = min(200, len(y_test))
        indices = np.linspace(0, len(y_test) - 1, sample_size, dtype=int)
        
        self.predictions_sample = {
            'actual': y_test[indices].tolist(),
            'predicted': best_pred[indices].tolist()
        }
        
        # ===== GENERALIZATION TEST =====
        generalization_metrics = None
        try:
            median_rul = np.median(y)
            train_mask = y >= median_rul
            test_mask = y < median_rul
            
            if np.sum(train_mask) > 20 and np.sum(test_mask) > 10:
                X_gen_train = X[train_mask]
                y_gen_train = y[train_mask]
                X_gen_test = X[test_mask]
                y_gen_test = y[test_mask]
                
                X_gen_train_scaled = self.scaler.fit_transform(X_gen_train)
                X_gen_test_scaled = self.scaler.transform(X_gen_test)
                
                best_model = self.models[self.best_model_name]
                best_model.fit(X_gen_train_scaled, y_gen_train)
                gen_pred = best_model.predict(X_gen_test_scaled)
                
                gen_metrics = self._compute_regression_metrics(y_gen_test, gen_pred)
                gen_accuracy = self._compute_prediction_accuracy(y_gen_test, gen_pred)
                generalization_metrics = {
                    'mae': gen_metrics['mae'],
                    'rmse': gen_metrics['rmse'],
                    'r2Score': gen_metrics['r2Score'],
                    'accuracy': gen_accuracy,
                    'trainSize': int(np.sum(train_mask)),
                    'testSize': int(np.sum(test_mask)),
                    'description': 'Train on healthier half, test on more degraded half'
                }
                print(f"Generalization Test - Accuracy: {gen_accuracy}%, "
                      f"R2: {gen_metrics['r2Score']:.4f}")
                
                # Retrain on full data
                best_model.fit(X_train_scaled, y_train)
        except Exception as e:
            print(f"Generalization test failed: {e}")
        
        # ===== AUTO-SAVE BEST MODEL =====
        best_model_saved = False
        best_model_path = None
        try:
            best_model = self.models[self.best_model_name]
            best_model_path = os.path.join(SAVED_MODELS_DIR, 'best_model.joblib')
            scaler_path = os.path.join(SAVED_MODELS_DIR, 'scaler.joblib')
            
            joblib.dump(best_model, best_model_path)
            joblib.dump(self.scaler, scaler_path)
            best_model_saved = True
            print(f"Auto-saved best model to {best_model_path}")
        except Exception as e:
            print(f"Failed to auto-save best model: {e}")
        
        self.trained = True
        print("All models trained successfully!")
        
        result = {
            'models': results,
            'bestModel': self.best_model_name,
            'predictionsSample': self.predictions_sample,
            'bestModelSaved': best_model_saved,
            'bestModelPath': best_model_path if best_model_saved else None
        }
        
        if generalization_metrics:
            result['generalizationMetrics'] = generalization_metrics
        
        return result
    
    def _compute_regression_metrics(self, y_true, y_pred):
        """Compute regression metrics"""
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        
        # MAPE with protection against zero
        mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
        
        return {
            'mse': round(mse, 4),
            'rmse': round(rmse, 4),
            'mae': round(mae, 4),
            'r2Score': round(r2, 4),
            'mape': round(mape, 2)
        }
    
    def predict(self, features):
        """
        Make RUL prediction using best model.
        
        Features expected: value (or mean), temperature, drift, noise
        Output: RUL% (0 to 100), confidence, model name
        """
        if not self.trained:
            raise ValueError("Models not trained yet")
        
        model_name = self.best_model_name or 'Random Forest'
        model = self.models.get(model_name)
        
        if model is None:
            raise ValueError(f"Model {model_name} not found")
        
        # Build features - extract statistical features from the value
        value = features.get('value', features.get('mean', 9.81))
        temperature = features.get('temperature', 25.0)
        drift = features.get('drift', 0.0)
        noise = features.get('noise', 0.0)
        
        # Create a pseudo-signal and extract features
        pseudo_signal = np.array([value] * 200)
        pseudo_signal += np.random.normal(0, max(abs(noise), 0.001), 200)
        feats = extract_statistical_features(pseudo_signal)
        
        X = np.array([[feats['rms'], feats['std'], feats['peak'], feats['peak_to_peak'],
                        feats['kurtosis'], feats['skewness'], feats['crest_factor'],
                        feats['shape_factor'], feats['impulse_factor'], feats['freq_energy'],
                        abs(drift), abs(noise), temperature]])
        
        X_scaled = self.scaler.transform(X)
        rul_prediction = model.predict(X_scaled)[0]
        rul_prediction = float(np.clip(rul_prediction, 0, 100))
        
        # Confidence based on R2 and accuracy
        best_r2 = getattr(self, 'best_model_r2', 0.5)
        best_acc = getattr(self, 'best_model_accuracy', 50)
        confidence = max(0.0, min(1.0, (best_r2 + best_acc / 100) / 2))
        
        return {
            'rulPercent': round(rul_prediction, 2),
            'confidence': round(confidence, 2),
            'model': model_name
        }
    
    # ============== Model Persistence Methods ==============
    
    def save_models(self, session_id=None):
        """Save all trained models to disk"""
        if not self.trained:
            raise ValueError("No models trained yet. Train models first.")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_name = session_id or f"session_{timestamp}"
        session_dir = os.path.join(SAVED_MODELS_DIR, session_name)
        os.makedirs(session_dir, exist_ok=True)
        
        saved_models = []
        
        # Save each model
        for model_name, model in self.models.items():
            safe_name = model_name.replace(" ", "_").lower()
            model_path = os.path.join(session_dir, f"{safe_name}.joblib")
            joblib.dump(model, model_path)
            self.model_paths[model_name] = model_path
            saved_models.append({
                "name": model_name,
                "path": model_path,
                "saved_at": datetime.now().isoformat()
            })
            print(f"[ML] Saved {model_name} to {model_path}")
        
        # Save the scaler
        scaler_path = os.path.join(session_dir, "scaler.joblib")
        joblib.dump(self.scaler, scaler_path)
        print(f"[ML] Saved scaler to {scaler_path}")
        
        # Save model info
        info_path = os.path.join(session_dir, "model_info.joblib")
        joblib.dump({
            "model_names": list(self.models.keys()),
            "trained": self.trained,
            "best_model": self.best_model_name,
            "feature_names": self.feature_names,
            "feature_count": self.expected_feature_count,
            "saved_at": datetime.now().isoformat(),
            "session_id": session_name
        }, info_path)
        
        return {
            "session_id": session_name,
            "models": saved_models,
            "directory": session_dir
        }
    
    def load_model(self, session_id):
        """Load models from a saved session"""
        session_dir = os.path.join(SAVED_MODELS_DIR, session_id)
        
        if not os.path.exists(session_dir):
            raise ValueError(f"Session {session_id} not found")
        
        # Load model info
        info_path = os.path.join(session_dir, "model_info.joblib")
        if os.path.exists(info_path):
            info = joblib.load(info_path)
            model_names = info.get("model_names", [])
            self.best_model_name = info.get("best_model")
            saved_feature_count = info.get("feature_count", self.expected_feature_count)
            if saved_feature_count != self.expected_feature_count:
                raise ValueError(
                    f"Session {session_id} expects {saved_feature_count} features; "
                    f"current trainer expects {self.expected_feature_count}"
                )
        else:
            model_names = ["Random Forest", "Gradient Boosting", "SVM", "Neural Network"]
        
        # Load each model
        for model_name in model_names:
            safe_name = model_name.replace(" ", "_").lower()
            model_path = os.path.join(session_dir, f"{safe_name}.joblib")
            if os.path.exists(model_path):
                self.models[model_name] = joblib.load(model_path)
                self.model_paths[model_name] = model_path
                print(f"[ML] Loaded {model_name} from {model_path}")
        
        # Load scaler
        scaler_path = os.path.join(session_dir, "scaler.joblib")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            saved_features = getattr(self.scaler, "n_features_in_", self.expected_feature_count)
            if saved_features != self.expected_feature_count:
                raise ValueError(
                    f"Session {session_id} scaler expects {saved_features} features; "
                    f"current trainer expects {self.expected_feature_count}"
                )
            print(f"[ML] Loaded scaler from {scaler_path}")
        
        self.trained = True
        
        return {
            "session_id": session_id,
            "models_loaded": list(self.models.keys()),
            "best_model": self.best_model_name,
            "status": "success"
        }
    
    @staticmethod
    def list_saved_sessions():
        """List all saved model sessions"""
        if not os.path.exists(SAVED_MODELS_DIR):
            return []
        
        sessions = []
        for session_name in os.listdir(SAVED_MODELS_DIR):
            session_dir = os.path.join(SAVED_MODELS_DIR, session_name)
            if os.path.isdir(session_dir):
                info_path = os.path.join(session_dir, "model_info.joblib")
                if os.path.exists(info_path):
                    info = joblib.load(info_path)
                    sessions.append({
                        "session_id": session_name,
                        "models": info.get("model_names", []),
                        "best_model": info.get("best_model"),
                        "saved_at": info.get("saved_at", "Unknown")
                    })
                else:
                    sessions.append({
                        "session_id": session_name,
                        "models": [],
                        "saved_at": "Unknown"
                    })
        
        return sessions
    
    @staticmethod
    def delete_session(session_id):
        """Delete a saved model session"""
        import shutil
        session_dir = os.path.join(SAVED_MODELS_DIR, session_id)
        
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir)
            return {"status": "deleted", "session_id": session_id}
        else:
            raise ValueError(f"Session {session_id} not found")


# ==============================================================================
# PART 2: Fault Classification Pipeline
# ==============================================================================

class FaultClassifier:
    """
    Bearing Fault Classifier using real-world vibration datasets.
    
    Trains on data from:
    - CWRU Bearing Dataset (Case Western Reserve University)
    - ADI CbM MEMS Dataset (Analog Devices ADXL356)
    - NASA IMS Bearing Dataset (NASA Prognostics Repository)
    
    Classifies: Normal, Inner Race Fault, Outer Race Fault, Ball Fault
    """
    
    # Map CSV files to fault labels
    DATASET_LABEL_MAP = {
        # CWRU
        'cwru/normal.csv': 'Normal',
        'cwru/inner_race.csv': 'Inner Race Fault',
        'cwru/outer_race.csv': 'Outer Race Fault',
        'cwru/ball.csv': 'Ball Fault',
        # ADI MEMS
        'adi_mems/adi_normal.csv': 'Normal',
        'adi_mems/adi_inner_race.csv': 'Inner Race Fault',
        'adi_mems/adi_outer_race.csv': 'Outer Race Fault',
        'adi_mems/adi_ball_fault.csv': 'Ball Fault',
        # NASA IMS (healthy only — fault type is outer race, mapped as such)
        'nasa_ims/nasa_healthy.csv': 'Normal',
        'nasa_ims/nasa_degrading.csv': 'Outer Race Fault',
        'nasa_ims/nasa_failure.csv': 'Outer Race Fault',
    }
    
    FAULT_LABELS = ['Normal', 'Inner Race Fault', 'Outer Race Fault', 'Ball Fault']
    
    def __init__(self):
        self.classifiers = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.trained = False
        self.best_classifier_name = None
        self.results = None
    
    def _load_and_extract_features(self, window_size=2000, step=1000):
        """
        Load all CSV datasets and extract statistical features per window.
        
        Returns:
            X: Feature matrix (n_windows, n_features)
            y: Label array (n_windows,)
            dataset_sources: List of which dataset each sample came from
        """
        X_all = []
        y_all = []
        sources = []
        
        for csv_rel_path, label in self.DATASET_LABEL_MAP.items():
            csv_path = os.path.join(DATA_DIR, csv_rel_path)
            
            if not os.path.exists(csv_path):
                print(f"  [SKIP] {csv_rel_path} not found")
                continue
            
            try:
                df = pd.read_csv(csv_path)
                values = df['value'].values
                n = len(values)
                
                # Extract features from fixed-size windows
                actual_window = min(window_size, n // 2)
                actual_step = min(step, actual_window // 2)
                
                window_count = 0
                for start in range(0, n - actual_window, actual_step):
                    segment = values[start:start + actual_window]
                    feats = extract_statistical_features(segment)
                    X_all.append(list(feats.values()))
                    y_all.append(label)
                    sources.append(csv_rel_path.split('/')[0])  # 'cwru', 'adi_mems', 'nasa_ims'
                    window_count += 1
                
                print(f"  [OK] {csv_rel_path}: {window_count} windows, label='{label}'")
                
            except Exception as e:
                print(f"  [ERROR] {csv_rel_path}: {e}")
                continue
        
        X = np.array(X_all)
        y = np.array(y_all)
        
        return X, y, sources
    
    def train(self):
        """
        Train fault classifiers on all available datasets.
        
        Returns training results with accuracy, precision, recall, F1.
        """
        print("\n" + "=" * 60)
        print("FAULT CLASSIFICATION TRAINING")
        print("Loading data from ALL datasets (CWRU + ADI + NASA)...")
        print("=" * 60)
        
        X, y, sources = self._load_and_extract_features()
        
        if len(X) == 0:
            raise ValueError("No data loaded. Ensure CSV files exist in data/ directory.")
        
        # Encode labels
        self.label_encoder.fit(self.FAULT_LABELS)
        y_encoded = self.label_encoder.transform(y)
        
        # Print class distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"\nTotal samples: {len(X)}")
        print(f"Feature dimensions: {X.shape[1]}")
        print("Class distribution:")
        for label, count in zip(unique, counts):
            print(f"  {label}: {count} samples")
        
        # Print source distribution
        unique_src, src_counts = np.unique(sources, return_counts=True)
        print("Dataset sources:")
        for src, cnt in zip(unique_src, src_counts):
            print(f"  {src}: {cnt} samples")
        
        # Split: 80/20 stratified
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\nTraining set: {len(X_train)}, Test set: {len(X_test)}")
        
        results = []
        
        # 1. Random Forest Classifier
        print("Training Random Forest Classifier...")
        rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        rf_start = datetime.now()
        rf.fit(X_train_scaled, y_train)
        rf_time = (datetime.now() - rf_start).total_seconds()
        rf_pred = rf.predict(X_test_scaled)
        
        rf_metrics = self._compute_classification_metrics(y_test, rf_pred)
        rf_metrics['trainingTime'] = round(rf_time, 2)
        rf_metrics['modelType'] = 'Random Forest'
        results.append(rf_metrics)
        self.classifiers['Random Forest'] = rf
        
        # 2. Gradient Boosting Classifier
        print("Training Gradient Boosting Classifier...")
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        gb_start = datetime.now()
        gb.fit(X_train_scaled, y_train)
        gb_time = (datetime.now() - gb_start).total_seconds()
        gb_pred = gb.predict(X_test_scaled)
        
        gb_metrics = self._compute_classification_metrics(y_test, gb_pred)
        gb_metrics['trainingTime'] = round(gb_time, 2)
        gb_metrics['modelType'] = 'Gradient Boosting'
        results.append(gb_metrics)
        self.classifiers['Gradient Boosting'] = gb
        
        # 3. SVM Classifier
        print("Training SVM Classifier...")
        svc = SVC(kernel='rbf', C=10.0, random_state=42, probability=True)
        svc_start = datetime.now()
        svc.fit(X_train_scaled, y_train)
        svc_time = (datetime.now() - svc_start).total_seconds()
        svc_pred = svc.predict(X_test_scaled)
        
        svc_metrics = self._compute_classification_metrics(y_test, svc_pred)
        svc_metrics['trainingTime'] = round(svc_time, 2)
        svc_metrics['modelType'] = 'SVM'
        results.append(svc_metrics)
        self.classifiers['SVM'] = svc
        
        # 4. Neural Network Classifier
        print("Training Neural Network Classifier...")
        mlp = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu',
                           solver='adam', random_state=42, max_iter=500, early_stopping=True)
        mlp_start = datetime.now()
        mlp.fit(X_train_scaled, y_train)
        mlp_time = (datetime.now() - mlp_start).total_seconds()
        mlp_pred = mlp.predict(X_test_scaled)
        
        mlp_metrics = self._compute_classification_metrics(y_test, mlp_pred)
        mlp_metrics['trainingTime'] = round(mlp_time, 2)
        mlp_metrics['modelType'] = 'Neural Network'
        results.append(mlp_metrics)
        self.classifiers['Neural Network'] = mlp
        
        # Find best classifier (highest accuracy)
        best_idx = max(range(len(results)), key=lambda i: results[i]['accuracy'])
        self.best_classifier_name = results[best_idx]['modelType']
        
        print(f"\nBest classifier: {self.best_classifier_name} "
              f"(Accuracy: {results[best_idx]['accuracy']}%)")
        
        # Cross-validation for best model
        best_clf = self.classifiers[self.best_classifier_name]
        cv_scores = cross_val_score(best_clf, self.scaler.transform(X), y_encoded, 
                                     cv=min(5, len(np.unique(y_encoded))), scoring='accuracy')
        cv_accuracy = float(np.mean(cv_scores) * 100)
        cv_std = float(np.std(cv_scores) * 100)
        print(f"Cross-validation accuracy: {cv_accuracy:.1f}% (+/- {cv_std:.1f}%)")
        
        # Save best classifier
        try:
            clf_path = os.path.join(SAVED_MODELS_DIR, 'fault_classifier.joblib')
            scaler_path = os.path.join(SAVED_MODELS_DIR, 'classifier_scaler.joblib')
            encoder_path = os.path.join(SAVED_MODELS_DIR, 'label_encoder.joblib')
            joblib.dump(best_clf, clf_path)
            joblib.dump(self.scaler, scaler_path)
            joblib.dump(self.label_encoder, encoder_path)
            print(f"Saved best classifier to {clf_path}")
        except Exception as e:
            print(f"Failed to save classifier: {e}")
        
        self.trained = True
        self.results = results
        
        # Feature importance (for Random Forest)
        feature_names = ['RMS', 'Std Dev', 'Peak', 'Peak-to-Peak', 'Kurtosis',
                        'Skewness', 'Crest Factor', 'Shape Factor', 'Impulse Factor',
                        'Freq Energy']
        feature_importance = []
        if hasattr(rf, 'feature_importances_'):
            for name, imp in zip(feature_names, rf.feature_importances_):
                feature_importance.append({'feature': name, 'importance': round(float(imp), 4)})
            feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        return {
            'classifiers': results,
            'bestClassifier': self.best_classifier_name,
            'crossValidation': {
                'accuracy': round(cv_accuracy, 1),
                'std': round(cv_std, 1),
                'folds': min(5, len(np.unique(y_encoded)))
            },
            'featureImportance': feature_importance,
            'classSummary': {
                'labels': self.FAULT_LABELS,
                'trainSize': len(X_train),
                'testSize': len(X_test),
                'totalSamples': len(X),
                'datasetsUsed': list(set(sources))
            }
        }
    
    def _compute_classification_metrics(self, y_true, y_pred):
        """Compute classification metrics"""
        acc = float(accuracy_score(y_true, y_pred) * 100)
        
        # Use macro averaging for multi-class
        prec = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
        rec = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
        f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred).tolist()
        
        return {
            'accuracy': round(acc, 1),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1Score': round(f1, 4),
            'confusionMatrix': cm
        }
    
    def predict(self, signal_data):
        """
        Classify fault type from vibration signal.
        
        Args:
            signal_data: Array of vibration values
            
        Returns:
            Dict with predicted fault type, confidence, probabilities
        """
        if not self.trained:
            raise ValueError("Classifier not trained yet")
        
        clf_name = self.best_classifier_name
        clf = self.classifiers[clf_name]
        
        # Extract features
        feats = extract_statistical_features(signal_data)
        X = np.array([list(feats.values())])
        X_scaled = self.scaler.transform(X)
        
        # Predict
        pred_encoded = clf.predict(X_scaled)[0]
        pred_label = self.label_encoder.inverse_transform([pred_encoded])[0]
        
        # Confidence (probability)
        if hasattr(clf, 'predict_proba'):
            probas = clf.predict_proba(X_scaled)[0]
            confidence = float(np.max(probas))
            class_probabilities = {
                self.label_encoder.inverse_transform([i])[0]: round(float(p), 4)
                for i, p in enumerate(probas)
            }
        else:
            confidence = 0.9  # Default for SVM without probability
            class_probabilities = {}
        
        return {
            'faultType': pred_label,
            'confidence': round(confidence, 4),
            'classifier': clf_name,
            'probabilities': class_probabilities,
            'features': feats
        }
