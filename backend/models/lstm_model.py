"""
LSTM Neural Network Model for MEMS Sensor RUL Prediction

This module provides LSTM-based deep learning for time-series 
Remaining Useful Life (RUL) prediction.
"""

import numpy as np
import os
import json
from datetime import datetime

# Try to import TensorFlow - graceful fallback if not available
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("[LSTM] TensorFlow not installed. LSTM features will be disabled.")


class LSTMPredictor:
    """
    LSTM Neural Network for MEMS Sensor RUL Prediction
    
    Features:
    - Time-series sequence learning
    - Configurable architecture
    - Training progress tracking
    - Model save/load functionality
    """
    
    def __init__(self, sequence_length=50, n_features=5):
        """
        Initialize LSTM Predictor
        
        Args:
            sequence_length: Number of time steps in each sequence
            n_features: Number of features per time step
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.training_history = None
        self.is_trained = False
        self.model_path = os.path.join(os.path.dirname(__file__), 'saved_models', 'lstm_model.h5')
        
    def build_model(self, lstm_units=64, dense_units=32, dropout_rate=0.2):
        """
        Build LSTM model architecture
        
        Architecture:
        - LSTM layer (with return sequences)
        - Dropout for regularization
        - LSTM layer
        - Dense layers for output
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not installed. Cannot build LSTM model.")
        
        self.model = Sequential([
            # First LSTM layer
            LSTM(lstm_units, return_sequences=True, 
                 input_shape=(self.sequence_length, self.n_features)),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            # Second LSTM layer
            LSTM(lstm_units // 2, return_sequences=False),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            # Dense layers
            Dense(dense_units, activation='relu'),
            Dropout(dropout_rate / 2),
            Dense(dense_units // 2, activation='relu'),
            
            # Output layer - RUL prediction
            Dense(1, activation='linear')
        ])
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return self.model.summary()
    
    def prepare_sequences(self, data, target=None):
        """
        Prepare time-series sequences for LSTM
        
        Args:
            data: Raw sensor data (n_samples, n_features)
            target: Target RUL values (optional for prediction)
            
        Returns:
            X: Sequences of shape (n_sequences, sequence_length, n_features)
            y: Target values (if provided)
        """
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            if target is not None:
                y.append(target[i + self.sequence_length])
        
        X = np.array(X)
        
        if target is not None:
            y = np.array(y)
            return X, y
        
        return X
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=50, batch_size=32, verbose=1):
        """
        Train LSTM model
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences (optional)
            y_val: Validation targets (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity level (0, 1, or 2)
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        # Validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Train
        self.training_history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_trained = True
        
        return {
            'loss': self.training_history.history['loss'],
            'mae': self.training_history.history['mae'],
            'val_loss': self.training_history.history.get('val_loss', []),
            'val_mae': self.training_history.history.get('val_mae', []),
            'epochs_trained': len(self.training_history.history['loss'])
        }
    
    def predict(self, X):
        """
        Make RUL predictions
        
        Args:
            X: Input sequences
            
        Returns:
            Predicted RUL values
        """
        if not self.is_trained and self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X, verbose=0).flatten()
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test: Test sequences
            y_test: True RUL values
            
        Returns:
            Dictionary with MSE and MAE metrics
        """
        predictions = self.predict(X_test)
        
        mse = np.mean((predictions - y_test) ** 2)
        mae = np.mean(np.abs(predictions - y_test))
        rmse = np.sqrt(mse)
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'predictions': predictions.tolist()
        }
    
    def save_model(self):
        """Save trained model to disk"""
        if self.model is None:
            raise ValueError("No model to save")
        
        # Create directory if not exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Save model
        self.model.save(self.model_path)
        
        # Save metadata
        metadata = {
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'is_trained': self.is_trained,
            'saved_at': datetime.now().isoformat()
        }
        
        metadata_path = self.model_path.replace('.h5', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return self.model_path
    
    def load_model(self):
        """Load trained model from disk"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        self.model = load_model(self.model_path)
        
        # Load metadata
        metadata_path = self.model_path.replace('.h5', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.sequence_length = metadata.get('sequence_length', 50)
                self.n_features = metadata.get('n_features', 5)
                self.is_trained = metadata.get('is_trained', True)
        
        return True


def generate_synthetic_rul_data(num_samples=1000, sequence_length=50, n_features=5):
    """
    Generate synthetic sensor data with RUL labels for training
    
    Simulates degradation patterns over time
    """
    # Generate base features
    np.random.seed(42)
    
    # Create degrading signals
    time = np.linspace(0, 10, num_samples)
    degradation = np.exp(time / 5)  # Exponential degradation
    
    # Features: vibration, temperature, pressure, noise, drift
    features = np.column_stack([
        np.sin(time * 2) + degradation * 0.1 + np.random.normal(0, 0.1, num_samples),  # Vibration
        20 + degradation * 2 + np.random.normal(0, 0.5, num_samples),  # Temperature
        100 - degradation * 3 + np.random.normal(0, 1, num_samples),  # Pressure
        np.random.normal(0, 0.1, num_samples) * degradation,  # Noise
        np.cumsum(np.random.normal(0, 0.01, num_samples))  # Drift
    ])
    
    # RUL: starts at 100, decreases over time
    rul = 100 - (np.arange(num_samples) / num_samples) * 100
    
    return features, rul


# Quick test if run directly
if __name__ == "__main__":
    if TENSORFLOW_AVAILABLE:
        print("Testing LSTM Predictor...")
        
        # Generate test data
        features, rul = generate_synthetic_rul_data(500)
        
        # Create predictor
        predictor = LSTMPredictor(sequence_length=20, n_features=5)
        predictor.build_model()
        
        # Prepare sequences
        X, y = predictor.prepare_sequences(features, rul)
        
        # Split data
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        # Train
        history = predictor.train(X_train, y_train, X_val, y_val, epochs=5, verbose=1)
        print(f"Training complete. Final loss: {history['loss'][-1]:.4f}")
        
        # Evaluate
        metrics = predictor.evaluate(X_val, y_val)
        print(f"Validation RMSE: {metrics['rmse']:.2f}")
    else:
        print("TensorFlow not available. Install with: pip install tensorflow")
