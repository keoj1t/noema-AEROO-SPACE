"""
CHRONOS TIME-SERIES ANOMALY DETECTION

Purpose:
- Inference-only anomaly detection using statistical forecasting
- Analyzes satellite telemetry as black-box time series
- No online training, no labels required

Input:
- Time-series windows: battery_temp, voltage, orientation_error

Output:
- Forecasted values with uncertainty intervals
- Residuals (actual - predicted)
- Anomaly scores per timestep
- Derived: health score, regime shifts, weak signals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import json

class ChronosAnomalyDetector:
    """
    Lightweight Chronos-inspired anomaly detector.
    Uses statistical forecasting: EWMA + dynamic thresholds.
    """
    
    def __init__(self, window_size: int = 5, sensitivity: float = 1.5):
        self.window_size = window_size
        self.sensitivity = sensitivity
        
    def _exponential_weighted_ma(self, values: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """Compute exponential weighted moving average"""
        ewma = np.zeros_like(values, dtype=float)
        ewma[0] = values[0]
        
        for t in range(1, len(values)):
            ewma[t] = alpha * values[t] + (1 - alpha) * ewma[t - 1]
        
        return ewma
    
    def _detect_regime_shift(self, values: np.ndarray, window: int = 3) -> float:
        """Detect sudden changes in distribution"""
        if len(values) < 2 * window:
            return 0.0
        
        early = values[:window]
        recent = values[-window:]
        
        # Compare means and stds
        mean_shift = abs(np.mean(recent) - np.mean(early)) / (np.std(early) + 1e-6)
        std_shift = abs(np.std(recent) - np.std(early)) / (np.std(early) + 1e-6)
        
        return (mean_shift + std_shift) / 2.0
    
    def analyze(self, 
                df: pd.DataFrame,
                columns: List[str] = None) -> Dict[str, Any]:
        """
        Analyze telemetry for anomalies using statistical forecasting.
        
        Args:
            df: DataFrame with time-series data
            columns: Columns to analyze
        
        Returns:
            Dict with anomaly scores, forecasts, health metrics
        """
        if columns is None:
            columns = ['battery_temp', 'voltage', 'orientation_error']
        
        results = {
            'timestamps': [str(ts) for ts in df['time'].tolist()] if 'time' in df.columns else list(range(len(df))),
            'anomaly_scores': np.zeros(len(df)).tolist(),
            'forecasts': {},
            'residuals': {},
            'health_scores': np.zeros(len(df)).tolist(),
            'weak_signals': [],
            'regime_shifts': [],
            'confidence': 0.0
        }
        
        # Analyze each telemetry channel
        combined_anomalies = np.zeros(len(df))
        
        for col in columns:
            if col not in df.columns:
                continue
            
            values = df[col].values.astype(float)
            values = np.nan_to_num(values, nan=np.nanmean(values))
            
            # Forecast using EWMA
            forecast = self._exponential_weighted_ma(values, alpha=0.3)
            residuals = values - forecast
            
            # Compute residual statistics
            residual_std = np.std(residuals)
            residual_mean = np.mean(residuals)
            
            # Anomaly score: normalized residual magnitude
            z_scores = (residuals - residual_mean) / (residual_std + 1e-6)
            channel_anomalies = np.clip(np.abs(z_scores) / 3.0, 0, 1.0)
            
            # Regime shift detection
            regime_anomaly = self._detect_regime_shift(values) / 10.0
            channel_anomalies = np.maximum(channel_anomalies, regime_anomaly)
            
            results['forecasts'][col] = forecast.tolist()
            results['residuals'][col] = residuals.tolist()
            combined_anomalies += channel_anomalies
        
        # Normalize combined anomaly score
        num_channels = len([c for c in columns if c in df.columns])
        if num_channels > 0:
            combined_anomalies /= num_channels
        
        results['anomaly_scores'] = np.clip(combined_anomalies, 0, 1.0).tolist()
        
        # Derive health scores (inverse of anomaly with lag)
        health_scores = 100 * (1 - np.array(results['anomaly_scores']))
        
        # Apply smoothing to health scores
        health_scores = self._exponential_weighted_ma(health_scores, alpha=0.2)
        results['health_scores'] = np.clip(health_scores, 0, 100).tolist()
        
        # Detect weak signals (first significant anomaly)
        threshold = 0.3
        first_weak_idx = np.where(np.array(results['anomaly_scores']) > threshold)[0]
        if len(first_weak_idx) > 0:
            idx = int(first_weak_idx[0])
            results['weak_signals'].append({
                'timestamp': results['timestamps'][idx],
                'index': idx,
                'anomaly_score': float(results['anomaly_scores'][idx]),
                'severity': 'weak' if results['anomaly_scores'][idx] < 0.6 else 'strong'
            })
        
        # Detect regime shifts
        for col in columns:
            if col in df.columns:
                values = df[col].values.astype(float)
                values = np.nan_to_num(values, nan=np.nanmean(values))
                shift_score = self._detect_regime_shift(values)
                if shift_score > 1.5:
                    results['regime_shifts'].append({
                        'channel': col,
                        'shift_score': float(shift_score)
                    })
        
        # Compute confidence based on variance in signals
        anomaly_variance = np.var(results['anomaly_scores'])
        results['confidence'] = float(np.clip(anomaly_variance, 0, 1.0))
        
        # Ensure all numpy types converted to Python native types
        results['anomaly_scores'] = [float(x) for x in results['anomaly_scores']]
        results['health_scores'] = [float(x) for x in results['health_scores']]
        results['confidence'] = float(results['confidence'])
        
        return results

def detect_chronos_anomalies(df: pd.DataFrame,
                            columns: List[str] = None) -> Dict[str, Any]:
    """
    Public interface for Chronos-based anomaly detection.
    
    Args:
        df: Input telemetry DataFrame
        columns: Columns to analyze
    
    Returns:
        Analysis results dict
    """
    detector = ChronosAnomalyDetector(window_size=5, sensitivity=1.5)
    return detector.analyze(df, columns)
