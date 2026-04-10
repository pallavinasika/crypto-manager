import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict

class ModelEvaluator:
    """
    Standardized evaluation metrics for all models.
    """
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate and return a dictionary of metrics."""
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # 1. MAE
        mae = float(mean_absolute_error(y_true, y_pred))
        
        # 2. RMSE
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        
        # 3. R2 Score (Capped at 1.0, minimum 0.0 for stability in UI)
        r2 = float(r2_score(y_true, y_pred))
        
        # 4. MAPE (Mean Absolute Percentage Error)
        # Avoid division by zero and cap at 100%
        nonzero_idx = np.abs(y_true) > 1e-9
        if np.any(nonzero_idx):
            mape = np.mean(np.abs((y_true[nonzero_idx] - y_pred[nonzero_idx]) / y_true[nonzero_idx])) * 100
            mape = float(min(mape, 100.0))
        else:
            mape = 100.0
            
        # 5. Confidence Score (Normalized 0.0 to 1.0 based on R2 and MAPE)
        r2_comp = max(0, min(1, r2))
        mape_comp = max(0, (100 - mape) / 100)
        confidence = (r2_comp * 0.4) + (mape_comp * 0.6)
        
        return {
            "mae": mae,
            "rmse": rmse,
            "r2_score": max(0, r2),  # UI-friendly R2
            "mape": mape,
            "confidence": float(confidence),
            "status": "reliable" if confidence > 0.6 else "unreliable"
        }
