import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import os
from datetime import datetime

# Modular imports
from ai_models.data_preprocessing import DataPreprocessor
from ai_models.model_training import ModelTrainer
from ai_models.evaluation import ModelEvaluator
from ai_models.prediction_engine import PredictionEngine
from utils.helpers import logger

class AIPredictionPipeline:
    """
    Main orchestration layer for the Crypto AI platform.
    Connects data, models, and evaluation into a single workflow.
    """
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.evaluator = ModelEvaluator()
        self.engine = PredictionEngine()
        
    def run_training_cycle(self, df: pd.DataFrame, asset_id: str) -> Dict:
        """
        Full pipeline: Clean -> Feature Eng -> Split -> Train -> Eval -> Save
        """
        logger.info(f"Starting full training cycle for {asset_id}...")
        
        # 1. Preprocessing
        clean_df = self.preprocessor.clean_data(df)
        eng_df = self.preprocessor.engineer_features(clean_df)
        
        # 2. Scaling
        # We'll use the first 80% for training + validation scaling logic
        scaled_data, features = self.preprocessor.get_scaled_data(eng_df, asset_id, is_training=True)
        
        # 3. Preparation (Sequences for LSTM)
        X, y = self.preprocessor.prepare_lstm_sequences(scaled_data)
        if len(X) < 50:
             return {"error": "Insufficient data for training"}
             
        # 4. Splitting
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.preprocessor.split_data(X, y)
        
        # 5. Training
        trainer = ModelTrainer(asset_id)
        
        # Train multiple models
        logger.info("Training Linear Regression...")
        trainer.train_linear_regression(X_train.reshape(len(X_train), -1), y_train)
        
        logger.info("Training Random Forest...")
        trainer.train_random_forest(X_train.reshape(len(X_train), -1), y_train)
        
        logger.info("Training XGBoost...")
        trainer.train_xgboost(X_train.reshape(len(X_train), -1), y_train)
        
        logger.info("Training LSTM...")
        trainer.train_lstm(X_train, y_train, X_val, y_val)
        
        # 6. Evaluation on Test Set
        metrics = {}
        for name, model in trainer.models.items():
            if name == "lstm":
                import torch
                model.eval()
                with torch.no_grad():
                    preds = model(torch.FloatTensor(X_test)).numpy().flatten()
            else:
                preds = model.predict(X_test.reshape(len(X_test), -1))
                
            metrics[name] = self.evaluator.calculate_metrics(y_test, preds)
            
        # 7. Save Models
        trainer.save_models(features)
        
        # Save validation results for the Engine's weighting logic
        # For now, we'll just return them
        return {
            "asset_id": asset_id,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }

    def get_prediction(self, df: pd.DataFrame, asset_id: str, days_ahead: int = 7) -> Dict:
        """
        Load models and produce ensemble forecast.
        """
        # Load available models for the engine
        model_names = ["linear_regression", "random_forest", "xgboost", "lstm"]
        self.engine.load_asset_models(asset_id, model_names)
        
        # Run forecast
        return self.engine.forecast(df, asset_id, days_ahead)
