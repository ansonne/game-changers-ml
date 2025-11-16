import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.trainer import ModelTrainer
from src.models.predictor import PlacementPredictor

class TestModelComponents:
    def test_model_trainer_initialization(self):
        trainer = ModelTrainer()
        assert trainer is not None
        assert len(trainer.models) == 3
        assert trainer.best_model is None
        assert not trainer.is_trained
    
    def test_model_training_with_sufficient_data(self):
        trainer = ModelTrainer()
        
        X = np.random.randn(20, 9)
        y = np.random.randint(1, 9, 20)
        
        results = trainer.train_models(X, y)
        
        assert trainer.is_trained
        assert trainer.best_model is not None
        assert trainer.best_model_name in trainer.models.keys()
    
    def test_model_training_with_insufficient_data(self):
        trainer = ModelTrainer()
        
        X = np.random.randn(2, 9)
        y = np.random.randint(1, 9, 2)
        
        results = trainer.train_models(X, y)
        
        assert not trainer.is_trained
        assert results is None
    
    def test_predictor_initialization(self):
        trainer = ModelTrainer()
        predictor = PlacementPredictor(trainer)
        
        assert predictor is not None
        assert predictor.model_trainer == trainer
    
    def test_fallback_prediction(self):
        trainer = ModelTrainer()
        predictor = PlacementPredictor(trainer)
        
        df = pd.DataFrame({
            'roster_stability': [0.8, 0.5],
            'avg_player_rating': [1.1, 0.9],
            'avg_acs': [210.0, 190.0],
            'avg_kdr': [1.05, 0.95],
            'avg_kast': [0.75, 0.65],
            'win_rate': [0.7, 0.4],
            'recent_form': [0.6, 0.3],
            'strength_of_schedule': [0.5, 0.6],
            'tournament_results': [0.8, 0.5]
        })
        
        predictions = predictor._fallback_prediction(df)
        
        assert len(predictions) == len(df)
        assert all(1 <= p <= 8 for p in predictions)