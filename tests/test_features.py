import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.domain.entities import TrainingFeatures
from src.features.engineering import FeatureEngineer

class TestFeatureEngineering:
    def test_feature_engineer_initialization(self):
        engineer = FeatureEngineer()
        assert engineer is not None
        assert not engineer.is_fitted
    
    def test_feature_preparation_with_fit(self):
        engineer = FeatureEngineer()
        
        test_features = [
            TrainingFeatures(
                team_id='1',
                roster_stability=0.8,
                avg_player_rating=1.1,
                avg_acs=210.0,
                avg_kdr=1.05,
                avg_kast=0.75,
                win_rate=0.7,
                recent_form=0.6,
                strength_of_schedule=0.5,
                tournament_results=0.8
            )
        ]
        
        df, processed = engineer.prepare_features(test_features, fit_scaler=True)
        
        assert isinstance(df, pd.DataFrame)
        assert isinstance(processed, np.ndarray)
        assert processed.shape[0] == len(test_features)
        assert engineer.is_fitted
    
    def test_composite_features(self):
        engineer = FeatureEngineer()
        
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
        
        enhanced_df = engineer.create_composite_features(df)
        
        expected_columns = [
            'stability_performance', 
            'form_consistency', 
            'individual_skill',
            'tournament_impact',
            'team_strength_index'
        ]
        
        for col in expected_columns:
            assert col in enhanced_df.columns
        
        assert 'team_strength_index' in enhanced_df.columns
        assert enhanced_df['team_strength_index'].between(0, 1).all()