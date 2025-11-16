import os
import sys
import pytest
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.trainer import ModelTrainer
from src.data.processors import DataProcessor
from src.data.collectors import VLRDataCollector
from src.models.predictor import PlacementPredictor
from src.features.engineering import FeatureEngineer

class TestIntegration:
    def test_end_to_end_with_mock_data(self):
        """Test the complete pipeline with mock data"""

        collector = VLRDataCollector()
        teams = collector.get_qualified_teams()
        
        assert len(teams) > 0, "Should load qualified teams"
        
        processor = DataProcessor()
        features = processor.extract_features(teams)
        
        assert len(features) == len(teams), "Should extract features for all teams"
        
        engineer = FeatureEngineer()
        df, X_processed = engineer.prepare_features(features, fit_scaler=True)
        
        assert X_processed.shape[0] == len(teams), "Should process features for all teams"
        
        if len(teams) >= 3:
            trainer = ModelTrainer()
            
            y_synthetic = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10][:len(teams)])
            
            training_results = trainer.train_models(X_processed, y_synthetic)
            
            predictor = PlacementPredictor(trainer)
            results = predictor.predict_placements(teams)
            
            assert len(results) == len(teams), "Should predict placements for all teams"
            assert all('predicted_placement' in result for result in results)
            assert all('confidence_score' in result for result in results)
    
    def test_feature_consistency(self):
        """Test that features are calculated consistently"""

        collector = VLRDataCollector()
        teams = collector.get_qualified_teams()
        
        processor = DataProcessor()
        features1 = processor.extract_features(teams)
        features2 = processor.extract_features(teams)
        
        for f1, f2 in zip(features1, features2):
            assert f1.roster_stability == f2.roster_stability
            assert f1.avg_player_rating == f2.avg_player_rating
            assert f1.win_rate == f2.win_rate