import numpy as np
import pandas as pd

from typing import List, Dict
from sklearn.exceptions import NotFittedError

from .trainer import ModelTrainer
from ..data.processors import DataProcessor
from ..features.engineering import FeatureEngineer
from ..domain.entities import Team, TrainingFeatures

class PlacementPredictor:
    """Predict team placements using trained model"""
    
    def __init__(self, model_trainer: ModelTrainer):
        self.model_trainer = model_trainer
        self.feature_engineer = FeatureEngineer()
        self.data_processor = DataProcessor()
    
    def predict_placements(self, teams: List[Team]) -> List[Dict]:
        """Predict placements for all teams"""

        features = self.data_processor.extract_features(teams)
        
        df, X_processed = self.feature_engineer.prepare_features(features, fit_scaler=True)
        
        try:
            if self.model_trainer.is_trained:
                predictions = self.model_trainer.predict(X_processed)
            else:
                predictions = self._fallback_prediction(df)
        except NotFittedError:
            predictions = self._fallback_prediction(df)
        
        results = []
        for i, team in enumerate(teams):
            results.append({
                'team': team.name,
                'region': team.region,
                'predicted_placement': int(round(predictions[i])),
                'confidence_score': self._calculate_confidence(predictions[i]),
                'features': {
                    'roster_stability': features[i].roster_stability,
                    'avg_player_rating': features[i].avg_player_rating,
                    'win_rate': features[i].win_rate,
                    'recent_form': features[i].recent_form
                }
            })
        
        results.sort(key=lambda x: x['predicted_placement'])
        
        self._adjust_placements(results)
        
        return results
    
    def _fallback_prediction(self, df: pd.DataFrame) -> np.ndarray:
        """Fallback prediction method if no model is trained"""

        print("🔧 Usando predição baseada em regras (fallback)")
        
        weights = np.array([0.15, 0.35, 0.5, 0.3, 0.2])
        scores = df[['roster_stability', 'avg_player_rating', 'win_rate', 'recent_form', 'strength_of_schedule']].values.dot(weights)
        
        normalized_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        placements = 1 + (1 - normalized_scores) * 7
        
        return placements
    
    def _calculate_confidence(self, prediction: float) -> float:
        """Calculate confidence score for prediction"""

        middle = 4.5
        distance_from_middle = abs(prediction - middle)
        confidence = distance_from_middle / 3.5
        
        return max(0.3, min(0.9, confidence))
    
    def _adjust_placements(self, results: List[Dict]) -> None:
        """Adjust placements to avoid ties and ensure proper ranking"""

        used_placements = set()
        
        for i, result in enumerate(results):
            placement = result['predicted_placement']
            
            while placement in used_placements and placement < 8:
                placement += 1
            
            placement = max(1, min(8, placement))
            result['predicted_placement'] = placement
            used_placements.add(placement)
        
        sorted_placements = sorted(used_placements)
        for i, result in enumerate(results):
            result['predicted_placement'] = i + 1