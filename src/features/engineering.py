import pandas as pd
import numpy as np

from typing import List, Tuple, Optional
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from ..domain.entities import TrainingFeatures

class FeatureEngineer:
    """Feature engineering and preprocessing"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.is_fitted = False
    
    def prepare_features(self, features: List[TrainingFeatures], fit_scaler: bool = False) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepare features for model training"""

        feature_data = []
        for feature in features:
            feature_data.append({
                'roster_stability': feature.roster_stability,
                'avg_player_rating': feature.avg_player_rating,
                'avg_acs': feature.avg_acs,
                'avg_kdr': feature.avg_kdr,
                'avg_kast': feature.avg_kast,
                'win_rate': feature.win_rate,
                'recent_form': feature.recent_form,
                'strength_of_schedule': feature.strength_of_schedule,
                'tournament_results': feature.tournament_results
            })
        
        df = pd.DataFrame(feature_data)
        
        if fit_scaler or not self.is_fitted:
            df_imputed = self.imputer.fit_transform(df)
            df_scaled = self.scaler.fit_transform(df_imputed)
            self.is_fitted = True
        else:
            df_imputed = self.imputer.transform(df)
            df_scaled = self.scaler.transform(df_imputed)
        
        return df, df_scaled
    
    def create_composite_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite features that might improve model performance"""

        df = df.copy()
        
        df['stability_performance'] = df['roster_stability'] * df['avg_player_rating']
        
        df['form_consistency'] = df['win_rate'] * df['recent_form']
        
        df['individual_skill'] = (
            df['avg_player_rating'] * 0.4 +
            df['avg_acs'] * 0.2 +
            df['avg_kdr'] * 0.2 +
            df['avg_kast'] * 0.2
        ) / 200
        
        df['tournament_impact'] = df['tournament_results'] * df['strength_of_schedule']
        
        df['team_strength_index'] = (
            df['individual_skill'] * 0.3 +
            df['win_rate'] * 0.25 +
            df['recent_form'] * 0.2 +
            df['roster_stability'] * 0.15 +
            df['tournament_impact'] * 0.1
        )
        
        return df