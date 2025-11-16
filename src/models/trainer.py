import os
import joblib
import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge
from typing import Tuple, Dict, Any, Optional
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from ..utils.config import Config

class ModelTrainer:
    """Machine Learning model trainer following SRP"""
    
    def __init__(self):
        self.config = Config()
        self.models = {
            'random_forest': RandomForestRegressor(random_state=42),
            'gradient_boosting': GradientBoostingRegressor(random_state=42),
            'ridge': Ridge(random_state=42)
        }
        self.best_model = None
        self.best_model_name = None
        self.is_trained = False
    
    def train_models(self, X: np.ndarray, y: np.ndarray) -> Optional[Dict[str, Any]]:
        """Train multiple models and select the best one"""

        if len(X) < 3:
            print("⚠️  Dados insuficientes para treinamento do modelo")
            self.is_trained = False
            return None
        
        results = {}
        
        try:
            for name, model in self.models.items():
                cv_folds = min(3, len(X))
                cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='neg_mean_absolute_error')
                results[name] = {
                    'cv_mae_mean': -cv_scores.mean(),
                    'cv_mae_std': cv_scores.std(),
                    'model': model
                }
                
                model.fit(X, y)
            
            self.best_model_name = min(results.keys(), 
                                     key=lambda x: results[x]['cv_mae_mean'])
            self.best_model = results[self.best_model_name]['model']
            self.is_trained = True
            
            return results
            
        except Exception as e:
            print(f"❌ Erro no treinamento: {e}")
            self.is_trained = False
            return None
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the best model"""

        if self.best_model is not None and self.is_trained:
            return self.best_model.predict(X)
        else:
            raise NotFittedError("Modelo não foi treinado ainda")
    
    def hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray) -> None:
        """Perform hyperparameter tuning for the best model"""

        if not self.is_trained or len(X) < 5:
            print("⚠️  Dados insuficientes para tuning de hiperparâmetros")
            return
        
        if self.best_model_name == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [5, 10],
            }
        elif self.best_model_name == 'gradient_boosting':
            param_grid = {
                'n_estimators': [50, 100],
                'learning_rate': [0.05, 0.1],
            }
        else:
            param_grid = {
                'alpha': [0.1, 1.0, 10.0]
            }
        
        try:
            grid_search = GridSearchCV(
                self.models[self.best_model_name],
                param_grid,
                cv=min(3, len(X)),
                scoring='neg_mean_absolute_error'
            )
            
            grid_search.fit(X, y)
            self.best_model = grid_search.best_estimator_
            print(f"✅ Tuning completo. Melhores parâmetros: {grid_search.best_params_}")
            
        except Exception as e:
            print(f"❌ Erro no tuning: {e}")
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model"""
        
        if self.best_model is not None and self.is_trained:
            joblib.dump({
                'model': self.best_model,
                'model_name': self.best_model_name,
                'is_trained': self.is_trained
            }, filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model"""

        if os.path.exists(filepath):
            loaded = joblib.load(filepath)
            self.best_model = loaded['model']
            self.best_model_name = loaded['model_name']
            self.is_trained = loaded['is_trained']