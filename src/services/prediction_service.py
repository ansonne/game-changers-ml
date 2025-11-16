import numpy as np
import pandas as pd

from typing import List, Dict, Optional, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ..models.trainer import ModelTrainer
from ..data.processors import DataProcessor
from ..data.collectors import VLRDataCollector
from ..models.predictor import PlacementPredictor
from ..features.engineering import FeatureEngineer
from ..domain.entities import Team, TrainingFeatures
from ..domain.tournament import TournamentConfig, TournamentManager

class PredictionService:
    """Generic prediction service with historical data training"""
    
    def __init__(self, config: Dict):
        self.tournament_manager = TournamentManager(config)
        self.data_collector = VLRDataCollector()
        self.data_processor = DataProcessor()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        self.predictor: Optional[PlacementPredictor] = None
        self.use_historical_data = config.get('model', {}).get('use_historical_data', True)
        
    def list_available_tournaments(self) -> List[str]:
        """List all available tournaments"""
        return self.tournament_manager.list_tournaments()
    
    def train_with_historical_data(self) -> Tuple[bool, Dict]:
        """Train model using historical tournament data"""
        historical_tournaments = self.tournament_manager.get_historical_tournaments()
        
        if not historical_tournaments:
            print("⚠️  Nenhum torneio histórico encontrado para treinamento")
            return False, {}
        
        print(f"📚 Treinando modelo com {len(historical_tournaments)} torneio(s) histórico(s)")
        
        all_features = []
        all_targets = []
        
        for tournament in historical_tournaments:
            print(f"🔍 Processando {tournament.name}...")
            
            teams = self.data_collector.get_teams(tournament)
            actual_placements = tournament.get_actual_placements()
            
            features = self.data_processor.extract_features(teams)
            
            for feature in features:
                team_placement = actual_placements.get(feature.team_id)
                if team_placement:
                    feature.target = team_placement
                    all_features.append(feature)
                    all_targets.append(team_placement)

        if len(all_features) < 3:
            print("❌ Dados insuficientes para treinamento histórico")
            return False, {}
        
        feature_df, X_processed = self.feature_engineer.prepare_features(all_features, fit_scaler=True)
        y = np.array(all_targets)
        
        print(f"✅ Dados de treinamento: {X_processed.shape[0]} amostras, {X_processed.shape[1]} features")
        
        training_results = self.model_trainer.train_models(X_processed, y)
        
        if self.model_trainer.is_trained:
            predictions = self.model_trainer.predict(X_processed)
            mae = mean_absolute_error(y, predictions)
            mse = mean_squared_error(y, predictions)
            
            performance = {
                'mae': mae,
                'mse': mse,
                'rmse': np.sqrt(mse),
                'samples': len(y),
                'best_model': self.model_trainer.best_model_name
            }
            
            print(f"✅ Modelo treinado com MAE: {mae:.2f}, RMSE: {np.sqrt(mse):.2f}")
            return True, performance
        else:
            print("❌ Falha no treinamento do modelo")
            return False, {}
    
    def predict_tournament(self, tournament_key: str, include_actual: bool = False) -> Dict:
        """Make predictions for a specific tournament"""
        tournament = self.tournament_manager.get_tournament(tournament_key)
        if not tournament:
            raise ValueError(f"Torneio não encontrado: {tournament_key}")
        
        print(f"🎯 Iniciando predições para: {tournament.name}")
        
        if self.use_historical_data:
            trained, performance = self.train_with_historical_data()
            if not trained:
                print("⚠️  Usando dados sintéticos para treinamento")
                self._train_with_synthetic_data()
        else:
            self._train_with_synthetic_data()
        
        teams = self.data_collector.get_teams(tournament)
        
        features = self.data_processor.extract_features(teams)
        df, X_processed = self.feature_engineer.prepare_features(features, fit_scaler=False)
        
        self.predictor = PlacementPredictor(self.model_trainer)
        predictions = self.predictor.predict_placements(teams)
        
        if include_actual and tournament.is_historical:
            actual_placements = tournament.get_actual_placements()
            for prediction in predictions:
                team_id = next((team.id for team in teams if team.name == prediction['team']), None)
                if team_id and team_id in actual_placements:
                    prediction['actual_placement'] = actual_placements[team_id]
        
        return {
            'tournament': tournament.name,
            'predictions': predictions,
            'teams_analyzed': len(teams),
            'features_used': X_processed.shape[1],
            'model_performance': performance if self.use_historical_data else None
        }
    
    def validate_historical_predictions(self, tournament_key: str) -> Dict:
        """Validate predictions against actual historical results"""
        tournament = self.tournament_manager.get_tournament(tournament_key)
        if not tournament or not tournament.is_historical:
            raise ValueError(f"Torneio histórico não encontrado: {tournament_key}")
        
        print(f"🔍 Validando predições para {tournament.name}...")
        
        result = self.predict_tournament(tournament_key, include_actual=True)
        
        predicted_placements = []
        actual_placements = []
        
        for prediction in result['predictions']:
            if 'actual_placement' in prediction:
                predicted_placements.append(prediction['predicted_placement'])
                actual_placements.append(prediction['actual_placement'])
        
        if len(predicted_placements) > 0:
            mae = mean_absolute_error(actual_placements, predicted_placements)
            mse = mean_squared_error(actual_placements, predicted_placements)
            accuracy = self._calculate_placement_accuracy(actual_placements, predicted_placements)
            
            validation_metrics = {
                'mae': mae,
                'mse': mse,
                'rmse': np.sqrt(mse),
                'accuracy': accuracy,
                'samples': len(predicted_placements)
            }
            
            result['validation_metrics'] = validation_metrics
            print(f"✅ Validação - MAE: {mae:.2f}, Acurácia: {accuracy:.1%}")
        
        return result
    
    def _calculate_placement_accuracy(self, actual: List[int], predicted: List[int]) -> float:
        """Calculate accuracy of placement predictions"""
        correct = 0
        total = len(actual)
        
        for a, p in zip(actual, predicted):
            if abs(a - p) <= 1:
                correct += 1
        
        return correct / total if total > 0 else 0.0
    
    def _train_with_synthetic_data(self, num_samples: int = 50):
        """Train with synthetic data as fallback"""
        print("🔧 Gerando dados sintéticos para treinamento...")
        X_synthetic, y_synthetic = self._generate_synthetic_training_data(num_samples)
        self.model_trainer.train_models(X_synthetic, y_synthetic)
    
    def _generate_synthetic_training_data(self, num_samples: int = 50) -> tuple:
        """Generate synthetic training data"""
        np.random.seed(42)
        
        X_synthetic = np.column_stack([
            np.random.uniform(0.2, 1.0, num_samples),
            np.random.uniform(0.9, 1.3, num_samples),
            np.random.uniform(160, 280, num_samples),
            np.random.uniform(0.7, 1.4, num_samples),
            np.random.uniform(0.6, 0.9, num_samples),
            np.random.uniform(0.3, 0.9, num_samples),
            np.random.uniform(0.3, 0.9, num_samples),
            np.random.uniform(0.4, 0.9, num_samples),
            np.random.uniform(0.3, 0.9, num_samples)
        ])
        
        placement_probs = [0.15, 0.2, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05]
        y_synthetic = np.random.choice(range(1, 9), size=num_samples, p=placement_probs)
        
        return X_synthetic, y_synthetic
    
    def predict_custom_teams(self, team_ids: List[str], tournament_name: str = "Custom Tournament") -> Dict:
        """Make predictions for custom list of teams"""
        print(f"🎯 Iniciando predições para torneio personalizado: {tournament_name}")
        
        if self.use_historical_data:
            self.train_with_historical_data()
        else:
            self._train_with_synthetic_data()
        
        teams = self.data_collector.get_teams_by_ids(team_ids)
        
        features = self.data_processor.extract_features(teams)
        df, X_processed = self.feature_engineer.prepare_features(features, fit_scaler=False)
        
        self.predictor = PlacementPredictor(self.model_trainer)
        predictions = self.predictor.predict_placements(teams)
        
        return {
            'tournament': tournament_name,
            'predictions': predictions,
            'teams_analyzed': len(teams),
            'features_used': X_processed.shape[1]
        }
    
    def add_custom_tournament(self, tournament_key: str, tournament_name: str, teams: List[Dict[str, str]]):
        """Add a custom tournament configuration"""
        tournament_config = TournamentConfig(
            name=tournament_name,
            teams=teams
        )
        self.tournament_manager.add_tournament(tournament_key, tournament_config)
        print(f"✅ Torneio personalizado adicionado: {tournament_name}")