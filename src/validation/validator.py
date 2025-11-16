import numpy as np
import pandas as pd

from typing import List, Dict, Tuple, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error


class ModelValidator:
    """Validate model predictions against actual results"""
    
    def __init__(self):
        self.metrics = {}
    
    def validate_predictions(self, predictions: List[Dict], actual_placements: Dict[str, int]) -> Dict:
        """
        Validate predictions against actual placements
        
        Args:
            predictions: List of prediction dictionaries
            actual_placements: Dict mapping team names to actual placements
        
        Returns:
            Dictionary with validation metrics
        """

        valid_predictions = []
        valid_actuals = []
        valid_teams = []
        
        for prediction in predictions:
            team_name = prediction['team']
            if team_name in actual_placements:
                valid_predictions.append(prediction['predicted_placement'])
                valid_actuals.append(actual_placements[team_name])
                valid_teams.append(team_name)
        
        if not valid_predictions:
            return {"error": "No matching teams found for validation"}
        

        y_pred = np.array(valid_predictions)
        y_true = np.array(valid_actuals)
        
        metrics = {
            'num_teams_validated': len(valid_predictions),
            'mean_absolute_error': float(mean_absolute_error(y_true, y_pred)),
            'mean_squared_error': float(mean_squared_error(y_true, y_pred)),
            'root_mean_squared_error': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'accuracy_within_1': self._accuracy_within_range(y_true, y_pred, range_size=1),
            'accuracy_within_2': self._accuracy_within_range(y_true, y_pred, range_size=2),
            'accuracy_within_3': self._accuracy_within_range(y_true, y_pred, range_size=3),
            'top_3_accuracy': self._top_n_accuracy(y_true, y_pred, n=3),
            'top_5_accuracy': self._top_n_accuracy(y_true, y_pred, n=5),
            'perfect_predictions': self._perfect_predictions(y_true, y_pred),
        }
        
        metrics.update(self._detailed_analysis(y_true, y_pred, valid_teams, actual_placements))
        
        return metrics
    
    def _accuracy_within_range(self, y_true: np.ndarray, y_pred: np.ndarray, range_size: int) -> float:
        """Calculate accuracy within ±range_size positions"""

        within_range = np.abs(y_true - y_pred) <= range_size
        return float(np.mean(within_range))
    
    def _top_n_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, n: int) -> float:
        """Calculate accuracy for top N positions"""

        pred_top_n = y_pred <= n
        actual_top_n = y_true <= n
        correct = np.sum(pred_top_n & actual_top_n)
        total_top_n = np.sum(actual_top_n)
        return correct / total_top_n if total_top_n > 0 else 0.0
    
    def _perfect_predictions(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate percentage of perfect predictions"""

        perfect = y_true == y_pred
        return float(np.mean(perfect))
    
    def _detailed_analysis(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          teams: List[str], actual_placements: Dict[str, int]) -> Dict:
        """Perform detailed analysis of predictions"""

        analysis = {
            'prediction_details': [],
            'biggest_surprises': [],
            'biggest_disappointments': []
        }
        
        for i, team in enumerate(teams):
            pred = y_pred[i]
            actual = y_true[i]
            error = abs(pred - actual)
            
            analysis['prediction_details'].append({
                'team': team,
                'predicted': int(pred),
                'actual': int(actual),
                'error': int(error),
                'within_1': error <= 1,
                'within_2': error <= 2,
                'within_3': error <= 3
            })
        
        surprises = []
        for detail in analysis['prediction_details']:
            if detail['actual'] < detail['predicted']:
                surprise_magnitude = detail['predicted'] - detail['actual']
                surprises.append((detail['team'], surprise_magnitude))
        
        surprises.sort(key=lambda x: x[1], reverse=True)
        analysis['biggest_surprises'] = surprises[:3]
        
        disappointments = []
        for detail in analysis['prediction_details']:
            if detail['actual'] > detail['predicted']:
                disappointment_magnitude = detail['actual'] - detail['predicted']
                disappointments.append((detail['team'], disappointment_magnitude))
        
        disappointments.sort(key=lambda x: x[1], reverse=True)
        analysis['biggest_disappointments'] = disappointments[:3]
        
        return analysis
    
    def generate_validation_report(self, metrics: Dict, tournament_name: str) -> str:
        """Generate a human-readable validation report"""

        report = []
        report.append(f"📊 RELATÓRIO DE VALIDAÇÃO - {tournament_name}")
        report.append("=" * 60)
        report.append(f"Times validados: {metrics['num_teams_validated']}")
        report.append("")
        report.append("📈 Métricas Principais:")
        report.append(f"  • MAE (Mean Absolute Error): {metrics['mean_absolute_error']:.2f}")
        report.append(f"  • RMSE (Root Mean Squared Error): {metrics['root_mean_squared_error']:.2f}")
        report.append(f"  • Precisão ±1 posição: {metrics['accuracy_within_1']:.1%}")
        report.append(f"  • Precisão ±2 posições: {metrics['accuracy_within_2']:.1%}")
        report.append(f"  • Precisão ±3 posições: {metrics['accuracy_within_3']:.1%}")
        report.append(f"  • Previsões perfeitas: {metrics['perfect_predictions']:.1%}")
        report.append("")
        
        if metrics['biggest_surprises']:
            report.append("🎯 Maiores Surpresas (performaram melhor que o previsto):")
            for team, magnitude in metrics['biggest_surprises']:
                report.append(f"  • {team}: +{magnitude} posições")
        
        if metrics['biggest_disappointments']:
            report.append("")
            report.append("📉 Maiores Decepções (performaram pior que o previsto):")
            for team, magnitude in metrics['biggest_disappointments']:
                report.append(f"  • {team}: -{magnitude} posições")
        
        report.append("")
        report.append("📋 Detalhes das Predições:")
        report.append("  Time                    | Previsto | Real | Diferença")
        report.append("  ------------------------|----------|------|----------")
        for detail in metrics['prediction_details']:
            team_name = detail['team'][:20].ljust(20)
            predicted = str(detail['predicted']).center(8)
            actual = str(detail['actual']).center(4)
            error = str(detail['error']).center(9)
            report.append(f"  {team_name} | {predicted} | {actual} | {error}")
        
        return "\n".join(report)