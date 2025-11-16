#!/usr/bin/env python3
"""
Main execution script for Game Changers prediction system - Simplified Version
"""

import os
import sys
from typing import Dict, List

from src.utils.config import Config
from src.services.prediction_service import PredictionService

def display_predictions(result: Dict):
    """Display prediction results in a formatted way"""

    print(f"\n🏆 PREDIÇÕES - {result['tournament']}")
    print("=" * 70)
    
    for i, prediction in enumerate(result['predictions'], 1):
        print(f"\n{i}º Lugar: {prediction['team']}")
        print(f"   📈 Colocação Prevista: {prediction['predicted_placement']}º")
        print(f"   🎯 Confiança: {prediction['confidence_score']:.1%}")
        if 'actual_placement' in prediction:
            print(f"   🏆 Colocação Real: {prediction['actual_placement']}º")
            diff = abs(prediction['predicted_placement'] - prediction['actual_placement'])
            status = "✅ EXATO" if diff == 0 else "⚠️  PRÓXIMO" if diff == 1 else "❌ LONGE"
            print(f"   📊 Diferença: {diff} posição(ões) - {status}")

def display_validation_metrics(metrics: Dict):
    """Display validation metrics"""

    print(f"\n📈 MÉTRICAS DE VALIDAÇÃO:")
    print(f"   📊 MAE (Mean Absolute Error): {metrics['mae']:.2f}")
    print(f"   📈 RMSE (Root Mean Squared Error): {metrics['rmse']:.2f}")
    print(f"   🎯 Acurácia (±1 posição): {metrics['accuracy']:.1%}")
    print(f"   📋 Amostras: {metrics['samples']}")

def main():
    print("🚀 Sistema de Predição de Torneios de Valorant")
    print("=" * 50)
    
    try:
        config = Config()
        prediction_service = PredictionService(config._config)
        
        available_tournaments = prediction_service.list_available_tournaments()
        print("📋 Torneios Disponíveis:")
        for i, tournament_key in enumerate(available_tournaments, 1):
            tournament = prediction_service.tournament_manager.get_tournament(tournament_key)
            type_flag = "📚" if tournament.is_historical else "🔮"
            print(f"   {i}. {type_flag} {tournament.name}")
        
        print("\n1. 📚 TREINANDO COM DADOS HISTÓRICOS...")
        trained, performance = prediction_service.train_with_historical_data()
        
        if trained:
            print(f"✅ Modelo treinado com sucesso!")
            print(f"   🎯 Melhor modelo: {performance['best_model']}")
            print(f"   📊 MAE: {performance['mae']:.2f}")
            print(f"   📈 RMSE: {performance['rmse']:.2f}")
            print(f"   📋 Amostras de treino: {performance['samples']}")
        else:
            print("⚠️  Usando modelo com dados sintéticos")
        
        print("\n2. 📊 VALIDAÇÃO COM GAME CHANGERS 2024 BERLIN")
        try:
            historical_result = prediction_service.validate_historical_predictions('game_changers_2024_berlin')
            display_predictions(historical_result)
            
            if 'validation_metrics' in historical_result:
                display_validation_metrics(historical_result['validation_metrics'])
        except Exception as e:
            print(f"❌ Erro na validação histórica: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n3. 🔮 PREDIÇÃO PARA GAME CHANGERS 2025 SEOUL")
        try:
            future_result = prediction_service.predict_tournament('game_changers_2025_seoul')
            display_predictions(future_result)
        except Exception as e:
            print(f"❌ Erro na predição futura: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n4. 🤖 RESUMO DO MODELO")
        if trained:
            print("✅ Modelo validado com dados históricos reais")
            print("💡 O sistema está pronto para predições futuras!")
        else:
            print("⚠️  Modelo usando dados sintéticos - considere adicionar mais dados históricos")
            
    except Exception as e:
        print(f"❌ Erro inicializando o sistema: {e}")
        import traceback
        traceback.print_exc()

    print("\n🎉 Sistema finalizado!")

if __name__ == "__main__":
    main()