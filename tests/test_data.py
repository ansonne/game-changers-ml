import os
import sys
import pytest

from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.domain.entities import Team, Player
from src.data.processors import DataProcessor
from src.data.collectors import VLRDataCollector

class TestDataComponents:
    def test_data_collector_initialization(self):
        collector = VLRDataCollector()
        assert collector is not None
        assert collector.base_url is not None
    
    def test_qualified_teams_loading(self):
        collector = VLRDataCollector()
        teams = collector.get_qualified_teams()
        
        assert len(teams) > 0
        assert all(isinstance(team, Team) for team in teams)
    
    def test_data_processor_calculation(self):
        processor = DataProcessor()
        
        test_team = Team(
            id='test',
            name='Test Team',
            region='TEST',
            players=[],
            formed_date=datetime(2024, 1, 1),
            recent_matches=[]
        )
        
        stability = processor.calculate_roster_stability(test_team)
        assert 0 <= stability <= 1
        
        individual_perf = processor.calculate_individual_performance(test_team)
        assert individual_perf['avg_rating'] == 0.0
    
    def test_feature_extraction(self):
        processor = DataProcessor()
        collector = VLRDataCollector()
        
        teams = collector.get_qualified_teams()
        features = processor.extract_features(teams)
        
        assert len(features) == len(teams)
        assert all(hasattr(f, 'roster_stability') for f in features)
        assert all(hasattr(f, 'avg_player_rating') for f in features)