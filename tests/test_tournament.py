import os
import sys
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.domain.tournament import TournamentConfig, TournamentManager

class TestTournamentComponents:
    def test_tournament_config(self):
        """Test tournament configuration"""

        config = TournamentConfig(
            name="Test Tournament",
            teams=[{"id": "1", "name": "Team A"}, {"id": "2", "name": "Team B"}]
        )
        
        assert config.name == "Test Tournament"
        assert len(config.teams) == 2
        assert config.get_team_ids() == ["1", "2"]
        assert config.get_team_names() == ["Team A", "Team B"]
    
    def test_tournament_manager(self):
        """Test tournament manager"""

        config_data = {
            'tournaments': {
                'test_tournament': {
                    'name': 'Test Tournament',
                    'teams': [{'id': '1', 'name': 'Team A'}]
                }
            }
        }
        
        manager = TournamentManager(config_data)
        
        assert 'test_tournament' in manager.list_tournaments()
        tournament = manager.get_tournament('test_tournament')
        assert tournament.name == 'Test Tournament'
        
        new_tournament = TournamentConfig("New Tournament", [{'id': '3', 'name': 'Team C'}])
        manager.add_tournament('new_tournament', new_tournament)
        assert 'new_tournament' in manager.list_tournaments()