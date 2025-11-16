import os
import sys
import pytest

from unittest.mock import Mock, patch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.collectors import VLRDataCollector

class TestAPIComponents:
    
    @patch('src.data.collectors.requests.Session')
    def test_api_team_data_fetch(self, mock_session):
        """Test API team data fetching with mock"""

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'status': 'OK',
            'data': {
                'info': {
                    'name': 'Test Team',
                    'tag': 'TEST',
                    'logo': 'test.png'
                },
                'players': [
                    {
                        'id': '1',
                        'name': 'Test Player',
                        'country': 'test'
                    }
                ],
                'events': []
            }
        }
        
        mock_session.return_value.get.return_value = mock_response
        
        collector = VLRDataCollector()
        team_data = collector._fetch_team_data('123')
        
        assert team_data is not None
        assert team_data['status'] == 'OK'
        assert team_data['data']['info']['name'] == 'Test Team'
    
    def test_region_mapping(self):
        """Test region mapping logic"""

        collector = VLRDataCollector()
        
        test_cases = [
            ('G2 Gozen', 'EMEA'),
            ('Team Liquid Brazil', 'BR'),
            ('Shopify Rebellion Gold', 'NA'),
            ('Unknown Team', 'International')
        ]
        
        for team_name, expected_region in test_cases:
            region = collector._determine_region(team_name)
            assert region == expected_region
    
    def test_player_stats_default(self):
        """Test default player stats fallback"""
        
        collector = VLRDataCollector()
        default_stats = collector._get_default_player_stats()
        
        expected_keys = ['rating', 'acs', 'kdr', 'kast', 'adr', 'win_rate']
        for key in expected_keys:
            assert key in default_stats
            assert isinstance(default_stats[key], float)