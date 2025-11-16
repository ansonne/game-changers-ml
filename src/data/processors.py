from typing import List, Dict
from datetime import datetime

from ..utils.config import Config
from ..domain.entities import Team, TrainingFeatures

class DataProcessor:
    """Process raw data into training features"""
    
    def __init__(self):
        self.config = Config()
    
    def calculate_roster_stability(self, team: Team) -> float:
        """Calculate roster stability score (0-1) based on formation date"""

        if not team.formed_date:
            return 0.5
        
        days_together = (datetime.now() - team.formed_date).days
        max_days = 365
        
        stability = min(days_together / max_days, 1.0)
        
        roster_completeness = len(team.players) / 5
        stability *= roster_completeness
        
        return stability
    
    def calculate_individual_performance(self, team: Team) -> Dict[str, float]:
        """Calculate individual performance metrics"""

        if not team.players:
            return {
                'avg_rating': 0.0,
                'avg_acs': 0.0,
                'avg_kdr': 0.0,
                'avg_kast': 0.0
            }
        
        total_rating = sum(player.stats.get('rating', 0) for player in team.players)
        total_acs = sum(player.stats.get('acs', 0) for player in team.players)
        total_kdr = sum(player.stats.get('kdr', 0) for player in team.players)
        total_kast = sum(player.stats.get('kast', 0) for player in team.players)
        
        num_players = len(team.players)
        
        return {
            'avg_rating': total_rating / num_players,
            'avg_acs': total_acs / num_players,
            'avg_kdr': total_kdr / num_players,
            'avg_kast': total_kast / num_players
        }
    
    def calculate_team_performance(self, team: Team) -> Dict[str, float]:
        """Calculate team performance metrics from recent matches"""

        if not team.recent_matches:
            return {
                'win_rate': 0.5,
                'recent_form': 0.5,
                'strength_of_schedule': 0.5
            }
        
        wins = sum(1 for match in team.recent_matches if match['result'] == 'win')
        total_matches = len(team.recent_matches)
        win_rate = wins / total_matches if total_matches > 0 else 0.5
        
        recent_form = 0.0
        form_weight = 0.0
        
        for i, match in enumerate(team.recent_matches[:6]):
            weight = 1.0 / (i + 1)
            form_weight += weight
            recent_form += weight * (1.0 if match['result'] == 'win' else 0.0)
        
        recent_form = recent_form / form_weight if form_weight > 0 else win_rate
        
        strong_opponents = ['G2 Gozen', 'Team Liquid Brazil', 'Shopify Rebellion Gold']
        strong_opponent_matches = sum(1 for match in team.recent_matches 
                                    if match['opponent'] in strong_opponents)
        
        strength_of_schedule = 0.5 + (strong_opponent_matches / total_matches * 0.3 
                                    if total_matches > 0 else 0)
        
        return {
            'win_rate': win_rate,
            'recent_form': recent_form,
            'strength_of_schedule': min(strength_of_schedule, 1.0)
        }
    
    def calculate_tournament_performance(self, team: Team) -> float:
        """Calculate performance in recent tournaments"""

        if not team.recent_matches:
            return 0.5
        
        gc_matches = [m for m in team.recent_matches if 'Game Changers' in m.get('event', '')]
        if not gc_matches:
            return 0.5
        
        gc_wins = sum(1 for m in gc_matches if m['result'] == 'win')
        gc_win_rate = gc_wins / len(gc_matches)
        
        return gc_win_rate
    
    def extract_features(self, teams: List[Team]) -> List[TrainingFeatures]:
        """Extract all features for model training"""

        features = []
        
        for team in teams:
            roster_stability = self.calculate_roster_stability(team)
            individual_perf = self.calculate_individual_performance(team)
            team_perf = self.calculate_team_performance(team)
            tournament_perf = self.calculate_tournament_performance(team)
            
            feature_set = TrainingFeatures(
                team_id=team.id,
                roster_stability=roster_stability,
                avg_player_rating=individual_perf['avg_rating'],
                avg_acs=individual_perf['avg_acs'],
                avg_kdr=individual_perf['avg_kdr'],
                avg_kast=individual_perf['avg_kast'],
                win_rate=team_perf['win_rate'],
                recent_form=team_perf['recent_form'],
                strength_of_schedule=team_perf['strength_of_schedule'],
                tournament_results=tournament_perf
            )
            features.append(feature_set)
        
        return features