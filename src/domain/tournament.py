from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class TournamentConfig:
    """Configuration for a tournament"""

    name: str
    teams: List[Dict[str, any]]
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    is_historical: bool = False
    
    def get_team_ids(self) -> List[str]:
        """Get list of team IDs"""

        return [team['id'] for team in self.teams]
    
    def get_team_names(self) -> List[str]:
        """Get list of team names"""

        return [team['name'] for team in self.teams]
    
    def get_actual_placements(self) -> Dict[str, int]:
        """Get dictionary of team ID to actual placement"""

        placements = {}
        for team in self.teams:
            if 'actual_placement' in team:
                placements[team['id']] = team['actual_placement']
        return placements
    
    def find_team_by_id(self, team_id: str) -> Optional[Dict[str, any]]:
        """Find team configuration by ID"""

        return next((team for team in self.teams if team['id'] == team_id), None)
    
    def find_team_by_name(self, team_name: str) -> Optional[Dict[str, any]]:
        """Find team configuration by name"""

        return next((team for team in self.teams if team['name'] == team_name), None)

class TournamentManager:
    """Manager for multiple tournaments"""
    
    def __init__(self, config: Dict):
        self.tournaments: Dict[str, TournamentConfig] = {}
        self._load_tournaments(config)
    
    def _load_tournaments(self, config: Dict):
        """Load tournaments from configuration"""

        tournaments_config = config.get('tournaments', {})
        
        for tournament_key, tournament_data in tournaments_config.items():
            is_historical = False
            if tournament_data.get('teams'):
                first_team = tournament_data['teams'][0]
                is_historical = 'actual_placement' in first_team
            
            self.tournaments[tournament_key] = TournamentConfig(
                name=tournament_data['name'],
                teams=tournament_data['teams'],
                is_historical=is_historical
            )
    
    def get_tournament(self, tournament_key: str) -> Optional[TournamentConfig]:
        """Get tournament by key"""

        return self.tournaments.get(tournament_key)
    
    def get_historical_tournaments(self) -> List[TournamentConfig]:
        """Get all historical tournaments for training"""

        return [t for t in self.tournaments.values() if t.is_historical]
    
    def get_future_tournaments(self) -> List[TournamentConfig]:
        """Get all future tournaments for prediction"""

        return [t for t in self.tournaments.values() if not t.is_historical]
    
    def list_tournaments(self) -> List[str]:
        """List all available tournament keys"""

        return list(self.tournaments.keys())
    
    def add_tournament(self, tournament_key: str, tournament_config: TournamentConfig):
        """Add a new tournament"""

        self.tournaments[tournament_key] = tournament_config
    
    def remove_tournament(self, tournament_key: str):
        """Remove a tournament"""
        
        if tournament_key in self.tournaments:
            del self.tournaments[tournament_key]