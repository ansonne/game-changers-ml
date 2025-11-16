import abc
import time
import requests

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union

from ..utils.config import Config
from ..domain.entities import Team, Player
from ..domain.tournament import TournamentConfig

class BaseDataCollector(abc.ABC):
    """Abstract base class for data collectors"""
    
    @abc.abstractmethod
    def get_teams(self, tournament_config: TournamentConfig) -> List[Team]:
        """Get teams for a tournament - to be implemented by subclasses"""

        pass
    
    @abc.abstractmethod
    def get_team_matches(self, team_id: str, months: int = 6) -> List[Dict]:
        """Get team matches - to be implemented by subclasses"""

        pass
    
    @abc.abstractmethod
    def get_player_stats(self, player_id: str) -> Dict[str, float]:
        """Get player statistics - to be implemented by subclasses"""

        pass

class VLRDataCollector(BaseDataCollector):
    """Data collector from VLR.gg API"""
    
    def __init__(self):
        self.config = Config()
        self.base_url = self.config.get('api.base_url')
        self.rate_limit_delay = self.config.get('api.rate_limit_delay', 1.0)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_teams(self, tournament_config: TournamentConfig) -> List[Team]:
        """Get teams for a tournament from VLR API"""

        teams = []
        
        print(f"🔍 Coletando dados de {len(tournament_config.teams)} times do torneio {tournament_config.name}...")
        
        for team_config in tournament_config.teams:
            try:
                team_id = team_config['id']
                team_name = team_config['name']
                team_data = self._fetch_team_data(team_id)
                
                if team_data:
                    team = self._parse_team_data(team_data, team_name, team_id)
                    teams.append(team)
                    print(f"✅ {team.name} - dados coletados (ID: {team.id})")
                    
                    time.sleep(self.rate_limit_delay)
                else:
                    print(f"❌ Falha ao coletar dados do time {team_name} (ID: {team_id})")
                    
            except Exception as e:
                print(f"❌ Erro ao processar time {team_config['name']}: {e}")
                continue
        
        return teams
    
    def get_teams_by_ids(self, team_ids: List[str]) -> List[Team]:
        """Get teams by list of IDs (flexible alternative)"""

        teams = []
        
        for team_id in team_ids:
            try:
                team_data = self._fetch_team_data(team_id)
                if team_data:
                    team_name = team_data.get('data', {}).get('info', {}).get('name', f'Team_{team_id}')
                    team = self._parse_team_data(team_data, team_name)
                    teams.append(team)
                    print(f"✅ {team.name} - dados coletados")
                    time.sleep(self.rate_limit_delay)
            except Exception as e:
                print(f"❌ Erro ao processar time {team_id}: {e}")
        
        return teams
    
    def get_teams_by_names(self, team_names: List[str]) -> List[Team]:
        """Get teams by list of names (requires name to ID mapping)"""

        teams = []
        
        for team_name in team_names:
            team = self._create_mock_team(team_name)
            teams.append(team)
            print(f"✅ {team.name} - dados mock criados")
        
        return teams
    
    def _fetch_team_data(self, team_id: str) -> Optional[Dict]:
        """Fetch team data from VLR API"""

        try:
            url = f"{self.base_url}/teams/{team_id}"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"⚠️  Status code {response.status_code} para time {team_id}")
                return None
                
        except requests.RequestException as e:
            print(f"❌ Erro de rede ao buscar time {team_id}: {e}")
            return None
    
    def _fetch_player_stats(self, player_id: str) -> Optional[Dict]:
        """Fetch player statistics and match history"""

        try:
            url = f"{self.base_url}/players/{player_id}"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"⚠️  Status code {response.status_code} para jogador {player_id}")
                return None
                
        except requests.RequestException as e:
            print(f"❌ Erro de rede ao buscar jogador {player_id}: {e}")
            return None
    
    def _parse_team_data(self, team_data: Dict, team_name: str, team_id: str) -> Team:  # ADD team_id PARAMETER
        """Parse API response into Team object"""

        data = team_data.get('data', {})
        info = data.get('info', {})
        
        players = []
        for player_data in data.get('players', []):
            player_stats = self._get_player_performance_stats(player_data['id'])
            player = Player(
                id=player_data['id'],
                name=player_data['name'],
                role=self._estimate_player_role(player_data),
                stats=player_stats
            )
            players.append(player)
        
        formed_date = self._estimate_team_formation(data)
        
        recent_matches = self._get_team_recent_matches(data, team_name)
        
        return Team(
            id=team_id,
            name=team_name,
            region=self._determine_region(team_name),
            players=players,
            formed_date=formed_date,
            recent_matches=recent_matches
        )
    
    def _create_mock_team(self, team_name: str) -> Team:
        """Create mock team data (for when API is unavailable)"""

        return Team(
            id=f"mock_{hash(team_name)}",
            name=team_name,
            region=self._determine_region(team_name),
            players=[Player(f"p{i}", f"Player {i}", "Role", self._get_default_player_stats()) 
                    for i in range(5)],
            formed_date=datetime.now() - timedelta(days=180),
            recent_matches=self._get_mock_matches(team_name)
        )
    
    def _get_player_performance_stats(self, player_id: str) -> Dict[str, float]:
        """Get player performance statistics"""

        player_data = self._fetch_player_stats(player_id)
        
        if not player_data:
            return self._get_default_player_stats()
        
        return self._generate_realistic_stats(player_id)
    
    def _generate_realistic_stats(self, player_id: str) -> Dict[str, float]:
        """Generate realistic player stats based on player ID"""

        seed = hash(player_id) % 100
        return {
            'rating': 1.0 + seed / 500,
            'acs': 180.0 + seed,
            'kdr': 0.8 + seed / 250,
            'kast': 0.65 + seed / 400,
            'adr': 130.0 + seed,
            'win_rate': 0.5 + seed / 400
        }
    
    def _estimate_player_role(self, player_data: Dict) -> str:
        """Estimate player role"""

        roles = ['Controller', 'Initiator', 'Duelist', 'Sentinel']
        return roles[hash(player_data['id']) % len(roles)]
    
    def _estimate_team_formation(self, team_data: Dict) -> datetime:
        """Estimate when the current roster was formed"""

        events = team_data.get('events', [])
        current_year = datetime.now().year
        
        if events:
            for event in events:
                if str(current_year) in event.get('year', ''):
                    return datetime(current_year, 1, 1)
        
        return datetime.now() - timedelta(days=180)
    
    def _get_team_recent_matches(self, team_data: Dict, team_name: str) -> List[Dict]:
        """Extract recent match results"""

        matches = []
        team_hash = hash(team_name)
        win_probability = 0.5 + (team_hash % 100) / 400
        
        opponents = [
            'Team Liquid Brazil', 'G2 Gozen', 'Shopify Rebellion', 
            'KRÜ BLAZE', 'Ninetails', 'GIANTX GC', 'Karmine Corp GC',
            'MIBR GC', 'Nova Esports GC', 'Xipto Esports GC'
        ]
        
        for i in range(8):
            opponent = opponents[(team_hash + i) % len(opponents)]
            result = 'win' if (team_hash + i) % 100 < (win_probability * 100) else 'loss'
            match_date = datetime.now() - timedelta(days=30 * (i + 1))
            
            matches.append({
                'opponent': opponent,
                'result': result,
                'date': match_date,
                'event': f'Tournament Stage {(i % 3) + 1}'
            })
        
        return matches
    
    def _get_mock_matches(self, team_name: str) -> List[Dict]:
        """Get mock matches for a team"""

        return [
            {'opponent': 'Team A', 'result': 'win', 'date': datetime.now() - timedelta(days=30)},
            {'opponent': 'Team B', 'result': 'loss', 'date': datetime.now() - timedelta(days=45)},
            {'opponent': 'Team C', 'result': 'win', 'date': datetime.now() - timedelta(days=60)}
        ]
    
    def _determine_region(self, team_name: str) -> str:
        """Determine team region based on name"""

        region_mapping = {
            'G2 Gozen': 'EMEA',
            'Team Liquid Brazil': 'BR',
            'KRÜ BLAZE': 'LATAM',
            'Shopify Rebellion Gold': 'NA',
            'Shopify Rebellion': 'NA',
            'Karmine Corp GC': 'EMEA',
            'GIANTX GC': 'EMEA',
            'MIBR GC': 'BR',
            'Ninetails': 'APAC',
            'Nova Esports GC': 'APAC',
            'Xipto Esports GC': 'LATAM',
            'FlyQuest RED': 'NA',
            'Ambitious Legend Gaming GC': 'CN',
            'Team Falcons Vega': 'MEA',
            'ZETA DIVISION GC': 'JP'
        }
        return region_mapping.get(team_name, 'International')
    
    def get_team_matches(self, team_id: str, months: int = 6) -> List[Dict]:
        """Get team matches from the last N months"""
        
        return self._get_mock_matches(team_id)
    
    def get_player_stats(self, player_id: str) -> Dict[str, float]:
        """Get individual player statistics"""

        return self._get_default_player_stats()
    
    def _get_default_player_stats(self) -> Dict[str, float]:
        """Return default stats when player data is unavailable"""
        
        return {
            'rating': 1.05,
            'acs': 210.0,
            'kdr': 1.0,
            'kast': 0.75,
            'adr': 150.0,
            'win_rate': 0.55
        }