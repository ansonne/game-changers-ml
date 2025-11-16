from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, Dict

@dataclass
class Player:
    id: str
    name: str
    role: str
    stats: Dict[str, float]

@dataclass
class Team:
    id: str
    name: str
    region: str
    players: List[Player]
    formed_date: datetime
    recent_matches: List[Dict]
    events: List[Dict] = None

@dataclass
class Match:
    id: str
    team1_id: str
    team2_id: str
    team1_score: int
    team2_score: int
    date: datetime
    event: str

@dataclass
class TrainingFeatures:
    team_id: str
    roster_stability: float
    avg_player_rating: float
    avg_acs: float
    avg_kdr: float
    avg_kast: float
    win_rate: float
    recent_form: float
    strength_of_schedule: float
    tournament_results: float
    target: Optional[int] = None