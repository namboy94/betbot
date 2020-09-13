"""LICENSE
Copyright 2020 Hermann Krumrey <hermann@krumreyh.com>

This file is part of betbot.

betbot is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

betbot is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with betbot.  If not, see <http://www.gnu.org/licenses/>.
LICENSE"""

from typing import Dict, Any
from betbot.neural.data.openligadb.Team import Team


class Match:
    """
    Class that models an OpenligaDB match
    """

    def __init__(
            self,
            league: str,
            season: int,
            match_json: Dict[str, Any],
            teams: Dict[int, Team]
    ):
        """
        Initializes the match object
        :param league: The league of the match
        :param season: The season of the match
        :param match_json: The OpenligaDB JSON data for the match
        :param teams: The teams mapped to their IDs
        """
        finished = match_json["MatchIsFinished"]
        if finished:
            results = match_json["MatchResults"]
            if results[0]["ResultName"] == "Endergebnis":
                result = results[0]
            else:
                result = results[1]
            home_score = result["PointsTeam1"]
            away_score = result["PointsTeam2"]
        else:
            home_score, away_score = None, None

        self.league = league
        self.season = season
        self.matchday = match_json["Group"]["GroupOrderID"]
        self.home_team = teams[match_json["Team1"]["TeamId"]]
        self.away_team = teams[match_json["Team2"]["TeamId"]]
        self.home_score = home_score
        self.away_score = away_score
        self.finished = finished
