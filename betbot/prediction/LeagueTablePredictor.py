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

from typing import List
from betbot.api.Bet import Bet
from betbot.api.Match import Match
from betbot.prediction.Predictor import Predictor


class LeagueTablePredictor(Predictor):
    """
    Class that always 2:1 for the team higher up in the league table
    """

    @classmethod
    def name(cls) -> str:
        """
        :return: The name of the predictor
        """
        return "league-table"

    def predict(self, matches: List[Match]) -> List[Bet]:
        """
        Performs the prediction
        :param matches: The matches to predict
        :return: The predictions as Bet objects
        """
        league_table = self.api.get_league_table(self.league, self.season)
        bets = []
        for match in matches:
            home_team_index = league_table.index(match.home_team)
            away_team_index = league_table.index(match.away_team)
            if home_team_index < away_team_index:
                home_goals, away_goals = 2, 1
            else:
                home_goals, away_goals = 1, 2
            bets.append(Bet(match, home_goals, away_goals))
        return bets
