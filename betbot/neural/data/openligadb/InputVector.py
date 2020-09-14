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

from typing import List, Union
from betbot.neural.data.openligadb.HistoryRecord import HistoryRecord


class InputVector:
    """
    Class that models an input vector to be used in training for a neural
    network that predicts scores.
    """

    def __init__(
            self,
            season: int,
            matchday: int,
            finished: bool,
            home_team_history: HistoryRecord,
            away_team_history: HistoryRecord
    ):
        """
        Initializes the InputVector
        :param season: The season of the input
        :param matchday: The match day
        :param finished: Whether or not the match has finished
        :param home_team_history: The table history of the home team
        :param away_team_history: The table history of the away team
        """
        self.season = season
        self.matchday = matchday
        self.home_team_history = home_team_history
        self.away_team_history = away_team_history
        self.current_stats = \
            list(home_team_history.get_stats(season, matchday, finished)) + \
            list(away_team_history.get_stats(season, matchday, finished))
        self.history_stats = {}

        for interval in [5]:
            self.history_stats[interval] = \
                list(home_team_history.get_stats_interval(
                    season, matchday, interval, finished
                )) + \
                list(away_team_history.get_stats_interval(
                    season, matchday, interval, finished
                ))

    @property
    def vector(self) -> List[int]:
        """
        :return: The input vector as a list of integers
        """
        vector = self.current_stats
        for key in sorted(self.history_stats.keys()):
            vector += self.history_stats[key]
        return vector

    def normalize(self, vector: List[Union[int, float]]) -> List[float]:
        """
        Normalizes the input vector to values between 0 and 1
        :param vector: The vector to normalize
        :return: The normalized vector
        """
        vector = list(vector)  # copy
        for overall_index in range(0, 6):
            vector[overall_index] = \
                min(1.0, vector[overall_index] / (self.matchday * 3))

        for history_index in range(6, len(vector)):
            vector[history_index] = \
                min(1.0, vector[history_index] / (5 * 3))
        return vector
