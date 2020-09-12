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
        self.home_total_points, self.home_total_goals_for, \
            self.home_total_goals_against = \
            home_team_history.get_stats(season, matchday, finished)
        self.away_total_points, self.away_total_goals_for, \
            self.away_total_goals_against = \
            away_team_history.get_stats(season, matchday, finished)

        self.home_points_last_5_matches, self.home_goals_for_last_5_matches, \
            self.home_goals_against_last_5_matches = \
            home_team_history.get_stats_interval(
                season, matchday, 5, finished
            )
        self.away_points_last_5_matches, self.away_goals_for_last_5_matches,\
            self.away_goals_against_last_5_matches = \
            away_team_history.get_stats_interval(
                season, matchday, 5, finished
            )

    @property
    def vector(self) -> List[int]:
        """
        :return: The input vector as a list of integers
        """
        return [
            self.home_total_points,
            self.home_total_goals_for,
            self.home_total_goals_against,
            self.home_points_last_5_matches,
            self.home_goals_for_last_5_matches,
            self.home_goals_against_last_5_matches,
            self.away_total_points,
            self.away_total_goals_for,
            self.away_total_goals_against,
            self.away_points_last_5_matches,
            self.away_goals_for_last_5_matches,
            self.away_goals_against_last_5_matches
        ]
