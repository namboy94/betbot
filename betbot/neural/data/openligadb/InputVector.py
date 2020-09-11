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

from typing import List, Optional
from betbot.neural.data.openligadb.TableEntry import TableEntry


class InputVector:
    """
    Class that models an input vector to be used in training for a neural
    network that predicts scores.
    """

    def __init__(
            self,
            home_table_entry: TableEntry,
            away_table_entry: TableEntry,
            home_table_entry_5_matchdays_ago: TableEntry,
            away_table_entry_5_matchdays_ago: TableEntry,
            home_last_season_entry: Optional[TableEntry],
            away_last_season_entry: Optional[TableEntry]
    ):
        """
        Initializes the InputVector
        :param home_table_entry: The current table entry of the home team
        :param away_table_entry: The current table entry of the away team
        :param home_table_entry_5_matchdays_ago: The table entry of the home
                                                 team 5 matchdays ago
        :param away_table_entry_5_matchdays_ago: The table entry of the away
                                                 team 5 matchdays ago
        :param home_last_season_entry: The table entry on the last matchday
                                       of last season for the home team
        :param away_last_season_entry: The table entry on the last matchday
                                       of last season for the away team
        """
        self.home_total_points = home_table_entry.points
        self.home_total_goals_for = home_table_entry.goals_for
        self.home_total_goals_against = home_table_entry.goals_against
        self.away_total_points = away_table_entry.points
        self.away_total_goals_for = away_table_entry.goals_for
        self.away_total_goals_against = away_table_entry.goals_against

        if home_table_entry.matchday < 6 and \
                home_last_season_entry is not None and \
                away_last_season_entry is not None:
            self.home_points_last_5_matches = \
                home_table_entry.points + \
                home_last_season_entry.points - \
                home_table_entry_5_matchdays_ago.points
            self.home_goals_for_last_5_matches = \
                home_table_entry.goals_for + \
                home_last_season_entry.goals_for - \
                home_table_entry_5_matchdays_ago.goals_for
            self.home_goals_against_last_5_matches = \
                home_table_entry.goals_against + \
                home_last_season_entry.goals_against - \
                home_table_entry_5_matchdays_ago.goals_against
            self.away_points_last_5_matches = \
                away_table_entry.points + \
                away_last_season_entry.points - \
                away_table_entry_5_matchdays_ago.points
            self.away_goals_for_last_5_matches = \
                away_table_entry.goals_for + \
                away_last_season_entry.goals_for - \
                away_table_entry_5_matchdays_ago.goals_for
            self.away_goals_against_last_5_matches = \
                away_table_entry.goals_against + \
                away_last_season_entry.goals_against - \
                away_table_entry_5_matchdays_ago.goals_against
        else:
            self.home_points_last_5_matches = \
                home_table_entry.points - \
                home_table_entry_5_matchdays_ago.points
            self.home_goals_for_last_5_matches = \
                home_table_entry.goals_for - \
                home_table_entry_5_matchdays_ago.goals_for
            self.home_goals_against_last_5_matches = \
                home_table_entry.goals_against - \
                home_table_entry_5_matchdays_ago.goals_against
            self.away_points_last_5_matches = \
                away_table_entry.points - \
                away_table_entry_5_matchdays_ago.points
            self.away_goals_for_last_5_matches = \
                away_table_entry.goals_for - \
                away_table_entry_5_matchdays_ago.goals_for
            self.away_goals_against_last_5_matches = \
                away_table_entry.goals_against - \
                away_table_entry_5_matchdays_ago.goals_against

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
