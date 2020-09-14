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

from typing import Dict, Tuple
from betbot.neural.data.openligadb.Team import Team
from betbot.neural.data.openligadb.TableEntry import TableEntry


class HistoryRecord:
    """
    Class that keeps track of a team's recent history
    Only keeps track of stats within a league!
    """

    def __init__(
            self,
            team: Team
    ):
        """
        Initializes the HistoryRecord object
        """
        self.team = team
        self._history: Dict[int, Dict[int, Tuple[int, int, int]]] \
            = {}

    def add_table_entry(self, entry: TableEntry):
        """
        Adds points and goals history data based on a table entry
        :param entry: The table entry
        :return: None
        """
        if entry.season not in self._history:
            self._history[entry.season] = {}

        self._history[entry.season][entry.matchday] = (
            entry.points,
            entry.goals_for,
            entry.goals_against
        )

    def get_stats(self, season: int, matchday: int, previous: bool) \
            -> Tuple[int, int, int]:
        """
        Retrieves stats for a season and matchday for this team's history
        If no history is found, 0 values will be returned
        :param season: The season
        :param matchday: The matchday
        :param previous: Whether or not to use the previous matchday
                         This is useful if a match is already finished
        :return: The points, goals for and goals against
        """
        if previous:
            if matchday == 1:
                if (season - 1) not in self._history:
                    return 0, 0, 0
                else:
                    season -= 1
                    matchday = max(self._history[season].keys())
            else:
                matchday -= 1
        stats = self._history.get(season, {}).get(matchday)
        return stats if stats is not None else (0, 0, 0)

    def get_stats_interval(
            self,
            season: int,
            matchday: int,
            interval: int,
            finished: bool
    ) -> Tuple[int, int, int]:
        """
        Retrieves stats for an interval
        :param season: The season
        :param matchday: The 'anchor' matchday
        :param interval: The size of the interval
        :param finished: If the match was finished or not
        :return: The points and goals during the interval
        """
        # To actually include the first entry in the interval
        interval += 1

        points, goals_for, goals_against = \
            self.get_stats(season, matchday, finished)

        if matchday - interval <= 0 and (season - 1) in self._history:
            max_matchday = max(self._history[season - 1].keys())
            start_matchday = max_matchday - (interval - matchday)
            start_points, start_goals_for, start_goals_against = \
                self.get_stats(season - 1, start_matchday, False)
            end_points, end_goals_for, end_goals_against = \
                self.get_stats(season - 1, max_matchday, False)
            points += (end_points - start_points)
            goals_for += (end_goals_for - start_goals_for)
            goals_against += (end_goals_against - start_goals_against)
        elif matchday == 1:
            return 0, 0, 0
        elif matchday - interval <= 0:
            pass  # In the first season, simply use current stats
        else:
            start_points, start_goals_for, start_goals_against = \
                self.get_stats(season, matchday - interval, False)
            points -= start_points
            goals_for -= start_goals_for
            goals_against -= start_goals_against

        return points, goals_for, goals_against

    def get_max_matchday(self, season: int):
        return max(self._history[season].keys())
