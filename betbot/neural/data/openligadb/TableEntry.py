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

from betbot.neural.data.openligadb.Team import Team


class TableEntry:
    """
    Class that models a league table entry for one club
    """

    def __init__(
            self,
            matchday: int,
            team: Team,
            points: int,
            goals_for: int,
            goals_against: int
    ):
        """
        Initializes the TableEntry object
        :param matchday: The matchday of the entry
        :param team: The team for which this is an entry
        :param points: The points of the team
        :param goals_for: The goals the team scored
        :param goals_against: The goals the team conceded
        """
        self.matchday = matchday
        self.team = team
        self.points = points
        self.goals_for = goals_for
        self.goals_against = goals_against

    def merge(self, entry: "TableEntry"):
        """
        Merges another table entry's points and goals data into this one
        :param entry: The entry to merge into this one
        :return: Nonew
        """
        self.points += entry.points
        self.goals_for += entry.goals_for
        self.goals_against += entry.goals_against
