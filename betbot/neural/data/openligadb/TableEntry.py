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
from betbot.neural.data.openligadb.Match import Match


class TableEntry:
    """
    Class that models a league table entry for one club
    """

    def __init__(
            self,
            league: str,
            season: int,
            matchday: int,
            team: Team,
            points: int,
            goals_for: int,
            goals_against: int
    ):
        """
        Initializes the TableEntry object
        :param league: The league of the entry
        :param season: The season of the entry
        :param matchday: The matchday of the entry
        :param team: The team for which this is an entry
        :param points: The points of the team
        :param goals_for: The goals the team scored
        :param goals_against: The goals the team conceded
        """
        self.league = league
        self.season = season
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

    @classmethod
    def from_match(cls, match: Match, team: Team) -> "TableEntry":
        """
        Generates a table entry based on a match
        :param match: The match
        :param team: The team for which to generate the entry
        :return: The table entry
        """
        is_home_team = team.id == match.home_team.id

        if is_home_team:
            score_for, score_against = match.home_score, match.away_score
        else:
            score_for, score_against = match.away_score, match.home_score

        if not match.finished:
            points = 0
            score_for = 0
            score_against = 0
        elif score_for > score_against:
            points = 3
        elif score_for == score_against:
            points = 1
        else:
            points = 0

        return cls(
            match.league,
            match.season,
            match.matchday,
            team,
            points,
            score_for,
            score_against
        )
