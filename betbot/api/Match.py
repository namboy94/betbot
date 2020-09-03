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
from dataclasses import dataclass


@dataclass
class Match:
    """
    Class that encapsulates information about a Match
    """

    id: int
    """
    The ID of the match
    """

    matchday: int
    """
    The matchday of the match
    """

    home_team: str
    """
    The name of the home team
    """

    away_team: str
    """
    The name of the away team
    """

    @classmethod
    def from_json(cls, json_data: Dict[str, Any]):
        """
        Generates a Match object from JSON data
        :param json_data: The JSON data
        :return: The generated Match
        """
        return cls(
            json_data["id"],
            json_data["matchday"],
            json_data["home_team"]["abbreviation"],
            json_data["away_team"]["abbreviation"]
        )
