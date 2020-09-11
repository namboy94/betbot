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


class OutputVector:
    """
    Class that models the output vector for a neural network training dataset
    """

    def __init__(self, home_score: int, away_score: int):
        """
        Initializes the OutputVector object
        :param home_score: The home score
        :param away_score: The away score
        """
        self.home_score = home_score
        self.away_score = away_score

    @property
    def vector(self) -> List[int]:
        return [self.home_score, self.away_score]
