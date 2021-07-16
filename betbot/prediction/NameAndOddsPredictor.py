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

from typing import Tuple
from betbot.prediction.SKLearnPredictor import SKLearnPredictor


class NameAndOddsPredictor(SKLearnPredictor):
    """
    scikit-learn powered predictor that uses the name and betting odds
    """

    @classmethod
    def name(cls) -> str:
        """
        :return: The name of the predictor
        """
        return "name-and-odds"

    # noinspection PyMethodMayBeStatic
    def interpret_results(self, home_result: float, away_result: float) -> \
            Tuple[int, int]:
        """
        Interprets the raw results
        :param home_result: The home goals result
        :param away_result: The away goals result
        :return: The home goals, the away goals
        """
        min_score = min([home_result, away_result])
        normer = round(min_score)
        home_score = round(home_result - min_score + normer)
        away_score = round(away_result - min_score + normer)
        if home_score == away_score:
            if min_score == home_result:
                away_score += 1
            else:
                home_score += 1

        return home_score, away_score
