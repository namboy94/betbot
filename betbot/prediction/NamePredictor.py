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

from betbot.api.Match import Match
from numpy import concatenate, ndarray, array
from sklearn.feature_extraction.text import CountVectorizer
from typing import Dict, Union, Optional, Tuple
from betbot.prediction.SKLearnPredictor import SKLearnPredictor


class NamePredictor(SKLearnPredictor):
    """
    scikit-learn powered predictor that uses only the team names
    """

    @classmethod
    def name(cls) -> str:
        """
        :return: The name of the predictor
        """
        return "name-only"

    # noinspection PyMethodMayBeStatic
    def interpret_results(self, home_result: float, away_result: float) -> \
            Tuple[int, int]:
        """
        Interprets the raw results
        :param home_result: The home goals result
        :param away_result: The away goals result
        :return: The home goals, the away goals
        """
        home_converted = (1 / home_result) - 1
        away_converted = (1 / away_result) - 1
        min_score = min([home_converted, away_converted])
        normer = round(min_score)
        home_score = round(home_converted - min_score + normer)
        away_score = round(away_converted - min_score + normer)
        if home_score == away_score:
            if min_score == home_converted:
                away_score += 1
            else:
                home_score += 1

        return home_score, away_score

    def vectorize(self, match_data: Dict[str, Union[str, float]]) -> ndarray:
        """
        Defines how a match is vectorized
        :param match_data: The match data to vectorize
        :return: The vector for the match
        """
        team_vectorizer: CountVectorizer = self.model["team_vectorizer"]
        home, away = team_vectorizer.transform([
            match_data["home_team"], match_data["away_team"]
        ]).toarray()
        return concatenate((home, away))

    def predict_match(self, match: Match) -> Optional[Tuple[int, int]]:
        """
        Predicts the result of a single match using the trained model
        :param match: The match to predict
        :return: The home goals and away goals or None
                 if no prediction took place
        """
        match_data: Dict[str, Union[str, float]] = {
            "home_team": match.home_team,
            "away_team": match.away_team
        }
        vector = array([self.vectorize(match_data)])
        results = self.model["regressor"].predict(vector)[0]
        return self.interpret_results(*results)
