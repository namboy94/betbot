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

import os
from typing import List, Tuple, Dict, Union, Optional, Any

from betbot.data.OddsPortal import OddsPortal
from joblib import load, dump
from numpy import ndarray, array, concatenate
from betbot.api.Bet import Bet
from betbot.api.Match import Match
from betbot.api.ApiConnection import ApiConnection
from betbot.prediction.Predictor import Predictor
from betbot.data.FootballDataCoUk import FootballDataUk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPRegressor


# noinspection PyAbstractClass
class SKLearnPredictor(Predictor):
    """
    Abstract class that defines how a scikit-learn-based predictor
    should operate
    """

    def __init__(self, api: ApiConnection, league: str, season: int):
        """
        Initializes the scikit-learn model
        :param api: The bundesliga-tippspiel API connection
        :param league: The league for which to predict matches
        :param season: The season for which to predict matches
        """
        super().__init__(api, league, season)
        self.__odds: \
            Optional[Dict[Tuple[str, str], Tuple[float, float, float]]] = None
        self.model_path = os.path.join(self.model_dir, self.name() + ".model")
        self.history_path = os.path.join(self.model_dir, "history")
        self.fetcher = FootballDataUk(self.history_path)
        os.makedirs(self.history_path, exist_ok=True)

        if not os.path.isfile(self.model_path):
            self.model: Dict[str, Any] = {}
            self.logger.info("Training model")
            self.train()
            dump(self.model, self.model_path)
        else:
            self.model = load(self.model_path)

    @property
    def odds(self) -> Dict[Tuple[str, str], Tuple[float, float, float]]:
        """
        Retrieves current odds using a mix of football-data.co.uk and
        oddsportal.com
        :return: The odds for each match in the selected league
        """
        if self.__odds is None:
            self.__odds = self.fetcher.get_odds()
            if len(self.__odds) < 9:
                oddsportal = OddsPortal()
                self.__odds.update(oddsportal.get_odds(self.league))
        return self.__odds

    @classmethod
    def regressor(cls) -> MLPRegressor:
        """
        Defines the regressor used during the prediction process
        :return: The predictor
        """
        return MLPRegressor(hidden_layer_sizes=(64,))

    def train(self):
        """
        Trains the prediction model
        """
        self.fetcher.download_history()
        matches = self.fetcher.get_history_matches()
        team_vectorizer = CountVectorizer(binary=True)
        team_vectorizer.fit([
            " ".join([x["home_team"] for x in matches]),
            " ".join([x["away_team"] for x in matches])
        ])
        self.model["team_vectorizer"] = team_vectorizer

        regressor = self.regressor()

        inputs = []
        outputs = []
        for match in matches:
            inputs.append(self.vectorize(match))
            outputs.append(
                self.encode_result(match["home_score"], match["away_score"])
            )
        regressor.fit(array(inputs), array(outputs))
        self.model["regressor"] = regressor

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
        odds = array([
            1 / float(match_data["home_odds"]),
            1 / float(match_data["draw_odds"]),
            1 / float(match_data["away_odds"]),
        ])
        return concatenate((home, away, odds))

    def predict_match(self, match: Match) -> Optional[Tuple[int, int]]:
        """
        Predicts the result of a single match using the trained model
        :param match: The match to predict
        :return: The home goals and away goals or None
                 if no prediction took place
        """
        match_tuple = (match.home_team, match.away_team)
        odds = self.odds.get(match_tuple)
        if odds is None:
            return None

        match_data: Dict[str, Union[str, float]] = {
            "home_team": match.home_team,
            "away_team": match.away_team,
            "home_odds": float(odds[0]),
            "draw_odds": float(odds[1]),
            "away_odds": float(odds[2])
        }
        vector = array([self.vectorize(match_data)])
        results = self.model["regressor"].predict(vector)[0]
        home_score, away_score = results
        min_score = min([home_score, away_score])
        normer = round(min_score)
        home_score = round(home_score - min_score + normer)
        away_score = round(away_score - min_score + normer)
        return home_score, away_score

    # noinspection PyMethodMayBeStatic
    def encode_result(self, home_score: int, away_score: int) -> ndarray:
        """
        Encodes a result vector
        This is done as a normalization step.
        :param home_score: The home score to encode
        :param away_score: The away score to encode
        :return: The encoded result vector
        """
        encoded_home_score = 1 if home_score == 0 else 1 / (home_score + 1)
        encoded_away_score = 1 if away_score == 0 else 1 / (away_score + 1)
        return array([encoded_home_score, encoded_away_score])

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
        return home_score, away_score

    def predict(self, matches: List[Match]) -> List[Bet]:
        """
        Performs the prediction
        :param matches: The matches to predict
        :return: The predictions as Bet objects
        """
        bets = []
        for match in matches:
            prediction = self.predict_match(match)
            if prediction is not None:
                home_goals, away_goals = prediction
                bets.append(Bet(match, home_goals, away_goals))
        return bets
