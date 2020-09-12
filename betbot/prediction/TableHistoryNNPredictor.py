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
from typing import List
from keras.models import load_model
from betbot.api.Bet import Bet
from betbot.api.Match import Match
from betbot.prediction.Predictor import Predictor
from betbot.neural.keras_models import table_history_model
from betbot.neural.data.openligadb.InputVector import InputVector
from betbot.neural.data.openligadb.processing import calculate_history, \
    load_data, generate_training_csv, load_csv


class TableHistoryNNPredictor(Predictor):
    """
    Class that predicts results based on historical table data
    """

    def __init__(self):
        """
        Initializes the Neural Network.
        Loads existing weights if they exist, otherwise will train new weights
        """
        super().__init__()
        self.model_data_path = os.path.join(self.model_dir, self.name())
        if os.path.exists(self.model_data_path):
            self.model = load_model(self.model_data_path)
        else:
            self.model = table_history_model()
            csv_file = self.model_data_path + ".csv"
            if not os.path.isfile(csv_file):
                generate_training_csv(csv_file)
            inputs, outputs = load_csv(csv_file)
            model = table_history_model()
            model.fit(inputs, outputs, epochs=200, batch_size=10)
            model.save(self.model_data_path)

    @classmethod
    def name(cls) -> str:
        """
        :return: The name of the predictor
        """
        return "table-history"

    def predict(self, matches: List[Match]) -> List[Bet]:
        """
        Performs the prediction
        :param matches: The matches to predict
        :return: The predictions as Bet objects
        """
        bets = []
        vectors = self.generate_input_vectors(matches)
        predictions = self.model.predict(vectors).tolist()

        for i, match in enumerate(matches):
            result = [int(round(x)) for x in predictions[i]]
            bets.append(Bet(match.id, result[0], result[1]))

        return bets

    @staticmethod
    def generate_input_vectors(matches: List[Match]) -> List[List[float]]:
        """
        Generates input vectors for matches
        :param matches: The matches
        :return: The input vectors
        """
        league = "bl1"
        season = 2020

        current_teams, current_matches = load_data(league, season)
        previous_teams, previous_matches = load_data(league, season - 1)
        history = calculate_history({
            season: current_matches,
            season - 1: previous_matches
        })
        history_by_abbreviation = {
            item.team.abbreviation: item for item in history.values()
        }

        vectors = []
        for match in matches:
            home_history = history_by_abbreviation[match.home_team]
            away_history = history_by_abbreviation[match.away_team]
            vector = InputVector(
                season,
                match.matchday,
                match.finished,
                home_history,
                away_history
            )
            vectors.append([float(x) for x in vector.vector])
        return vectors
