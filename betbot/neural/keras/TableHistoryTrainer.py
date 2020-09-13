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
from typing import List, Tuple
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense
from betbot.neural.keras.BetPredictorTrainer import BetPredictorTrainer
from betbot.neural.data.openligadb.processing import load_csv, \
    generate_training_csv


class TableHistoryTrainer(BetPredictorTrainer):
    """
    Trainer Class that specifies a neural network for use with historical
    table data
    """

    def load_training_data(self, force_refresh: bool) \
            -> List[Tuple[List[float], List[float]]]:
        """
        Loads data for training the model
        :param force_refresh: Forces a refresh of data
        :return: The training data, with the input and output vectors separate
        """
        output = os.path.join(self.model_dir, "table-history")
        csv_file = output + ".csv"
        if not os.path.isfile(csv_file) or force_refresh:
            generate_training_csv(csv_file)
        vectors = load_csv(csv_file)
        return vectors

    def _define_model(self) -> Model:
        """
        Specifies the model of the neural network
        :return: The model
        """
        model = Sequential()
        model.add(Flatten(input_shape=(12,)))
        model.add(Dense(20, activation="sigmoid"))
        model.add(Dense(2, activation="relu"))
        return model

    def _compile_model(self, model: Model):
        """
        Compiles the keras model
        :param model: The model to compile
        :return: None
        """
        model.compile(loss="mae", optimizer="nadam")
