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

import json
import random
from typing import List, Optional, Tuple
from betbot.neural.layer.Layer import Layer


class NeuralNetworkFramework:
    """
    Class that defines a framework for neural networks
    """

    def __init__(
            self,
            layers: List[Layer],
            initial_weights: Optional[List[List[List[float]]]]
    ):
        self.layers = layers
        if initial_weights is None:
            self.weights = []
            for layer in layers:
                self.weights.append([])
                for i in range(0, layer.outputs):
                    self.weights[-1].append([])
                    for j in range(0, layer.inputs):
                        self.weights[-1][-1].append(random.random())
        else:
            self.weights = initial_weights

    def train(
            self,
            labelled_data: List[Tuple[List[float], float]],
            learning_rate: float,
            epochs: int
    ):
        raise NotImplementedError()

    @staticmethod
    def split_labelled_data(
            labelled_data: List[Tuple[List[float], List[float]]]
    ) -> Tuple[
        List[Tuple[List[float], List[float]]],
        List[Tuple[List[float], List[float]]],
        List[Tuple[List[float], List[float]]]
    ]:
        random.shuffle(labelled_data)

        border_0 = 0
        border_1 = int(len(labelled_data) / 2)
        border_2 = int(3 * len(labelled_data) / 4)
        border_3 = len(labelled_data)

        training_data = labelled_data[border_0:border_1]
        validation_data = labelled_data[border_1:border_2]
        test_data = labelled_data[border_2:border_3]

        return training_data, validation_data, test_data

    def classify(self, inputs: List[float]) -> List[float]:
        for i, layer in enumerate(self.layers):
            if layer.has_bias:
                inputs = [1.0] + inputs
            inputs = layer.execute(inputs, self.weights[i])
        return inputs

    def save_weights(self, file_path: str):
        with open(file_path, "w") as f:
            json.dump({"weights": self.weights}, f, indent=4)

"""
Inputs:

Home Team + Away Team:
    current position in table
    current points per game
    current goals per game
    current goals conceded per game
    points per game last 5 matches
    goals per game last 5 matches
    goals conceded per last 5 matches



"""


