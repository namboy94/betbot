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
from typing import List, Optional
from betbot.neural.components.Layer import Layer


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

    def train(self):
        raise NotImplementedError()

    def classify(self, inputs: List[float]) -> List[float]:
        for i, layer in enumerate(self.layers):
            inputs = layer.execute(inputs, self.weights[i])
        return inputs

    def save_weights(self, file_path: str):
        with open(file_path, "w") as f:
            json.dump({"weights": self.weights}, f)

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


