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
import json
import random
import logging
from typing import List, Optional, Tuple
from betbot.neural.layer.Layer import Layer
from betbot.neural.components.Weights import Weights
from betbot.neural.data.LabelledData import LabelledData


class NeuralNetworkFramework:
    """
    Class that defines a framework for neural networks
    """

    def __init__(
            self,
            layers: List[Layer],
            model_file: Optional[str] = None
    ):
        """
        Initializes the neural network
        :param layers: The layers of the network
        :param model_file: Optional model file which contains previously
                           trained weights
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.layers = layers
        for i, layer in enumerate(self.layers):
            layer.index = i
        self.weights = self.load_weights(model_file)

    def load_weights(self, model_file: Optional[str]) -> Weights:
        """
        Loads weights from an existing model file
        If the model file is None or does not exist, random initial weights
        will be generated
        :param model_file: The model file to use
        :return: The loaded/generated weights
        """
        if model_file is None or not os.path.isfile(model_file):
            weights = Weights(self.layers)
        else:
            with open(model_file, "r") as f:
                existing_weights = json.load(f)["weights"]
                weights = Weights(self.layers, existing_weights)
        return weights

    def train(
            self,
            labelled_data: List[LabelledData],
            learning_rate: float,
            epochs: int
    ):
        """
        Trains the neural network
        :param labelled_data: The labelled data used to learn
        :param learning_rate: The learning rate
        :param epochs: The amount of iterations to learn
        :return: None
        """
        raise NotImplementedError()

    def classify(self, inputs: List[float]) -> List[float]:
        """
        Performs a forward pass through the network to classify an input
        :param inputs: The inputs to pass through the network
        :return: The outputs from the output layer
        """
        current_inputs = inputs
        for i, layer in enumerate(self.layers):
            current_inputs = layer.feed_forward(current_inputs, self.weights)
        return current_inputs

    def save_model(self, model_file: str):
        """
        Saves the current weights in a model file
        :param model_file: The path to the file to create
        :return: None
        """
        with open(model_file, "w") as f:
            json.dump({
                "weights": self.weights.to_matrix()
            }, f, indent=4)

    @staticmethod
    def split_labelled_data(
            labelled_data: List[LabelledData]
    ) -> Tuple[List[LabelledData], List[LabelledData], List[LabelledData]]:
        """
        Creates a training, validation and test set from labelled data
        in a 2:1:1 ratio
        :param labelled_data: The labelled data
        :return: The training, validation and test sets
        """
        random.shuffle(labelled_data)

        border_0 = 0
        border_1 = int(len(labelled_data) / 2)
        border_2 = int(3 * len(labelled_data) / 4)
        border_3 = len(labelled_data)

        training_data = labelled_data[border_0:border_1]
        validation_data = labelled_data[border_1:border_2]
        test_data = labelled_data[border_2:border_3]

        return training_data, validation_data, test_data

    def display_training_iteration(
            self,
            epoch: int,
            learning_rate: float,
            validation_data: List[LabelledData]
    ) -> float:
        """
        Displays the current state of the classifier after a training iteration
        :param epoch: The epoch of the iteration
        :param learning_rate: The learning rate
        :param validation_data: The validation data
        :return: The average error
        """
        errors: List[float] = []
        for validation in validation_data:
            calculated = self.classify(validation.inputs)
            error = 0.0
            for i, expected in enumerate(validation.outputs):
                error += (expected - calculated[i]) ** 2
            errors.append(error)
        average_error = sum(errors) / len(errors)
        print(f"Epoch: {epoch}; "
              f"Learning Rate: {learning_rate}; "
              f"Error: {average_error:.2f}"
              )
        return average_error


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
