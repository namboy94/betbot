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

import random
from typing import List, Optional, Union
from betbot.neural.layer.Layer import Layer
from betbot.neural.components.Neuron import Neuron


class Weights:
    """
    Class that abstracts the weights of a neural network to make them more
    intuitive to use
    """

    def __init__(
            self,
            layers: List[Layer],
            initial_weights: Optional[List[List[List[float]]]] = None
    ):
        """
        Initializes the weights
        :param layers: The layers of the neural network
        :param initial_weights: The initial weights. If this is None, random
                                weights will be used.
                                The weights are a three-dimensional matrix,
                                with the values corresponding to the
                                layer, neuron and input.
        """
        self._layers = layers
        if initial_weights is None:
            self._weights = self.generate_random_weights()
        else:
            self._weights = initial_weights

    def generate_random_weights(self) -> List[List[List[float]]]:
        """
        Generates random weights for all layers, neurons and inputs
        :return: The generated weights
        """
        weights: List[List[List[float]]] = []
        for layer in self._layers:
            weights.append([])
            for neuron in layer.neurons:
                weights[-1].append([])
                for _ in range(neuron.input_count):
                    weights[-1][-1].append(random.random())
        return weights

    def get_weight(
            self,
            layer: Union[int, Layer],
            neuron: Union[int, Neuron],
            input_: int
    ) -> float:
        """
        Retrieves a weight for a layer, neuron and input.
        Allows for use of Layer or Neuron objects instead of indices
        :param layer: The layer for which to get the weight
        :param neuron: The neuron for which to get the weight
        :param input_: The input of the neuron for which to get the weight
        :return: The weight
        """
        if isinstance(layer, Layer):
            layer = layer.index
        if isinstance(neuron, Neuron):
            neuron = neuron.index
        return self._weights[layer][neuron][input_]

    def get_neuron_weights(self, neuron: Neuron) -> List[float]:
        """
        Gets the weights for a single neuron's inputs
        :param neuron: The neuron
        :return: The weights for the neuron's inputs
        """
        return [
            self.get_weight(neuron.layer_index, neuron.index, i)
            for i in range(0, neuron.input_count)
        ]

    def set_weight(
            self,
            layer: Union[int, Layer],
            neuron: Union[int, Neuron],
            input_: int,
            weight: float
    ):
        """
        Sets a new value for a weight for a layer, neuron and input.
        Allows for use of Layer or Neuron objects instead of indices
        :param layer: The layer for which to set the weight
        :param neuron: The neuron for which to set the weight
        :param input_: The input of the neuron for which to set the weight
        :param weight: The weight to set
        :return: None
        """
        if isinstance(layer, Layer):
            layer = layer.index
        if isinstance(neuron, Neuron):
            neuron = neuron.index
        self._weights[layer][neuron][input_] = weight

    def to_matrix(self) -> List[List[List[float]]]:
        """
        :return: The weights as a 3-dimensional matrix
        """
        return self._weights
