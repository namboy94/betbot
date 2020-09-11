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

from typing import List, TYPE_CHECKING
from betbot.neural.components.Neuron import Neuron
from betbot.neural.functions.ActivationFunction import ActivationFunction
if TYPE_CHECKING:
    from betbot.neural.components.Weights import Weights


class Layer:
    """
    Class that models a layer in a neural network
    """

    def __init__(
            self,
            inputs: int,
            outputs: int,
            activation_function: ActivationFunction,
            has_bias: bool
    ):
        """
        Initializes the layer
        :param inputs: The amount of inputs
        :param outputs: The amount of outputs/neurons
        :param activation_function: The activation function to use
        :param has_bias: Whether or not the layer has a bias
        """
        self._index = 0
        self.inputs = inputs
        self.outputs = outputs
        self.activation_function = activation_function
        self.has_bias = has_bias
        if self.has_bias:
            self.inputs += 1
        self.neurons = [
            Neuron(i, self.index, activation_function, self.inputs)
            for i in range(0, self.outputs)
        ]

    @property
    def index(self) -> int:
        """
        :return: The index of the layer
        """
        return self._index

    @index.setter
    def index(self, new: int):
        """
        Setter method for the layer index
        :param new: The new layer index
        :return: None
        """
        self._index = new
        for neuron in self.neurons:
            neuron.layer_index = new

    def feed_forward(
            self,
            input_data: List[float],
            weights: "Weights"
    ) -> List[float]:
        """
        Performs a feed-forward iteration on input data
        :param input_data: The input data
        :param weights: The weights of the neural network
        :return: The resulting output from the neurons
        """
        raise NotImplementedError()
