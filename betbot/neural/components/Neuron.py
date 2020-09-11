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
from betbot.neural.functions.Function import Function
if TYPE_CHECKING:
    from betbot.neural.components.Weights import Weights


class Neuron:
    """
    Class that models a neuron in a neural network
    """

    def __init__(
            self,
            index: int,
            layer_index: int,
            activation_function: Function,
            input_count: int
    ):
        """
        Initializes the neuron
        :param index: The index of the neuron within the layer
        :param layer_index: The index of the layer within the network
        :param activation_function: The activation function to use
        :param input_count: The amount of inputs the neuron has
        """
        self.index = index
        self.layer_index = layer_index
        self.activation_function = activation_function
        self.input_count = input_count
        self.last_output = 0.0

    def execute(self, input_data: List[float], weights: "Weights") -> float:
        """
        Executes the neuron
        :param input_data: The input data to process
        :param weights: The weights of the neural network
        :return: The result of the execution after the activation function
        """
        neuron_weights = weights.get_neuron_weights(self)
        summed = 0.0
        for i, input_ in enumerate(input_data):
            weight = neuron_weights[i]
            summed += input_ * weight
        self.last_output = self.activation_function.function(summed)
        return self.last_output
