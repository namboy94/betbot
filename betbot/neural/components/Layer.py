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

from typing import Callable, List
from betbot.neural.components.Neuron import Neuron


class Layer:
    """
    Class that models a layer in a neural network
    """

    def __init__(
            self,
            inputs: int,
            outputs: int,
            activation_function: Callable[[float], float],
            has_bias: bool
    ):
        self.inputs = inputs
        self.outputs = outputs
        self.activation_function = activation_function
        self.has_bias = has_bias
        if self.has_bias:
            self.inputs += 1
        self.neurons = [
            Neuron(activation_function)
            for _ in range(0, self.outputs)
        ]

    def execute(
            self,
            input_data: List[float],
            weights: List[List[float]]
    ) -> List[float]:
        raise NotImplementedError()
