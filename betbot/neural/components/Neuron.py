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


class Neuron:
    """
    Class that models a neuron in a neural network
    """

    def __init__(self, activation_function: Callable[[float], float]):
        self.activation_function = activation_function

    def execute(self, input_data: List[float], weights: List[float]) -> float:
        summed = 0.0
        for i in range(0, len(input_data)):
            data = input_data[i]
            weight = weights[i]
            summed += data * weight
        return self.activation_function(summed)
