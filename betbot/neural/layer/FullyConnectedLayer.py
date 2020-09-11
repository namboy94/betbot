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

from typing import List
from betbot.neural.layer.Layer import Layer


class FullyConnectedLayer(Layer):
    """
    Class that models a fully connected layer in a neural network
    """

    def execute(
            self,
            input_data: List[float],
            weights: List[List[float]]
    ) -> List[float]:
        outputs = []
        for i in range(0, self.outputs):
            neuron = self.neurons[i]
            neuron_weights = weights[i]
            outputs.append(neuron.execute(input_data, neuron_weights))

        return outputs
