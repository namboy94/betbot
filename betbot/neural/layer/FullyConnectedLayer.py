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
from betbot.neural.components.Weights import Weights


class FullyConnectedLayer(Layer):
    """
    Class that models a fully connected layer in a neural network
    """

    def feed_forward(
            self,
            input_data: List[float],
            weights: Weights
    ) -> List[float]:
        """
        Performs a feed-forward iteration on input data
        :param input_data: The input data
        :param weights: The weights of the neural network
        :return: The resulting output from the neurons
        """
        outputs = []
        input_ = input_data
        if self.has_bias:
            input_ = [1.0] + input_
        for neuron in self.neurons:
            outputs.append(neuron.execute(input_, weights))
        return outputs
