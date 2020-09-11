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
from betbot.neural.data.LabelledData import LabelledData
from betbot.neural.NeuralNetworkFramework import NeuralNetworkFramework


class MLP(NeuralNetworkFramework):
    """
    Class that models a multi-layer perceptron
    """

    def train(
            self,
            labelled_data: List[LabelledData],
            learning_rate: float,
            epochs: int
    ):
        """
        Trains the neural network using backpropagation
        :param labelled_data: The labelled data used to learn
        :param learning_rate: The learning rate
        :param epochs: The amount of iterations to learn
        :return: None
        """
        training_data, validation_data, test_data = \
            self.split_labelled_data(labelled_data)

        for epoch in range(0, epochs):
            for training in training_data:
                errors = self.back_propagate(training.inputs, training.outputs)
                self.adjust_weights(errors, training.inputs, learning_rate)

            self.display_training_iteration(
                epoch, learning_rate, validation_data
            )

    def back_propagate(
            self,
            inputs: List[float],
            expected_outputs: List[float]
    ) -> List[List[float]]:
        """
        Back-propagates the error through the network
        :param inputs: The inputs to pass through the network
        :param expected_outputs: The expected outputs of the output layer
        :return: A 2-dimensional matrix containing the errors for each neuron
        """
        deltas: List[List[float]] = []

        for layer_number, layer in enumerate(reversed(self.layers)):
            layer_index = len(self.layers) - layer_number - 1
            errors = []
            deltas.append([])

            if layer_number == 0:  # Output Layer
                outputs = self.classify(inputs)
                for neuron_number, output in enumerate(outputs):
                    expected = expected_outputs[neuron_number]
                    errors.append(expected - output)

            else:
                previous_layer_number = layer_number - 1
                previous_layer_index = layer_index + 1
                previous_layer = self.layers[previous_layer_index]

                for i in range(0, layer.outputs):
                    error = 0.0
                    for j in range(0, previous_layer.outputs):
                        weight = self.weights.get_weight(
                            layer=previous_layer_index,
                            neuron=j,
                            input_=i
                        )
                        delta = deltas[previous_layer_number][j]
                        error += weight * delta
                    errors.append(error)

            for weight_index, neuron in enumerate(layer.neurons):
                neuron_error = errors[weight_index]
                delta = neuron_error * layer.activation_function.derivative(
                    neuron.last_output
                )
                deltas[layer_number].append(delta)

        return list(reversed(deltas))

    def adjust_weights(
            self,
            errors: List[List[float]],
            inputs: List[float],
            learning_rate: float
    ):
        """
        Adjusts the network's weights based on the errors in the
        backpropagation.
        :param errors: The errors of the backpropagation
        :param inputs: The inputs that led to these errors
        :param learning_rate: The learning rate
        :return: None
        """
        for layer_index, layer in enumerate(self.layers):

            if layer_index == 0:
                layer_inputs = inputs
            else:
                previous_layer = self.layers[layer_index - 1]
                layer_inputs = [x.last_output for x in previous_layer.neurons]

            for neuron_index, neuron in enumerate(layer.neurons):
                neuron_weights = self.weights.get_neuron_weights(neuron)
                for i, input_data in enumerate(layer_inputs):
                    current_weight = neuron_weights[i]
                    error = errors[layer_index][neuron_index]
                    new_weight = \
                        current_weight + learning_rate * error * input_data
                    self.weights.set_weight(
                        layer=layer_index,
                        neuron=neuron_index,
                        input_=i,
                        weight=new_weight
                    )
