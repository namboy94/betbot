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

from typing import List, Tuple
from betbot.neural.NeuralNetworkFramework import NeuralNetworkFramework


class MLP(NeuralNetworkFramework):
    """
    Class that models a multi-layer perceptron
    """

    def train(
            self,
            labelled_data: List[Tuple[List[float], List[float]]],
            learning_rate: float,
            epochs: int
    ):
        training_data, validation_data, test_data = \
            self.split_labelled_data(labelled_data)

        for epoch in range(0, epochs):
            for inputs, expected_outputs in training_data:
                errors = self.back_propagate(inputs, expected_outputs)
                self.adjust_weights(errors, inputs, learning_rate)
                calculated = self.classify(inputs)
                total_error = 0
                for i, expected in enumerate(expected_outputs):
                    print(expected_outputs)
                    print(calculated)
                    total_error += (expected - calculated[i]) ** 2

                print(f"Epoch {epoch}: Error: {total_error:.3f}")

            self.display_accuracy(test_data)

    def back_propagate(self, inputs: List[float], expected_outputs: List[float]) -> List[List[float]]:

        deltas = []

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
                        weight = self.weights[previous_layer_index][j][i]
                        delta = deltas[previous_layer_number][j]
                        error += weight * delta
                    errors.append(error)

            for weight_index, neuron in enumerate(layer.neurons):
                neuron_error = errors[weight_index]
                delta = neuron_error * layer.activation_function.derivative(neuron.last_output)
                deltas[layer_number].append(delta)

        return list(reversed(deltas))

    def adjust_weights(self, errors: List[List[float]], inputs: List[float], learning_rate: float):
        for layer_index, layer in enumerate(self.layers):

            if layer_index == 0:
                layer_inputs = inputs
            else:
                previous_layer = self.layers[layer_index - 1]
                layer_inputs = [x.last_output for x in previous_layer.neurons]

            for neuron_index, neuron in enumerate(layer.neurons):
                neuron_weights = self.weights[layer_index][neuron_index]
                for i, input_data in enumerate(layer_inputs):
                    current_weight = neuron_weights[i]
                    error = errors[layer_index][neuron_index]
                    new_weight = current_weight + learning_rate * error * input_data
                    self.weights[layer_index][neuron_index][i] = new_weight

    def display_accuracy(self, test_data: List[Tuple[List[float], List[float]]]):
        correct = 0
        for inputs, expected_outputs in test_data:
            calculated = self.classify(inputs)
            if calculated == expected_outputs:
                correct += 1
