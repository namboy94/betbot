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
import random
from typing import Tuple, List
from keras.models import Model, load_model


class Trainer:
    """
    Class that models common functions of a neural network trainer
    """

    def __init__(self, model_dir: str):
        """
        Initializes the trainer
        :param model_dir: The directory in which to store models
        """
        self.model_dir = model_dir
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        self.name = self.__class__.__name__
        self.model_path = os.path.join(model_dir, self.name)

    def load_training_data(self, force_refresh: bool) \
            -> List[Tuple[List[float], List[float]]]:
        """
        Loads data for training the model
        :param force_refresh: Forces a refresh of data
        :return: The training data, with the input and output vectors separate
        """
        raise NotImplementedError()

    def define_model(self) -> Model:
        """
        Specifies the model of the neural network
        :return: The model
        """
        raise NotImplementedError()

    def evaluate(
            self,
            predictions: List[List[float]],
            expected_output: List[List[float]]
    ) -> float:
        """
        Evaluates predictions of the neural network
        :param predictions: The predictions to check
        :param expected_output: The expected output for the predictions
        :return: A percentage of how well the prediction matches the
                 expected output
        """
        raise NotImplementedError()

    def load_trained_model(self, force_retrain: bool) -> Model:
        """
        Generates a trained keras model that can be used to predict output
        immediately
        :param force_retrain: Whether or not to force retraining
        :return: The trained model
        """
        if force_retrain or not os.path.exists(self.model_path):
            model = self.train(3, 100, 25)[0]  # TODO Adjust these parameters
        else:
            model = load_model(self.model_path)
        model.save(self.model_path)
        return model

    def train(self, iterations: int, epochs: int, batch_size: int) \
            -> Tuple[Model, float]:
        """
        Trains the model using the training data for a specified amount of
        times and returns the best performing model
        Iterations and epochs are different things!
        Each iteration starts from scratch.
        :param iterations: The amount of iterations
        :param epochs: The amount of epochs to train
        :param batch_size: The batch size to use
        :return: The best trained model and its accuracy/score
        """
        best_score = -1.0
        best_model = None
        for i in range(iterations):
            train_in, train_out, valid_in, valid_out, test_in, test_out = \
                self.prepare_training_data()
            model = self.define_model()
            model.fit(
                train_in,
                train_out,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(valid_in, valid_out)
            )
            predictions = [
                [float(x) for x in vector]
                for vector in model.predict(test_in).tolist()
            ]
            score = self.evaluate(predictions, test_out)
            if score > best_score or best_model is None:
                best_score = score
                best_model = best_model
        return best_model, best_score

    def prepare_training_data(self) -> Tuple[
        List[List[float]],
        List[List[float]],
        List[List[float]],
        List[List[float]],
        List[List[float]],
        List[List[float]]
    ]:
        """
        Prepares the training data by shuffling the data and splitting
        it up into a training, validation and test set
        :return: The prepared training data
        """
        labelled_data = self.load_training_data(False)
        random.shuffle(labelled_data)
        inputs = [x[0] for x in labelled_data]
        outputs = [x[1] for x in labelled_data]
        divider_1 = int(len(inputs) / 2)
        divider_2 = int(3 * len(inputs) / 4)
        train_in = inputs[0:divider_1]
        train_out = outputs[0:divider_1]
        valid_in = inputs[divider_1:divider_2]
        valid_out = outputs[divider_1:divider_2]
        test_in = inputs[divider_2:]
        test_out = outputs[divider_2:]
        return train_in, train_out, valid_in, valid_out, test_in, test_out
