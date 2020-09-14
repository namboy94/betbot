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
from betbot.neural.keras.Trainer import Trainer


# noinspection PyAbstractClass
class BetPredictorTrainer(Trainer):
    """
    Trainer Class that specifies an evaluation for bets
    """

    def _evaluate(
            self,
            predictions: List[List[float]],
            expected_output: List[List[float]]
    ) -> Tuple[float, float]:
        """
        Evaluates predictions of the neural network
        :param predictions: The predictions to check
        :param expected_output: The expected output for the predictions
        :return: The evaluation score and a percentage of how well the
                 prediction matches the expected output
        """
        points = 0

        for i, expected in enumerate(expected_output):
            predicted = [float(round(x)) for x in predictions[i]]

            predicted_diff = predicted[0] - predicted[1]
            expected_diff = expected[0] - expected[1]

            if predicted_diff * expected_diff > 0:
                points += 7
            elif predicted_diff == 0 and expected_diff == 0:
                points += 7

            if predicted_diff == expected_diff:
                points += 5

            if predicted[0] == expected[0] or predicted[1] == expected[1]:
                points += 3

        score = points / len(predictions)
        accuracy = 100 * score / 15
        return score, accuracy
