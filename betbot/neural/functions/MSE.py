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
from betbot.neural.functions.ErrorFunction import ErrorFunction


class MSE(ErrorFunction):
    """
    Class that models the MSE (Meas Square Error) error function
    """

    def calculate_total_error(
            self,
            output: List[float],
            expected: List[float]
    ) -> float:
        """
        Calculates the total error for an output with known results
        :param output: The output to check
        :param expected: The label/known output
        :return: The error calculated using the error function
        """
        summed = 0.0
        for i, output_value in enumerate(output):
            summed += (1/2) * ((expected[i] - output_value) ** 2)
        return summed
