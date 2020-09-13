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

from unittest import TestCase
from betbot.neural.data.openligadb.InputVector import InputVector
from betbot.neural.data.openligadb.processing import load_data, \
    calculate_history


class TestOpenligaDbVectors(TestCase):
    """
    Class that tests the generation of training data using openligadb
    """

    def test_vectorization(self):
        """
        Tests if the input/output vectors are generated correctly
        """
        bl2018 = load_data("bl1", 2018)[1]
        bl2017 = load_data("bl1", 2017)[1]
        history = calculate_history({2018: bl2018, 2017: bl2017})

        fc_bayern = history[40]
        dortmund = history[7]
        schalke = history[9]
        bayer = history[6]
        wolfsburg = history[131]
        self.assertEqual(fc_bayern.get_stats(2018, 17, False), (36, 36, 18))
        self.assertEqual(fc_bayern.get_stats(2018, 17, True), (33, 33, 18))
        self.assertEqual(fc_bayern.get_stats(2018, 1, False), (3, 3, 1))
        self.assertEqual(fc_bayern.get_stats(2017, 1, True), (0, 0, 0))
        self.assertEqual(
            fc_bayern.get_stats(2018, 1, True),
            fc_bayern.get_stats(2017, 34, False)
        )

        print(fc_bayern.get_stats_interval(2017, 6, 5, True))

        self.assertEqual(
            InputVector(2018, 11, True, dortmund, fc_bayern).vector,
            [24, 30, 10] + [20, 18, 11] + [13, 15, 7] + [7, 6, 8]
        )
        self.assertEqual(
            InputVector(2018, 28, True, fc_bayern, dortmund).vector,
            [61, 69, 28] + [63, 66, 30] + [13, 19, 2] + [12, 12, 7]
        )
        self.assertEqual(
            InputVector(2018, 4, True, schalke, fc_bayern).vector,
            [0, 2, 6] + [9, 9, 2] + [6, 5, 7] + [12, 13, 7]
        )
        self.assertEqual(
            InputVector(2017, 1, True, fc_bayern, bayer).vector,
            [0, 0, 0] + [0, 0, 0] + [0, 0, 0] + [0, 0, 0]
        )
        self.assertEqual(
            InputVector(2017, 6, True, fc_bayern, wolfsburg).vector,
            [12, 12, 3, 5, 3, 6, 12, 12, 3, 5, 3, 6]
        )
