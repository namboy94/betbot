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


class LabelledData:
    """
    Class that models labelled data used for supervised training
    """

    def __init__(self, inputs: List[float], outputs: List[float]):
        """
        Initializes the labelled data
        :param inputs: The inputs
        :param outputs: The expected outputs
        """
        self.inputs = inputs
        self.outputs = outputs
