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
import logging
from typing import List
from betbot.api.ApiConnection import ApiConnection
from betbot.api.Match import Match
from betbot.api.Bet import Bet


class Predictor:
    """
    Class that specifies required methods for predictor objects
    """

    def __init__(self, api: ApiConnection, league: str, season: int):
        """
        Initializes the model directory if it does not exist
        :param api: The bundesliga-tippspiel API connection
        :param league: The league for which to predict matches
        :param season: The season for which to predict matches
        """
        self.api = api
        self.league = league
        self.season = season
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_dir = os.path.join(
            os.path.expanduser("~"),
            ".config/betbot"
        )
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)

    @classmethod
    def name(cls) -> str:
        """
        :return: The name of the predictor
        """
        raise NotImplementedError()

    def predict(self, matches: List[Match]) -> List[Bet]:
        """
        Performs the prediction
        :param matches: The matches to predict
        :return: The predictions as Bet objects
        """
        raise NotImplementedError()
