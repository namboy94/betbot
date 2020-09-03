#!/usr/bin/env python3
"""LICENSE
Copyright 2017 Hermann Krumrey <hermann@krumreyh.com>

This file is part of bundesliga-tippspiel.

bundesliga-tippspiel is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

bundesliga-tippspiel is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with bundesliga-tippspiel.  If not, see <http://www.gnu.org/licenses/>.
LICENSE"""

import time
import logging
from betbot.api.ApiConnection import ApiConnection
from betbot.prediction import predictors


def main(predictor_name: str, username: str, password: str, url: str):
    """
    The main function of the betbot
    Periodically predicts results and places bets accordingly
    :param predictor_name: The name of the predictor to use
    :param username: The username for bundesliga-tippspiel
    :param password: The password for bundesliga-tippspiel
    :param url: The base URL for the bundesliga-tippspiel instance
    :return: None
    """
    logging.info(f"Starting betbot for predictor {predictor_name} "
                 f"and user {username}@{url}")

    predictor_map = {
        x.name(): x for x in predictors
    }
    predictor = predictor_map[predictor_name]()
    api = ApiConnection(username, password, url)

    while True:
        matches = api.get_current_matchday_matches()
        bets = predictor.predict(matches)
        api.place_bets(bets)
        time.sleep(60 * 60)