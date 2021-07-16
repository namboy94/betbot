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
from typing import List, Tuple
from betbot.api.ApiConnection import ApiConnection
from betbot.prediction import predictors


def main(
        predictor_name: str,
        username: str,
        password: str,
        url: str,
        loop: bool = False
):
    """
    The main function of the betbot
    Predicts results and places bets accordingly
    :param predictor_name: The name of the predictor to use
    :param username: The username for bundesliga-tippspiel
    :param password: The password for bundesliga-tippspiel
    :param url: The base URL for the bundesliga-tippspiel instance
    :param loop: If true will place new bets once an hour
    :return: None
    """
    logging.info(f"Starting betbot for predictor {predictor_name} "
                 f"and user {username}@{url}")

    api = ApiConnection(username, password, url)

    if not api.authorized():
        return

    predictor_map = {
        x.name(): x for x in predictors
    }
    predictor_cls = predictor_map[predictor_name]

    while True:
        leagues = api.get_active_leagues()
        for league, season in leagues:
            logging.info(f"Placing bets for league {league}/{season}")
            predictor = predictor_cls(api, league, season)
            matches = api.get_current_matchday_matches(league, season)
            bets = predictor.predict(matches)
            api.place_bets(bets)

        if loop:
            time.sleep(60 * 60)
        else:
            break

    api.logout()


def multi_main(
        url: str,
        config: List[Tuple[str, str, str]],
        loop: bool = False
):
    """
    Predicts using multiple procedures simultaneously
    :param url: The base URL for the bundesliga-tippspiel site
    :param config: The predictor names and credentials for the API
    :param loop: Whether or not to loop this once an hour
    :return: None
    """
    while True:
        for predictor_name, user, password in config:
            main(predictor_name, user, password, url, False)
        if loop:
            time.sleep(60 * 60)
        else:
            break
