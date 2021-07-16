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
import json
import requests
import logging
from typing import List, Dict, Optional, Any, Tuple
from base64 import b64encode
from betbot.api.Bet import Bet
from betbot.api.Match import Match


class ApiConnection:
    """
    Class that handles API calls to the bundesliga-tippspiel instance
    """

    def __init__(self, username: str, password: str, url: str):
        """
        Intitializes the API connection
        :param username: The username on bundesliga-tippspiel
        :param password: The password for that user
        :param url: The base url to the bundesliga-tippspiel instance
        """
        self.logger = logging.getLogger(__name__)
        self.username = username
        self.password = password
        self.url = os.path.join(url, "api/v3") + "/"
        self.api_key = ""
        self.login()
        self.logger.debug("Initialized")

    @property
    def auth_headers(self) -> Dict[str, str]:
        """
        :return: Authorization headers for API calls
        """
        return {"Authorization": "Basic " + self.api_key}

    def login(self) -> bool:
        """
        Retrieves an API Key using a username and password if no key
        has been retrieved yet or if the existing key has expired
        :return: True if the login was successfulle
        """
        if not self.authorized():
            self.logger.info("Requesting new API Key")
            creds = {"username": self.username, "password": self.password}
            data = self.execute_api_call("key", "POST", False, creds)

            if data["status"] != "ok":
                self.logger.warning("Login attempt failed")
                return False
            else:
                api_key = data["data"]["api_key"]
                encoded = b64encode(api_key.encode("utf-8"))
                self.api_key = encoded.decode("utf-8")
                self.logger.info("Login successful")
                return True
        else:
            return True

    def authorized(self) -> bool:
        """
        Checks if the stored API key is valid
        :return: True if valid, False if not (for example because it expired)
        """
        self.logger.debug("Authorization Check")
        data = self.execute_api_call("authorize", "GET", True)
        return data["status"] == "ok"

    def logout(self):
        """
        Logs out the bot by deleting the API key
        :return: None
        """
        self.execute_api_call(
            "key", "DELETE", json_data={"api_key": self.api_key}
        )
        self.logger.info("Logging out.")

    def get_active_leagues(self) -> List[Tuple[str, int]]:
        """
        :return: A list of tuples of leagues and seasons of active leagues
        """
        leagues = self.execute_api_call(
            "leagues", "GET", True
        )["data"]["leagues"]
        newest_season = max([x[1] for x in leagues])
        return [x for x in leagues if x[1] == newest_season]

    def get_league_table(self, season: str, league: int) -> List[str]:
        """
        Retrieves the current league table order
        :param season: The league for which to retrieve the league table
        :param league: The season for which to retrieve the league table
        :return: The team abbreviations in order of their league rankings
        """
        league_table = self.execute_api_call(
            f"league_table/{season}/{league}", "GET", True
        )["data"]["league_table"]
        return [x[1]["abbreviation"] for x in league_table]

    def get_current_matchday_matches(
            self, league: str, season: int
    ) -> List[Match]:
        """
        Retrieves a list of matches for the current matchday
        :param league: The league to retrieve matches for
        :param season: The season to retrieve matches for
        :return: The list of matches
        """
        self.logger.debug("Getting current matches")
        match_data = self.execute_api_call(
            f"matchday/{league}/{season}", "GET", True
        )["data"]["matches"]
        return [Match.from_json(x) for x in match_data]

    def place_bets(self, bets: List[Bet]):
        """
        Places a list of bets
        :param bets: The bets to place
        :return: None
        """
        for bet in bets:
            self.logger.info(f"Placing bet: "
                             f"{bet.match.home_team} VS {bet.match.away_team}:"
                             f" {bet.home_score}:{bet.away_score}")

        bet_dicts = [bet.to_dict() for bet in bets]
        data = self.execute_api_call(
            "place_bets", "PUT", True, {"bets": bet_dicts}
        )
        if data["status"] != "ok":
            self.logger.error("Failed to place bets")
        else:
            placed = len(data["data"]["placed"])
            self.logger.info(f"Placed {placed} bets (user:{self.username})")

    def execute_api_call(
            self,
            endpoint: str,
            method: str,
            authorization_required: bool = False,
            json_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes an API call
        :param endpoint: The API endpoint
        :param method: The request method
        :param authorization_required: Whether authoirzation is required
        :param json_data: The JSON data to send
        :return: The response JSON
        """
        if authorization_required and endpoint != "authorize":
            logged_in = self.login()
            if not logged_in:
                return {"status": "error"}

        extras = {}
        if authorization_required:
            extras["headers"] = self.auth_headers
        if json_data is not None:
            extras["json"] = json_data

        api_url = self.url + endpoint
        resp = requests.request(method, api_url, **extras)  # type: ignore
        return json.loads(resp.text)
