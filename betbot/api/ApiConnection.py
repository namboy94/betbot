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
from typing import List, Dict
from base64 import b64encode
from betbot.api.Bet import Bet
from betbot.api.Match import Match


class ApiConnection:

    def __init__(self, username: str, password: str, url: str):
        self.logger = logging.getLogger(__name__)
        self.username = username
        self.password = password
        self.url = os.path.join(url, "api/v2") + "/"
        self.api_key = ""
        self.login()

    @property
    def headers(self) -> Dict[str, str]:
        return {"Authorization": "Basic " + self.api_key}

    def login(self):
        if not self.authorized():
            self.logger.info("Requesting new API Key")
            creds = {"username": self.username, "password": self.password}
            resp = requests.post(self.url + "key", json=creds)
            data = json.loads(resp.text)
            api_key = data["data"]["api_key"]
            self.api_key = b64encode(api_key.encode("utf-8")).decode("utf-8")

    def authorized(self) -> bool:
        url = self.url + "authorize"
        resp = requests.get(url, headers=self.headers)
        data = json.loads(resp.text)
        return data["status"] == "ok"

    def get_current_matchday_matches(self) -> List[Match]:
        self.login()
        url = self.url + "match?matchday=-1"
        resp = requests.get(url, headers=self.headers)
        data = json.loads(resp.text)
        matches = [Match.from_json(x) for x in data["data"]["matches"]]
        return matches

    def place_bets(self, bets: List[Bet]):
        self.login()
        bet_dict = {}
        for bet in bets:
            bet_dict.update(bet.to_dict())
        url = self.url + "bet"
        requests.put(url, headers=self.headers, json=bet_dict)
