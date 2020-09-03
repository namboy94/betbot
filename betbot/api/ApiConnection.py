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
from typing import List, Dict
from base64 import b64encode
from betbot.api.Bet import Bet
from betbot.api.Match import Match


class ApiConnection:

    def __init__(self, username: str, password: str, url: str):
        self.username = username
        self.password = password
        self.url = os.path.join(url, "api/v2") + "/"
        self.api_key = self.login()

    @property
    def headers(self) -> Dict[str, str]:
        return {"Authorization": "Basic " + self.api_key}

    def login(self) -> str:
        creds = {"username": self.username, "password": self.password}
        resp = requests.post(self.url + "key", json=creds)
        data = json.loads(resp.text)
        api_key = data["data"]["api_key"]
        return b64encode(api_key.encode("utf-8")).decode("utf-8")

    def get_current_matchday_matches(self) -> List[Match]:
        url = self.url + "match?matchday=-1"
        resp = requests.get(url, headers=self.headers)
        data = json.loads(resp.text)
        matches = [Match.from_json(x) for x in data["data"]["matches"]]
        return matches

    def place_bets(self, bets: List[Bet]):
        bet_dict = {}
        for bet in bets:
            bet_dict.update(bet.to_dict())
        url = self.url + "bet"
        resp = requests.put(url, headers=self.headers, json=bet_dict)
        print(resp)


if __name__ == '__main__':
    api = ApiConnection("a", "a", "http://localhost:5000")
    print(api.api_key)
    matches = api.get_current_matchday_matches()
    bets = []
    for match in matches:
        bets.append(Bet(match.id, 10, 10))
    api.place_bets(bets)
