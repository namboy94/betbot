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
import csv
from typing import Union, Dict, List, Tuple

import requests


class FootballDataUk:
    """
    Class that handles retrieving data from football-data.co.uk
    """

    def __init__(self, data_path: str):
        """
        Initializes the object
        :param data_path: The path in which to store the data files
        """
        self.data_path = data_path
        os.makedirs(data_path, exist_ok=True)

    TEAM_MAP = {
        "Nurnberg": "FCN",
        "Mainz": "M05",
        "Leverkusen": "B04",
        "Dortmund": "BVB",
        "M'gladbach": "BMG",
        "Ein Frankfurt": "SGE",
        "Augsburg": "FCA",
        "Bayern Munich": "FCB",
        "Schalke 04": "S04",
        "Fortuna Dusseldorf": "F95",
        "Hannover": "H96",
        "Hertha": "BSC",
        "RB Leipzig": "RBL",
        "Freiburg": "SCF",
        "Hoffenheim": "TSG",
        "Stuttgart": "VFB",
        "Wolfsburg": "WOB",
        "Werder Bremen": "BRE",
        "Union Berlin": "FCU",
        "Paderborn": "SCP",
        "FC Koln": "KOE",
        "Bielefeld": "DSC",
        "Greuther Furth": "SGF",
        "Bochum": "BOC",
        "Heidenheim": "HDH",
        "Erzgebirge Aue": "AUE",
        "Hansa Rostock": "FCH",
        "Ingolstadt": "FCI",
        "St Pauli": "STP",
        "Hamburg": "HSV",
        "Holstein Kiel": "KIE",
        "Regensburg": "SSV",
        "Karlsruhe": "KSC",
        "Dresden": "SGD",
        "Darmstadt": "D98",
        "Sandhausen": "SVS",
    }

    def download_history(self):
        """
        Download history for the first and second bundesliga
        :return: None
        """
        for year in range(2000, 2021):
            next_year = year + 1
            year_string = str(year)[-2:] + str(next_year)[-2:]
            for league in ["D1", "D2"]:
                path = os.path.join(self.data_path, f"{league}-{year}.csv")
                if os.path.isfile(path):
                    continue
                url = f"https://www.football-data.co.uk/mmz4281/" \
                      f"{year_string}/{league}.csv"
                with open(path, "wb") as f:
                    f.write(requests.get(url).content)

    def get_history_matches(self) -> List[Dict[str, Union[str, int, float]]]:
        """
        Retrieves data on historical bundesliga and bundesliga 2 matches
        :return: A list of match dictionaries
        """
        matches = []
        for history_file in [
            os.path.join(self.data_path, x)
            for x in os.listdir(self.data_path)
        ]:
            matches += self.load_matches(history_file)
        return matches

    def load_matches(self, data_file: str) -> \
            List[Dict[str, Union[str, int, float]]]:
        """
        Loads match data from a single CSV file
        :param data_file: The path to the CSV file
        :return: The match data
        """
        matches: List[Dict[str, Union[str, int, float]]] = []
        file_matches = self.load_history_file(data_file)
        for match in file_matches:
            home = self.TEAM_MAP.get(str(match["HomeTeam"]))
            away = self.TEAM_MAP.get(str(match["AwayTeam"]))
            if home is None or away is None:
                continue
            try:
                matches.append({
                    "home_team": home,
                    "away_team": away,
                    "home_score": int(match["FTHG"]),
                    "away_score": int(match["FTAG"]),
                    "home_odds": float(match["WHH"]),
                    "away_odds": float(match["WHA"]),
                    "draw_odds": float(match["WHD"])
                })
            except ValueError:
                continue
        return matches

    @staticmethod
    def load_history_file(history_file: str) -> List[Dict[str, str]]:
        """
        Loads the contents of a history file
        :param history_file: The history file to load
        :return: A list of match dictionaries
        """
        with open(history_file, "r", encoding="latin1") as f:
            lines = [x for x in csv.reader(f)]
        header = lines.pop(0)
        data: List[Dict[str, str]] = []
        for line in lines:
            item = {}
            for i, key in enumerate(header):
                if not key or i >= len(line):
                    continue
                item[key] = line[i]
            data.append(item)
        return data

    def get_odds(self) -> Dict[Tuple[str, str], Tuple[float, float, float]]:
        """
        Loads currently available odds
        :return: The current odds (home, draw, away) mapped to home/away teams
        """
        path = "/tmp/fixtures.csv"
        url = "https://www.football-data.co.uk/fixtures.csv"
        with open(path, "wb") as f:
            f.write(requests.get(url).content)

        odds = {}
        for match in self.load_matches(path):
            match_tuple = (str(match["home_team"]), str(match["away_team"]))
            odds_tuple = (
                float(match["home_odds"]),
                float(match["draw_odds"]),
                float(match["away_odds"])
            )
            odds[match_tuple] = odds_tuple
        return odds
