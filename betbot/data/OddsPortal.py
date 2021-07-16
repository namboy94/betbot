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

from selenium.webdriver import Firefox
from selenium.webdriver.firefox.options import Options
from typing import Dict, Tuple


class OddsPortal:
    """
    Class that handles retrieving bet data using oddsportal.com
    """

    TEAM_MAP = {
        "Nurnberg": "FCN",
        "Mainz": "M05",
        "Bayer Leverkusen": "B04",
        "Dortmund": "BVB",
        "B. Monchengladbach": "BMG",
        "Eintracht Frankfurt": "SGE",
        "Augsburg": "FCA",
        "Bayern Munich": "FCB",
        "Schalke": "S04",
        "Dusseldorf": "F95",
        "Hannover": "H96",
        "Hertha Berlin": "BSC",
        "RB Leipzig": "RBL",
        "Freiburg": "SCF",
        "Hoffenheim": "TSG",
        "Stuttgart": "VFB",
        "Wolfsburg": "WOB",
        "Werder Bremen": "BRE",
        "Union Berlin": "FCU",
        "Paderborn": "SCP",
        "FC Koln": "KOE",
        "Arminia Bielefeld": "DSC",
        "Greuther Furth": "SGF",
        "Bochum": "BOC",
        "Heidenheim": "HDH",
        "Aue": "AUE",
        "Hansa Rostock": "FCH",
        "Ingolstadt": "FCI",
        "St. Pauli": "STP",
        "Hamburger SV": "HSV",
        "Holstein Kiel": "KIE",
        "Regensburg": "SSV",
        "Karlsruher": "KSC",
        "SG Dynamo Dresden": "SGD",
        "Darmstadt": "D98",
        "Sandhausen": "SVS",
    }

    def __init__(self):
        """
        Initializes the oddsportal scraper
        """
        options = Options()
        options.headless = True
        self.driver = Firefox(options=options)

    def get_odds(self, league: str) \
            -> Dict[Tuple[str, str], Tuple[float, float, float]]:
        """
        Retrieves the odds for upcoming matches from oddsportal.com
        :param league: The league to search for
        :return: Odds for the matches (home, draw, odds)
        """
        matches: Dict[Tuple[str, str], Tuple[float, float, float]] = {}
        base_url = "https://www.oddsportal.com/soccer/"
        endpoint = {
            "bl1": "germany/bundesliga/",
            "bl2": "germany/2-bundesliga/"
        }.get(league)
        if endpoint is None:
            return matches

        self.driver.get(base_url + endpoint)
        table = self.driver.find_element_by_id("tournamentTable")
        for row in table.find_elements_by_tag_name("tr"):
            participant_tag = \
                row.find_elements_by_class_name("table-participant")
            if len(participant_tag) == 0:
                continue

            home_team, away_team = participant_tag[0].text.split(" - ")
            home_abbrv = self.TEAM_MAP.get(home_team)
            away_abbrv = self.TEAM_MAP.get(away_team)
            if home_abbrv is None or away_abbrv is None:
                continue
            match_tuple = (home_abbrv, away_abbrv)
            odds = row.find_elements_by_class_name("odds-nowrp")
            try:
                matches[match_tuple] = (
                    float(odds[0].text),
                    float(odds[1].text),
                    float(odds[2].text)
                )
            except ValueError:
                continue

        return matches
