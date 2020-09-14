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

from typing import Dict, Any


class Team:
    """
    Class that models an OpenLigaDB Team
    """

    def __init__(self, team_json: Dict[str, Any]):
        """
        Initializes the team object
        :param team_json: The OpenLigaDB JSON for the team
        """
        self.id = team_json["TeamId"]
        self.team_name = team_json["TeamName"]
        self.abbreviation = {
            "1. FC Nürnberg": "FCN",
            "1. FSV Mainz 05": "M05",
            "Bayer Leverkusen": "B04",
            "Borussia Dortmund": "BVB",
            "Borussia Mönchengladbach": "BMG",
            "Eintracht Frankfurt": "SGE",
            "FC Augsburg": "FCA",
            "FC Bayern": "FCB",
            "FC Schalke 04": "S04",
            "Fortuna Düsseldorf": "F95",
            "Hannover 96": "H96",
            "Hertha BSC": "BSC",
            "RB Leipzig": "RBL",
            "SC Freiburg": "SCF",
            "TSG 1899 Hoffenheim": "TSG",
            "VfB Stuttgart": "VFB",
            "VfL Wolfsburg": "VFL",
            "Werder Bremen": "SVW",
            "1. FC Union Berlin": "SCU",
            "SC Paderborn 07": "PAD",
            "1. FC Köln": "FCK",
            "Arminia Bielefeld": "BIE"
        }.get(self.team_name, "N/A")
