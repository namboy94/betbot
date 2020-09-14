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
import requests
from bs4 import BeautifulSoup
from typing import List, Tuple, Dict
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense
from betbot.neural.keras.BetPredictorTrainer import BetPredictorTrainer


class BetOddsTrainer(BetPredictorTrainer):
    """
    """

    def load_training_data(self, force_refresh: bool) \
            -> List[Tuple[List[float], List[float]]]:
        """
        Loads data for training the model
        :param force_refresh: Forces a refresh of data
        :return: The training data, with the input and output vectors separate
        """
        vectors = []

        base_url = "https://www.football-data.co.uk/"
        start_url = base_url + "germanym.php"
        resp = requests.get(start_url)
        soup = BeautifulSoup(resp.text, "html.parser")
        csv_urls: Dict[str, Dict[int, str]] = {}

        for a in soup.select("a"):
            href = a["href"]
            if href.endswith("csv"):
                csv_url = os.path.join(base_url, href)
                season_string = href.split("/")[1]
                century = "19" if season_string.startswith("9") else "20"
                season = int(century + season_string[0:2])
                if season < 2005:
                    continue
                league = "bl1" if "D1.csv" in href else "bl2"

                if league not in csv_urls:
                    csv_urls[league] = {}
                csv_urls[league][season] = csv_url

        for league, season_data in csv_urls.items():
            for season, csv_url in season_data.items():
                print(f"{league}/{season}")
                csv_file_path = \
                    os.path.join(self.model_dir, f"{league}-{season}.csv")
                if os.path.isfile(csv_file_path):
                    with open(csv_file_path, "r") as f:
                        data = [x for x in csv.reader(f)]
                        keys = data.pop(0)
                else:
                    data = [
                        x.split(",")
                        for x in requests.get(csv_url).text.split("\r\n")
                    ]
                    keys = data.pop(0)
                    data.pop()
                    for i in range(len(data)):
                        for j in range(len(data[i])):
                            if data[i][j] == "":
                                data[i][j] = "0"

                    with open(csv_file_path, "w", newline="") as f:
                        csv.writer(f).writerows([keys] + data)

                data_dicts = [
                    {keys[j]: data[i][j] for j in range(len(keys))}
                    for i in range(len(data))
                ]

                for match in data_dicts:

                    input_vector = []
                    keys = []
                    for bookkeeper in [
                        "B365",
                        "BW",
                        "IW",
                        "VC",
                        "WH"
                    ]:
                        for category in ["H", "A", "D"]:
                            key = bookkeeper + category
                            keys.append(key)
                            input_vector.append(float(match[key]))

                    keys += [

                    ]

                    vectors.append((
                        input_vector,
                        [float(match["FTHG"]), float(match["FTAG"])]
                    ))

        return vectors

    def _define_model(self) -> Model:
        """
        Specifies the model of the neural network
        :return: The model
        """
        model = Sequential()
        model.add(Flatten(input_shape=(15,)))
        model.add(Dense(32, activation="sigmoid"))
        model.add(Dense(16, activation="sigmoid"))
        model.add(Dense(2, activation="relu"))
        return model

    def _compile_model(self, model: Model):
        """
        Compiles the keras model
        :param model: The model to compile
        :return: None
        """
        model.compile(loss="mae", optimizer="sgd")
