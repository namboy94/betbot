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

import csv
import json
import requests
from typing import Tuple, List, Dict
from betbot.neural.data.openligadb.Team import Team
from betbot.neural.data.openligadb.Match import Match
from betbot.neural.data.openligadb.TableEntry import TableEntry
from betbot.neural.data.openligadb.InputVector import InputVector
from betbot.neural.data.openligadb.OutputVector import OutputVector


def load_data(league: str, season: int) \
        -> Tuple[Dict[int, Team], Dict[int, List[Match]]]:
    """
    Loads match and team data for a season
    :param league: The league
    :param season: The season
    :return: The processed data
    """
    base_url = "https://www.openligadb.de/api/{}/{}/{}"
    team_data = json.loads(requests.get(
        base_url.format("getavailableteams", league, season)
    ).text)
    match_data = json.loads(requests.get(
        base_url.format("getmatchdata", league, season)
    ).text)

    _teams = [Team(team_json) for team_json in team_data]
    teams = {team.id: team for team in _teams}

    _matches = [Match(match_json, teams) for match_json in match_data]
    matches = {}
    for match in _matches:
        if match.matchday not in matches:
            matches[match.matchday] = []
        matches[match.matchday].append(match)

    return teams, matches


def calculate_tables(matches: Dict[int, List[Match]]) \
        -> Dict[int, Dict[int, TableEntry]]:
    """
    Calculates league tables for every matchday in a season
    :param matches: The matches in the season
    :return: The tables, one for each matchday
    """
    tables = {}
    for matchday, matchday_matches in matches.items():
        tables[matchday] = {}
        previous_table = tables.get(matchday - 1)

        for match in matchday_matches:

            if match.home_score > match.away_score:
                home_points, away_points = 3, 0
            elif match.home_score < match.away_score:
                home_points, away_points = 0, 3
            else:
                home_points, away_points = 1, 1

            home_entry = TableEntry(
                matchday,
                match.home_team,
                home_points,
                match.home_score,
                match.away_score
            )
            away_entry = TableEntry(
                matchday,
                match.away_team,
                away_points,
                match.away_score,
                match.home_score
            )

            if previous_table is not None:
                home_entry.merge(previous_table[match.home_team.id])
                away_entry.merge(previous_table[match.away_team.id])

            tables[matchday][match.home_team.id] = home_entry
            tables[matchday][match.away_team.id] = away_entry

    return tables


def generate_training_data() -> List[Tuple[InputVector, OutputVector]]:
    """
    Creates training data for multiple seasons and leagues
    :return: The training data as a list of input/output vector tuples
    """
    vectors = []

    leagues = ["bl1", "bl2", "bl3"]
    seasons = list(range(2010, 2020))

    all_teams = {}
    all_matches = {}
    all_tables = {}

    print("Loading data...")
    for league in leagues:
        all_teams[league] = {}
        all_matches[league] = {}
        all_tables[league] = {}
        for season in seasons:
            print(f"{league}/{season}")
            team_data, match_data = load_data(league, season)
            table_data = calculate_tables(match_data)
            all_teams[league][season] = team_data
            all_matches[league][season] = match_data
            all_tables[league][season] = table_data
    print("Done!")

    for league in leagues:
        for season in seasons:
            match_data = all_matches[league][season]
            table_data = all_tables[league][season]
            previous_season_tables = all_tables[league].get(season - 1)
            last_matchday = max(table_data.keys())

            for matchday, matches in match_data.items():

                previous_table = table_data.get(matchday - 1)
                previous_table5 = table_data.get(max(1, matchday - 6))

                if matchday < 6 and previous_season_tables is None:
                    continue
                elif matchday < 6:
                    if matchday == 1:
                        previous_table = previous_season_tables[last_matchday]
                    target_5 = last_matchday - (6 - matchday)
                    previous_table5 = previous_season_tables[target_5]

                for match in matches:
                    try:
                        home_team_entry = previous_table[match.home_team.id]
                        away_team_entry = previous_table[match.away_team.id]
                        home_team_entry5 = previous_table5[match.home_team.id]
                        away_team_entry5 = previous_table5[match.away_team.id]
                    except KeyError:
                        continue

                    if matchday < 6:
                        home_previous_season_final = previous_season_tables[
                            last_matchday].get(match.home_team.id)
                        away_previous_season_final = previous_season_tables[
                            last_matchday].get(match.away_team.id)
                        if home_previous_season_final is None \
                                or away_previous_season_final is None:
                            continue
                    else:
                        home_previous_season_final = None
                        away_previous_season_final = None

                    input_vector = InputVector(
                        home_team_entry,
                        away_team_entry,
                        home_team_entry5,
                        away_team_entry5,
                        home_previous_season_final,
                        away_previous_season_final
                    )
                    output_vector = OutputVector(
                        match.home_score,
                        match.away_score
                    )
                    vectors.append((input_vector, output_vector))

    return vectors


def generate_training_csv(csv_path: str):
    """
    Generates a CSV file with training data for a neural network
    :param csv_path: The path to the CSV file
    :return: None
    """
    vectors = generate_training_data()
    training_data = []

    for input_vector, output_vector in vectors:
        training_data.append(input_vector.vector + output_vector.vector)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(training_data)


def load_csv(csv_file: str) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Loads training data from a CSV file
    :param csv_file: The CSV file
    :return: The Input vectors and expected Output Vectors
    """
    inputs = []
    outputs = []
    with open(csv_file, "r", newline="") as f:
        for line in csv.reader(f):
            inputs.append([float(x) for x in line[0:-2]])
            outputs.append([float(x) for x in line[-2:]])

    return inputs, outputs
