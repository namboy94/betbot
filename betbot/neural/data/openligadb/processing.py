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
from betbot.neural.data.openligadb.HistoryRecord import HistoryRecord


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

    _matches = [
        Match(league, season, match_json, teams)
        for match_json in match_data
    ]
    matches: Dict[int, List[Match]] = {}
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
    tables: Dict[int, Dict[int, TableEntry]] = {}
    for matchday, matchday_matches in matches.items():
        tables[matchday] = {}
        previous_table = tables.get(matchday - 1)

        for match in matchday_matches:
            home_entry = TableEntry.from_match(match, match.home_team)
            away_entry = TableEntry.from_match(match, match.away_team)
            if previous_table is not None:
                home_entry.merge(previous_table[match.home_team.id])
                away_entry.merge(previous_table[match.away_team.id])
            tables[matchday][match.home_team.id] = home_entry
            tables[matchday][match.away_team.id] = away_entry

    return tables


def calculate_history(match_data: Dict[int, Dict[int, List[Match]]]) \
        -> Dict[int, HistoryRecord]:
    """
    Calculates historical table data
    :param match_data: The match data for all seasons to include
    :return: The history for each team
    """
    all_tables = {}
    for season, matches in match_data.items():
        all_tables[season] = calculate_tables(matches)

    history = {}
    for tables in all_tables.values():
        for matchday_tables in tables.values():
            for team_id, entry in matchday_tables.items():
                if team_id not in history:
                    history[team_id] = HistoryRecord(entry.team)
                history[team_id].add_table_entry(entry)
    return history


def generate_training_data(league_seasons: List[Tuple[str, int]]) \
        -> List[Tuple[InputVector, OutputVector]]:
    """
    Creates training data for multiple seasons and leagues
    :param league_seasons: The leagues+seasons for which to generate
                           training data
    :return: The training data as a list of input/output vector tuples
    """
    vectors = []

    all_teams: Dict[str, Dict[int, Dict[int, Team]]] = {}
    all_matches: Dict[str, Dict[int, Dict[int, List[Match]]]] = {}
    all_tables: Dict[str, Dict[int, Dict[int, Dict[int, TableEntry]]]] = {}
    histories = {}

    leagues: Dict[str, List[int]] = {}
    for league, season in league_seasons:
        if league not in leagues:
            leagues[league] = []
        leagues[league].append(season)
        leagues[league].sort()

    print("Loading data...")
    for league, seasons in leagues.items():
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
        histories[league] = calculate_history(all_matches[league])
    print("Done!")

    for league, seasons in leagues.items():
        for season in seasons:
            match_data = all_matches[league][season]
            for matchday, matches in match_data.items():

                if matchday < 6 and season == seasons[0]:
                    # Skip matches where we don't have enough historical data
                    continue

                for match in matches:

                    if not match.finished:  # Exclude unfinished matches
                        continue

                    home_team_history = histories[league][match.home_team.id]
                    away_team_history = histories[league][match.away_team.id]

                    input_vector = InputVector(
                        season,
                        matchday,
                        match.finished,
                        home_team_history,
                        away_team_history
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
    league_seasons = []
    for bl in ["bl1", "bl2", "bl3"]:
        for season in range(2010, 2020):
            league_seasons.append((bl, season))
    vectors = generate_training_data(league_seasons)
    training_data = []

    for input_vector, output_vector in vectors:
        training_data.append(input_vector.vector + output_vector.vector)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(training_data)


def load_csv(csv_file: str) -> List[Tuple[List[float], List[float]]]:
    """
    Loads training data from a CSV file
    :param csv_file: The CSV file
    :return: The Input vectors and expected Output Vectors
    """
    data = []
    with open(csv_file, "r", newline="") as f:
        for line in csv.reader(f):
            data.append(line)

    vectors = []
    for data_point in data:
        input_vector = [float(x) for x in data_point[0:-2]]
        output_vector = [float(x) for x in data_point[-2:]]
        vectors.append((input_vector, output_vector))

    return vectors
