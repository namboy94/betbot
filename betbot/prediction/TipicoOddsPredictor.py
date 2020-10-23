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

from math import sqrt
from typing import List, Tuple
from betbot.api.Bet import Bet
from betbot.api.Match import Match
from betbot.prediction.Predictor import Predictor
from betbot.neural.data.FootballDataFetcher import FootballDataFetcher
from betbot.neural.data.enums import Bookmakers
from betbot.neural.data.names import oddsportal_to_football_data, \
    abbreviations_to_football_data


class TipicoOddsPredictor(Predictor):
    """
    Class that determinalistically predicts matches based on Tipico quotes
    """

    @classmethod
    def name(cls) -> str:
        """
        :return: The name of the predictor
        """
        return "tipico-odds"

    def predict(self, matches: List[Match]) -> List[Bet]:
        """
        Performs the prediction
        :param matches: The matches to predict
        :return: The predictions as Bet objects
        """
        tipico = self.load_tipico_data()
        bets = []

        for home_team, away_team, home_odds, draw_odds, away_odds in tipico:
            match_found = False
            for match in matches:
                if home_team == match.home_team \
                        and away_team == match.away_team:
                    bet = self.generate_bet(
                        match.id, home_odds, draw_odds, away_odds
                    )
                    bets.append(bet)
                    match_found = True

            if not match_found:
                self.logger.warning(f"Failed to match {home_team}:{away_team}")

        return bets

    # noinspection PyMethodMayBeStatic
    def load_tipico_data(self) -> List[Tuple[str, str, float, float, float]]:
        """
        Loads tipico quote data for the 2020/21 bundesliga season
        :return: The quote data as a list of tuples of the form:
                    home team name, away team name,
                    home win odds, draw odds, away win odds
        """
        abbrev_map = {y: x for x, y in abbreviations_to_football_data.items()}

        matches = FootballDataFetcher().load_oddsportal_matches()
        data = []

        for match in matches:
            odds = match.bet_odds[Bookmakers.BWIN]
            home_team = abbrev_map[oddsportal_to_football_data.get(
                match.home_team, match.home_team
            )]
            away_team = abbrev_map[oddsportal_to_football_data.get(
                match.away_team, match.away_team
            )]
            data.append(
                (home_team, away_team, odds[0], odds[1], odds[2])
            )
        return data

    # noinspection PyMethodMayBeStatic
    def generate_bet(
            self,
            match_id: int,
            home_odds: float,
            draw_odds: float,
            away_odds: float
    ) -> Bet:
        """
        Generates a Bet based on the quote data
        :param match_id: The ID of the match to vote on
        :param home_odds: The odds that the home team wins
        :param draw_odds: The odds that the match ends in a draw
        :param away_odds: The odds that the away team wins
        :return: The generated bet
        """
        winner = int(sqrt(abs(home_odds - away_odds) * 2))
        loser = int(min(home_odds, away_odds) - 1)

        if home_odds < away_odds:
            home_score, away_score = winner, loser
        else:
            home_score, away_score = loser, winner

        if draw_odds > home_odds and draw_odds > away_odds:
            home_score += 1
            away_score += 1

        return Bet(match_id, home_score, away_score)
