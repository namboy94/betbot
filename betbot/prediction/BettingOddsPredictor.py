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
from betbot.data.OddsPortal import OddsPortal
from betbot.prediction.Predictor import Predictor


class BettingOddsPredictor(Predictor):
    """
    Class that determinalistically predicts matches based on Tipico quotes
    """

    @classmethod
    def name(cls) -> str:
        """
        :return: The name of the predictor
        """
        return "betting-odds"

    def predict(self, matches: List[Match]) -> List[Bet]:
        """
        Performs the prediction
        :param matches: The matches to predict
        :return: The predictions as Bet objects
        """
        bets = []
        oddsportal = OddsPortal()
        odds = oddsportal.get_odds(self.league)
        for match in matches:
            match_tuple = (match.home_team, match.away_team)
            match_odds = odds.get(match_tuple)
            if match_odds is None:
                continue
            home_score, away_score = self.generate_scores(*match_odds)
            bets.append(Bet(match, home_score, away_score))
        return bets

    # noinspection PyMethodMayBeStatic
    def generate_scores(
            self,
            home_odds: float,
            draw_odds: float,
            away_odds: float
    ) -> Tuple[int, int]:
        """
        Generates a scoreline based on the quote data
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

        return home_score, away_score
