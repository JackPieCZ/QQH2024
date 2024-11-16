import pandas as pd
import numpy as np


def calculate_win_probs_frantisek(summary, opp, games_inc, players_inc):
    """Calculates win probabilities for home and away team.

        Args:
            opp (pd.Dataframe): Betting opportunities with columns ['Season', 'Date', 'HID', 'AID', 'N', 'POFF', 'OddsH', 'OddsA', 'BetH', 'BetA'].

        Returns:
            tuple(float, float): Probability of home team winning, probability of away team winning.
        """
    current_season = opp['Season']
    match_date = opp['Date']
    home_ID = opp['HID']
    away_ID = opp['AID']
    neutral_ground = opp['N']
    playoff_game = opp['POFF']
    oddsH = opp['OddsH']
    oddsA = opp['OddsA']
    return 0.5, 0.5


def calculate_win_probs_frantisek2(summary, opp, games_inc, players_inc):
    """Calculates win probabilities for home and away team.

        Args:
            opp (pd.Dataframe): Betting opportunities with columns ['Season', 'Date', 'HID', 'AID', 'N', 'POFF', 'OddsH', 'OddsA', 'BetH', 'BetA'].

        Returns:
            tuple(float, float): Probability of home team winning, probability of away team winning.
        """
    current_season = opp['Season']
    match_date = opp['Date']
    home_ID = opp['HID']
    away_ID = opp['AID']
    neutral_ground = opp['N']
    playoff_game = opp['POFF']
    oddsH = opp['OddsH']
    oddsA = opp['OddsA']
    return 0.5, 0.5
