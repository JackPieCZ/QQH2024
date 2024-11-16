import numpy as np
import pandas as pd

from arbitrage import calculate_arbitrage_betting
from database import HistoricalDatabase
from strats.win_prob_lenka import calculate_win_probs_lenka, calculate_win_probs_lenka2
from strats.win_prob_frantisek import calculate_win_probs_frantisek, calculate_win_probs_frantisek2
from strats.win_prob_sviga import calculate_win_probs_sviga, calculate_win_probs_sviga2
from strats.win_prob_kuba import calculate_win_probs_kuba, calculate_win_probs_kuba2


class Model:
    def __init__(self):
        self.db = HistoricalDatabase()

    def kelly_criterion(self, probability, odds, bankroll, fraction):
        """
        Vypočítá optimální výši sázky pomocí Kellyho kritéria.

        :param probability: odhadovaná pravděpodobnost výhry (0 až 1).
        :param odds: kurz
        :param bankroll: dostupný kapitál
        :param fraction: frakční Kelly (např. 0.5 pro poloviční Kelly).
        :return: doporučená výše sázky.
        """
        q = 1 - probability
        b = odds - 1  # zisk

        optimal_fraction = probability - (q / b)
        # nesázet, pokud je Kelly záporný
        optimal_fraction = max(0, optimal_fraction)
        optimal_bet = bankroll * optimal_fraction * fraction

        return float(optimal_bet)

    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: tuple[pd.DataFrame, pd.DataFrame]):
        games_inc, players_inc = inc
        self.db.add_incremental_data(games_inc, players_inc)
        # print(f"{summary=}")
        # print(f"{opps=}")
        # print(f"{games_inc=}")
        # print(f"{players_inc=}")
        # input("Press Enter to continue...")
        min_bet = summary.iloc[0]["Min_bet"]
        max_bet = summary.iloc[0]["Max_bet"]

        kelly_fraction = 0.5
        # fraction of budget we are willing to spend today
        budget_fraction = 0.1
        todays_budget = summary.iloc[0]["Bankroll"] * budget_fraction

        # only iterate over opps with the current date while keeping the original index
        todays_date = summary.iloc[0]["Date"]
        assert opps[opps["Date"] < todays_date].empty, \
            "There are opps before today's date, which should never happen"
        todays_opps = opps[opps["Date"] == todays_date]

        # Add columns for new bet sizes and win probabilities
        # Temporarily disable SettingWithCopyWarning
        pd.options.mode.chained_assignment = None
        todays_opps['newBetH'] = 0.0
        todays_opps['newBetA'] = 0.0
        todays_opps['ProbH'] = 0.0
        todays_opps['ProbA'] = 0.0

        # upravte si čí funkci vyhodnocování pravděpodobností výhry chcete použít
        calculate_win_probs_fn = calculate_win_probs_kuba

        # Calculate win probabilities for each opportunity
        for opp_idx, opp in todays_opps.iterrows():
            betH = opp['BetH']
            betA = opp['BetA']
            assert betH == 0 and betA == 0, "Both bets should be zero at the beginning"

            prob_home, prob_away = calculate_win_probs_fn(summary, opp, games_inc, players_inc, self.db)
            assert 0 <= prob_home <= 1 and 0 <= prob_away <= 1, "Probabilities should be between 0 and 1"
            assert abs(1 - (prob_home + prob_away)) < 1e-9, "Probabilities should sum up to 1"
            
            todays_opps.loc[todays_opps.index == opp_idx, 'ProbH'] = prob_home
            todays_opps.loc[todays_opps.index == opp_idx, 'ProbA'] = prob_away

        # Sort win probabilities in descending order and keep track of original indices
        # Create a new column with the maximum probability between home and away
        todays_opps['MaxProb'] = todays_opps[['ProbH', 'ProbA']].max(axis=1)
        sorted_win_probs_opps = todays_opps.sort_values(by='MaxProb', ascending=False).reset_index()

        # Place bets based on Kelly criterion starting with the highest win probabilities first
        for _, row in sorted_win_probs_opps.iterrows():
            opp_idx = row['index']
            prob_home = row['ProbH']
            prob_away = row['ProbA']
            opp = todays_opps.loc[opp_idx]

            oddsH = opp['OddsH']
            oddsA = opp['OddsA']

            # Check if there is an arbitrage betting opportunity
            if calculate_arbitrage_betting(oddsH, oddsA):
                print(f"Arbitrage opportunity detected for opp {opp_idx}, nice!")
                # Take advantage of the arbitrage
                bet_home = todays_budget / 2
                bet_away = todays_budget / 2
            else:
                # Calculate Kelly bet sizes
                bet_home = self.kelly_criterion(prob_home, oddsH, todays_budget, kelly_fraction)
                bet_away = self.kelly_criterion(prob_away, oddsA, todays_budget, kelly_fraction)

                assert bet_home == 0 or bet_away == 0, \
                    "Only one bet should be placed, if there is no opportunity to arbitrage"

            # Bet sizes should be between min and max bets and be non-negative
            betH = max(min(bet_home, max_bet),min_bet) if bet_home > min_bet else 0
            betA = max(min(bet_away, max_bet),min_bet) if bet_away > min_bet else 0

            # Update the bets DataFrame with calculated bet sizes
            todays_opps.loc[todays_opps.index == opp_idx, 'newBetH'] = betH
            todays_opps.loc[todays_opps.index == opp_idx, 'newBetA'] = betA
            todays_budget -= betH + betA

            # Stop if we run out of budget
            if todays_budget <= 0:
                break

        bets = todays_opps[['newBetH', 'newBetA']]
        bets.rename(columns={'newBetH': 'BetH','newBetA': 'BetA'}, inplace=True)
        return bets


"""
conda create -n qqh python=3.12.4 -y
conda activate qqh
conda install -c conda-forge -c pytorch -c pyg numpy pandas py-xgboost-cpu scikit-learn scipy statsmodels pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 cpuonly pyg -y
"""
