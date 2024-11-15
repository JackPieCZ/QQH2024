import numpy as np
import pandas as pd


class Model:
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

        optimal_bet = bankroll * optimal_fraction * fraction

        optimal_bet = max(0, optimal_bet)  # nesázet, pokud je Kelly záporný

        return optimal_bet

    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: tuple[pd.DataFrame, pd.DataFrame]):
        #  print(f"{summary=}")
        #  print(f"{opps=}")
        #  print(f"{inc=}")
        # input("Press Enter to continue...")
        min_bet = summary.iloc[0]["Min_bet"]
        bankroll = summary.iloc[0]["Bankroll"]

        kelly_fraction = 1
        prob_home = 0.6  # zanalyzovat
        prob_away = 0.4  # zanalyzovat

        # if N > 0:

        #     bets = np.zeros((N, 2))

        #     for _, row in opps.iterrows():
        #         oddsH = row['OddsH']
        #         oddsA = row['OddsA']

        #         #  print("index: ", opps.index)
        #         bet_home = self.kelly_criterion(prob_home, oddsH, bankroll, kelly_fraction)
        #         bet_away = self.kelly_criterion(prob_away, oddsA, bankroll, kelly_fraction)

        #         bet_home = np.where(bet_home > 0, np.maximum(min_bet, bet_home), 0)
        #         bet_away = np.where(bet_away > 0, np.maximum(min_bet, bet_away), 0)

        #         bets = pd.DataFrame(data={"BetH": bet_home, "BetA": bet_away}, index=opps.index)
        #         return bets

        bets = []
        min_bet = summary['Min_bet'].iloc[0]
        max_bet = summary['Max_bet'].iloc[0]
        for _, row in opps.iterrows():
            # Calculate features for home and away teams based on historical data
            prob_home = 0.6
            prob_away = 0.4
            oddsH = row['OddsH']
            oddsA = row['OddsA']

            # Calculate Kelly bet sizes
            bet_home = self.kelly_criterion(prob_home, oddsH, bankroll, kelly_fraction)
            bet_away = self.kelly_criterion(prob_away, oddsA, bankroll, kelly_fraction)

            # # Bet sizes should be between min and max bets and be non-negative
            betH = max(min(bet_home * bankroll, max_bet), min_bet) if bet_home > 0 else 0
            betA = max(min(bet_away * bankroll, max_bet), min_bet) if bet_away > 0 else 0

            # Append the bets to the DataFrame
            bets.append([row.index, betH, betA])

        # Convert list of bets to DataFrame and return
        bets_df = pd.DataFrame(bets, columns=['ID', 'BetH', 'BetA'])
        return bets_df

        """""""""
        bets = []
        for idx, row in opps.iterrows():
            ...
            # Append the bets to the DataFrame
            bets.append([row['ID'], betH, betA])

        # Convert list of bets to DataFrame and return
        bets_df = pd.DataFrame(bets, columns=['ID', 'BetH', 'BetA'])
        
        """


"""
conda create -n qqh python=3.12.4 -y
conda activate qqh
conda install -c conda-forge -c pytorch -c pyg numpy pandas py-xgboost-cpu scikit-learn scipy statsmodels pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 cpuonly pyg -y
"""
