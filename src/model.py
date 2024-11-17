import numpy as np # noqa
import pandas as pd

from arbitrage import calculate_arbitrage_betting
from database import HistoricalDatabase
from strats.win_prob_lenka import calculate_win_probs_lenka, calculate_win_probs_lenka2  # noqa
from strats.win_prob_frantisek import calculate_win_probs_frantisek, calculate_win_probs_frantisek2  # noqa
from strats.win_prob_sviga import calculate_win_probs_sviga, calculate_win_probs_sviga2  # noqa
from strats.win_prob_kuba import calculate_win_probs_kuba, calculate_win_probs_kuba2  # noqa


class Model:
    def __init__(self):
        self.db = HistoricalDatabase()
        self.yesterdays_games = None
        self.yesterdays_bets = None
        self.money_spent_yesterday = 0
        self.bankroll_after_bets = 0

    def kelly_criterion(self, probability, odds):
        """
        Vypočítá optimální výši sázky pomocí Kellyho kritéria.

        :param probability: odhadovaná pravděpodobnost výhry (0 až 1).
        :param odds: kurz
        # :param bankroll: dostupný kapitál
        # :param fraction: frakční Kelly (např. 0.5 pro poloviční Kelly).
        :return: doporučený zlomek sázky.
        """
        q = 1 - probability
        b = odds - 1  # zisk

        optimal_fraction = probability - (q / b)
        # nesázet, pokud je Kelly záporný
        optimal_fraction = max(0, optimal_fraction)
        return float(optimal_fraction)

    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: tuple[pd.DataFrame, pd.DataFrame]):
        min_bet = summary.iloc[0]["Min_bet"]
        max_bet = summary.iloc[0]["Max_bet"]
        bankroll = summary.iloc[0]["Bankroll"]
        todays_date = summary.iloc[0]["Date"]
        # only iterate over opps with the current date while keeping the original index
        assert opps[opps["Date"] < todays_date].empty, "There are opps before today's date, which should never happen"
        todays_opps = opps[opps["Date"] == todays_date]
        games_inc, players_inc = inc
        self.yesterdays_games = games_inc

        # upravte si čí funkci vyhodnocování pravděpodobností výhry chcete použít
        calculate_win_probs_fn = calculate_win_probs_kuba
        kelly_fraction = 0.2
        # fraction of budget we are willing to spend today
        budget_fraction = 0.1
        use_kelly = False
        todays_budget = bankroll * budget_fraction
        non_kelly_bet_amount = min_bet * 2

        # Evaluate yesterday's predictions
        if self.yesterdays_bets is not None:
            # Calculate accuracy of yesterday's predictions
            correct_predictions = 0
            correct_bets = 0
            correct_bookmaker_bets = 0
            total_predictions = len(self.yesterdays_games)
            num_bets = ((self.yesterdays_bets["newBetH"] > 0) | (self.yesterdays_bets["newBetA"] > 0)).sum()

            for idx, game in self.yesterdays_games.iterrows():
                # Get corresponding prediction
                prediction = self.yesterdays_bets.loc[idx]
                # Export prediction with index to CSV
                prediction_df = pd.DataFrame(prediction).T
                game_df = pd.DataFrame(game).T

                # Determine which team was predicted to win
                assert prediction["ProbH"] + prediction["ProbA"] != 0, "Probabilities should not sum up to zero"
                predicted_home_win = prediction["ProbH"] > prediction["ProbA"]

                if prediction["newBetH"] + prediction["newBetA"] != 0:
                    betted_home_win = prediction["newBetH"] > prediction["newBetA"]
                    if (betted_home_win and game["H"] == 1) or (not betted_home_win and game["A"] == 1):
                        correct_bets += 1

                bookmaker_predicted_home_win = game["OddsH"] < game["OddsA"]
                if (bookmaker_predicted_home_win and game["H"] == 1) or (not bookmaker_predicted_home_win and game["A"] == 1):
                    correct_bookmaker_bets += 1

                # Check if prediction matches actual result
                if (predicted_home_win and game["H"] == 1) or (not predicted_home_win and game["A"] == 1):
                    correct_predictions += 1
            pred_accuracy = correct_predictions / total_predictions if total_predictions > 0 else None
            bets_accuracy = correct_bets / num_bets if num_bets > 0 else None
            bookmaker_accuracy = correct_bookmaker_bets / total_predictions if total_predictions > 0 else None
            print(f"Yesterday's prediction accuracy: {pred_accuracy} ({correct_predictions}/{total_predictions})")
            print(f"Yesterday's betting accuracy: {bets_accuracy} ({correct_bets}/{num_bets})")
            print(f"Yesterday's bookmaker's accuracy: {bookmaker_accuracy} ({correct_bookmaker_bets}/{total_predictions})")
            print(f"Money - spent: {self.money_spent_yesterday:.2f}$, gained: {bankroll - self.bankroll_after_bets:.2f}$")
            # input("Press Enter to continue...")

        self.db.add_incremental_data(games_inc, players_inc)

        # Temporarily disable SettingWithCopyWarning
        pd.options.mode.chained_assignment = None
        # Add columns for new bet sizes and win probabilities
        opps["newBetH"] = 0.0
        opps["newBetA"] = 0.0
        opps["ProbH"] = 0.0
        opps["ProbA"] = 0.0
        opps["KellyH"] = 0.0
        opps["KellyA"] = 0.0

        # Calculate win probabilities for each opportunity
        for opp_idx, opp in todays_opps.iterrows():
            betH = opp["BetH"]
            betA = opp["BetA"]
            oddsH = opp["OddsH"]
            oddsA = opp["OddsA"]
            assert betH == 0 and betA == 0, "Both bets should be zero at the beginning"

            prob_home, prob_away = calculate_win_probs_fn(summary, opp, games_inc, players_inc, self.db)
            assert isinstance(prob_home, (int, float)) and isinstance(
                prob_away, (int, float)
            ), f"Win probabilities should be numbers, currently they are of type {type(prob_home)} and {type(prob_away)}"
            assert 0 <= prob_home <= 1 and 0 <= prob_away <= 1, f"Probabilities should be between 0 and 1, currently they are {prob_home} and {prob_away}"
            assert abs(1 - (prob_home + prob_away)) < 1e-9, f"Probabilities should sum up to 1, currently they sum up to {prob_home + prob_away}"

            opps.loc[opps.index == opp_idx, "ProbH"] = prob_home
            opps.loc[opps.index == opp_idx, "ProbA"] = prob_away

            # Check if there is an arbitrage betting opportunity
            if calculate_arbitrage_betting(oddsH, oddsA):
                print(f"Arbitrage opportunity detected for opp {opp_idx}, nice!")
                # Take advantage of the arbitrage
                kellyH = 0.5
                kellyA = 0.5
            else:
                # Calculate Kelly bet sizes
                kellyH = self.kelly_criterion(prob_home, oddsH)
                kellyA = self.kelly_criterion(prob_away, oddsA)
                assert kellyH == 0 or kellyA == 0, "Only one kelly should be nonzero, if there is no opportunity to arbitrage"

            opps.loc[opps.index == opp_idx, "KellyH"] = kellyH
            opps.loc[opps.index == opp_idx, "KellyA"] = kellyA

        if use_kelly:
            # Sort Kelly criterion in descending order and keep track of original indices
            # Create a new column with the maximum kelly of home and away
            opps["MaxKelly"] = opps[["KellyH", "KellyA"]].max(axis=1)
            sorted_win_probs_opps = opps.sort_values(by="MaxKelly", ascending=False)

            # Place bets based on Kelly criterion starting with the highest one
            for opp_idx, row in sorted_win_probs_opps.iterrows():
                # opp_idx = row["index"]
                kellyH = row["KellyH"]
                kellyA = row["KellyA"]
                probH = row["ProbH"]
                probA = row["ProbA"]

                # # New logic: Only bet on the outcome with the higher probability
                # if probH >= probA:
                #     kellyA = 0  # Set the Kelly fraction for away team to zero if home is predicted higher
                # else:
                #     kellyH = 0  # Set the Kelly fraction for home team to zero if away is predicted higher

                # Skip if both Kelly fractions are zero
                if kellyH == 0 and kellyA == 0:
                    continue

                bet_home = kellyH * todays_budget * kelly_fraction
                bet_away = kellyA * todays_budget * kelly_fraction

                # Bet sizes should be between min and max bets and be non-negative
                betH = max(min(bet_home, max_bet), min_bet) if bet_home >= min_bet else 0
                betA = max(min(bet_away, max_bet), min_bet) if bet_away >= min_bet else 0

                # Update the bets DataFrame with calculated bet sizes
                opps.loc[opps.index == opp_idx, "newBetH"] = betH
                opps.loc[opps.index == opp_idx, "newBetA"] = betA
                todays_budget -= betH + betA

                # Stop if we run out of budget
                if todays_budget <= 0:
                    break
        else:
            # Bet on a team we predicted to win
            for opp_idx, row in opps.iterrows():
                probH = row["ProbH"]
                probA = row["ProbA"]
                if probH == 0 and probA == 0:
                    continue

                betH = non_kelly_bet_amount if probH >= probA else 0
                betA = non_kelly_bet_amount if probA > probH else 0

                betH = max(min(betH, max_bet), min_bet) if betH >= min_bet else 0
                betA = max(min(betA, max_bet), min_bet) if betA >= min_bet else 0

                opps.loc[opps.index == opp_idx, "newBetH"] = betH
                opps.loc[opps.index == opp_idx, "newBetA"] = betA
                todays_budget -= betH + betA

                if todays_budget <= 0:
                    break

        self.money_spent_yesterday = bankroll * budget_fraction - todays_budget
        bets = opps[["newBetH", "newBetA"]]
        bets.rename(columns={"newBetH": "BetH", "newBetA": "BetA"}, inplace=True)
        # Count and print the number of non-zero bets made today
        # num_bets = ((bets['BetH'] > 0) | (bets['BetA'] > 0)).sum()
        # print(f"Made {num_bets} non-zero bets today worth {self.money_spent_yesterday:.2f}$")
        self.yesterdays_bets = opps
        self.bankroll_after_bets = bankroll - self.money_spent_yesterday
        return bets


"""
conda create -n qqh python=3.12.4 -y
conda activate qqh
conda install -c conda-forge -c pytorch -c pyg numpy pandas py-xgboost-cpu scikit-learn scipy statsmodels pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 cpuonly pyg -y
"""
