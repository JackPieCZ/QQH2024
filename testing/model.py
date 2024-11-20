from collections import defaultdict, deque
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from torch import tensor, nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(340, 128)
        # self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, 32)
        self.layer5 = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(self.layer1.weight)
        # nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.xavier_uniform_(self.layer3.weight)
        nn.init.xavier_uniform_(self.layer4.weight)
        nn.init.xavier_uniform_(self.layer5.weight)

    def forward(self, x):
        # Forward pass through the network
        x = self.relu(self.layer1(x))
        # x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        x = self.sigmoid(self.layer5(x))
        return x


def calculate_win_probs_kuba2(opp, database, model):
    """Calculates win probabilities for home and away team.

        Args:
            summary (pd.Dataframe): Summary of games with columns | Bankroll | Date | Min_bet | Max_bet |.
            opp (pd.Dataframe): Betting opportunities with columns ['Season', 'Date', 'HID', 'AID', 'N', 'POFF', 'OddsH', 'OddsA', 'BetH', 'BetA'].
            games_inc (pd.Dataframe): Incremental data for games.
            players_inc (pd.Dataframe): Incremental data for players.
            database (HistoricalDatabase): Database storing all past incremental data.

        Returns:
            tuple(float, float): Probability of home team winning, probability of away team winning.
        """
    # Example use of opp
    home_ID = opp['HID']
    away_ID = opp['AID']
    # Example use of database
    home_data = database.get_team_data(home_ID).tail(5).drop(
        columns=['GameID', 'Season', 'Date', 'TeamID', 'OpponentID', 'TeamOdds', 'OpponentOdds'], inplace=False)
    away_data = database.get_team_data(away_ID).tail(5).drop(
        columns=['GameID', 'Season', 'Date', 'TeamID', 'OpponentID', 'TeamOdds', 'OpponentOdds'], inplace=False)
    # print(away_team_game_stats)
    home_win_prob = None
    away_win_prob = None
    
    if len(home_data) == 5 and len(away_data) == 5:
        home_team_games_stats = home_data[::-1].values.tolist()
        away_team_game_stats = away_data[::-1].values.tolist()
        all_data = []
        for game_data in home_team_games_stats:
            all_data.extend(game_data)
        for game_data in away_team_game_stats:
            all_data.extend(game_data)
        home_win_prob = model(tensor(all_data).float()).item()
        away_win_prob = 1 - home_win_prob
    return home_win_prob, away_win_prob


def calculate_arbitrage_betting(odds_a, odds_b):
    # Step 1: Calculate the implied probabilities for each outcome
    prob_a = 1 / odds_a
    prob_b = 1 / odds_b

    # Step 2: Calculate the sum of the implied probabilities
    total_prob = prob_a + prob_b

    # Step 3: Check if there is an arbitrage opportunity (total probability < 1)
    if total_prob < 1:
        return True
    else:
        return False


class Model:
    def __init__(self, model=None):
        self.db = HistoricalDatabase()
        self.yesterdays_games = None
        self.yesterdays_bets = None
        self.money_spent_yesterday = 0
        self.bankroll_after_bets = 0
        self.model = model
        self.last_bankrolls = deque(maxlen=30)
        if model is None:
            raise NotImplementedError
            # self.model = Net()
            # self.model.load_state_dict()

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

    def evaluate_yestedays_predictions(self, bankroll):
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
                # prediction_df = pd.DataFrame(prediction).T
                # game_df = pd.DataFrame(game).T

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

    def calculate_kelly(self, opps, todays_budget, kelly_fraction, min_bet, max_bet):
        # Sort Kelly criterion in descending order and keep track of original indices
        # Create a new column with the maximum kelly of home and away
        opps["MaxKelly"] = opps[["KellyH", "KellyA"]].max(axis=1)
        sorted_win_probs_opps = opps.sort_values(by="MaxKelly", ascending=False)

        # Place bets based on Kelly criterion starting with the highest one
        for opp_idx, row in sorted_win_probs_opps.iterrows():
            # opp_idx = row["index"]
            kellyH = row["KellyH"]
            kellyA = row["KellyA"]

            # # New logic: Only bet on the outcome with the higher probability
            # probH = row["ProbH"]
            # probA = row["ProbA"]
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

    def bet_on_higher_odds(self, opps, todays_budget, min_bet, max_bet, non_kelly_bet_amount):
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

    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: tuple[pd.DataFrame, pd.DataFrame]):
        games_inc, players_inc = inc
        bankroll = summary.iloc[0]["Bankroll"]
        min_bet = summary.iloc[0]["Min_bet"]
        # Check if bankroll has stayed constant for last 30 calls
        # Round to 2 decimal places to avoid float comparison issues
        self.last_bankrolls.append(int(bankroll))
        # Compare first value to all others for efficiency
        bankroll_stuck = len(self.last_bankrolls) == 30 and all(x == self.last_bankrolls[0] for x in self.last_bankrolls)
            
        if (opps.empty and games_inc.empty and players_inc.empty) or (bankroll < min_bet) or bankroll_stuck:
            return pd.DataFrame(columns=["BetH", "BetA"])

        max_bet = summary.iloc[0]["Max_bet"]
        todays_date = summary.iloc[0]["Date"]

        # only iterate over opps with the current date while keeping the original index
        assert opps[opps["Date"] < todays_date].empty, "There are opps before today's date, which should never happen"
        todays_opps = opps[opps["Date"] == todays_date]

        self.yesterdays_games = games_inc

        calculate_win_probs_fn = calculate_win_probs_kuba2
        kelly_fraction = 0.25
        # fraction of budget we are willing to spend today
        budget_fraction = 0.1
        use_kelly = False
        todays_budget = bankroll * budget_fraction
        non_kelly_bet_amount = min_bet * 2

        # Evaluate yesterday's predictions
        # self.evaluate_yestedays_predictions(bankroll)

        self.db.add_incremental_data(games_inc)

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

            prob_home, prob_away = calculate_win_probs_fn(opp, self.db, self.model)
            if prob_home is None or prob_away is None:
                # print(f"Could not calculate win probabilities for opp {opp_idx}, skipping")
                continue
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
            self.calculate_kelly(opps, todays_budget, kelly_fraction, min_bet, max_bet)
        else:
            self.bet_on_higher_odds(opps, todays_budget, min_bet, max_bet, non_kelly_bet_amount)

        self.money_spent_yesterday = bankroll * budget_fraction - todays_budget
        bets = opps[["newBetH", "newBetA"]]
        bets.rename(columns={"newBetH": "BetH", "newBetA": "BetA"}, inplace=True)
        self.yesterdays_bets = opps
        self.bankroll_after_bets = bankroll - self.money_spent_yesterday
        return bets


class HistoricalDatabase:
    def __init__(self):
        # fmt: off
        self.team_columns = [
            "GameID", "Season", "Date", "TeamID", "OpponentID", "N", "POFF", "TeamOdds", "OpponentOdds", "W", "Home", "SC",
            "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "ORB", "DRB", "RB", "AST", "STL", "BLK", "TOV", "PF",
            "OpponentSC", "OpponentFGM", "OpponentFGA", "OpponentFG3M", "OpponentFG3A", "OpponentFTM", "OpponentFTA",
            "OpponentORB", "OpponentDRB", "OpponentRB", "OpponentAST", "OpponentSTL", "OpponentBLK", "OpponentTOV", "OpponentPF"
        ]
        self.player_columns = [
            "PlayerID", "Season", "Date", "TeamID", "GameID", "MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
            "ORB", "DRB", "RB", "AST", "STL", "BLK", "TOV", "PF", "PTS"
        ]
        # fmt: on
        self.team_data: Dict[str, List[Dict]] = defaultdict(list)
        self.player_data: Dict[str, List[Dict]] = defaultdict(list)

    def _create_team_records(self, games_df: pd.DataFrame) -> List[Tuple[str, Dict]]:
        """Create team records maintaining the original game order."""
        records = []

        n_games = len(games_df) * 2
        game_ids = np.repeat(games_df.index.values, 2)
        seasons = np.repeat(games_df["Season"].values, 2)
        dates = np.repeat(games_df["Date"].values, 2)
        n_values = np.repeat(games_df["N"].values, 2)
        poff_values = np.repeat(games_df["POFF"].values, 2)

        team_ids = np.empty(n_games, dtype=object)
        team_ids[0::2] = games_df["HID"].values
        team_ids[1::2] = games_df["AID"].values

        opponent_ids = np.empty(n_games, dtype=object)
        opponent_ids[0::2] = games_df["AID"].values
        opponent_ids[1::2] = games_df["HID"].values

        home_flags = np.empty(n_games, dtype=int)
        home_flags[0::2] = 1
        home_flags[1::2] = 0

        wins = np.empty(n_games, dtype=int)
        wins[0::2] = games_df["H"].values
        wins[1::2] = games_df["A"].values

        team_odds = np.empty(n_games)
        opponent_odds = np.empty(n_games)
        team_odds[0::2] = games_df["OddsH"].values
        team_odds[1::2] = games_df["OddsA"].values
        opponent_odds[0::2] = games_df["OddsA"].values
        opponent_odds[1::2] = games_df["OddsH"].values

        stats_mapping = {
            "SC": ("HSC", "ASC"),
            "FGM": ("HFGM", "AFGM"),
            "FGA": ("HFGA", "AFGA"),
            "FG3M": ("HFG3M", "AFG3M"),
            "FG3A": ("HFG3A", "AFG3A"),
            "FTM": ("HFTM", "AFTM"),
            "FTA": ("HFTA", "AFTA"),
            "ORB": ("HORB", "AORB"),
            "DRB": ("HDRB", "ADRB"),
            "RB": ("HRB", "ARB"),
            "AST": ("HAST", "AAST"),
            "STL": ("HSTL", "ASTL"),
            "BLK": ("HBLK", "ABLK"),
            "TOV": ("HTOV", "ATOV"),
            "PF": ("HPF", "APF"),
        }

        stats_arrays = {}
        for stat, (h_col, a_col) in stats_mapping.items():
            stat_values = np.empty(n_games)
            stat_values[0::2] = games_df[h_col].values
            stat_values[1::2] = games_df[a_col].values
            stats_arrays[stat] = stat_values

        for i in range(n_games):
            record = {
                "GameID": game_ids[i],
                "Season": seasons[i],
                "Date": dates[i],
                "TeamID": team_ids[i],
                "OpponentID": opponent_ids[i],
                "N": n_values[i],
                "POFF": poff_values[i],
                "TeamOdds": team_odds[i],
                "OpponentOdds": opponent_odds[i],
                "Home": home_flags[i],
                "W": wins[i],
                **{stat: stats_arrays[stat][i] for stat in stats_mapping.keys()}
            }

            # Adding opponent stats to the record
            opponent_stats = {
                f"Opponent{stat}": stats_arrays[stat][i ^ 1]  # XOR with 1 to get the opponent's value
                for stat in stats_mapping.keys()
            }

            record.update(opponent_stats)
            records.append((team_ids[i], record))

        return records

    def add_incremental_data(self, games_df: pd.DataFrame) -> None:
        """Add new game and player data efficiently while maintaining game order."""
        team_records = self._create_team_records(games_df)

        for team_id, record in team_records:
            self.team_data[team_id].append(record)

    def get_team_data(self, team_id: str) -> pd.DataFrame:
        """Get team data, returning empty DataFrame if team not found."""
        return pd.DataFrame(self.team_data.get(team_id, []), columns=self.team_columns)


"""
conda create -n qqh python=3.12.4 -y
conda activate qqh
conda install -c conda-forge -c pytorch -c pyg numpy pandas py-xgboost-cpu scikit-learn scipy statsmodels pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 cpuonly pyg -y
pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
"""
