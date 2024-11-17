import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class HistoricalDatabase:
    def __init__(self):
        # fmt: off
        self.team_columns = [
            "GameID", "Season", "Date", "TeamID", "OpponentID", "N", "POFF", "TeamOdds", "OpponentOdds", "W", "Home", "SC",
            "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "ORB", "DRB", "RB", "AST", "STL", "BLK", "TOV", "PF"
        ]

        self.player_columns = [
            "PlayerID", "Season", "Date", "TeamID", "GameID", "MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
            "ORB", "DRB", "RB", "AST", "STL", "BLK", "TOV", "PF", "PTS"
        ]
        # fmt: on
        self.team_data: Dict[str, pd.DataFrame] = {}
        self.player_data: Dict[str, pd.DataFrame] = {}

    def _create_team_records(self, games_df: pd.DataFrame) -> List[Tuple[str, pd.DataFrame]]:
        """Create team records maintaining the original game order."""
        records = []

        # Pre-allocate arrays for better performance
        n_games = len(games_df) * 2  # Each game creates two records
        game_ids = np.repeat(games_df.index.values, 2)
        seasons = np.repeat(games_df["Season"].values, 2)
        dates = np.repeat(games_df["Date"].values, 2)
        n_values = np.repeat(games_df["N"].values, 2)
        poff_values = np.repeat(games_df["POFF"].values, 2)

        # Interleave home and away team data
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

        # Map odds correctly based on home/away status
        team_odds = np.empty(n_games)
        opponent_odds = np.empty(n_games)
        team_odds[0::2] = games_df["OddsH"].values  # Home team odds
        team_odds[1::2] = games_df["OddsA"].values  # Away team odds
        # Away team odds (opponent of home team)
        opponent_odds[0::2] = games_df["OddsA"].values
        # Home team odds (opponent of away team)
        opponent_odds[1::2] = games_df["OddsH"].values

        # Create arrays for all stats
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

        # Create DataFrame with all records
        all_records = pd.DataFrame(
            {
                "GameID": game_ids,
                "Season": seasons,
                "Date": dates,
                "TeamID": team_ids,
                "OpponentID": opponent_ids,
                "N": n_values,
                "POFF": poff_values,
                "TeamOdds": team_odds,
                "OpponentOdds": opponent_odds,
                "Home": home_flags,
                "W": wins,
                **stats_arrays,
            }
        )

        # Split records by team while maintaining order
        for team_id, team_games in all_records.groupby("TeamID"):
            team_games_sorted = team_games.sort_values(["Season", "Date", "GameID"])
            records.append((team_id, team_games_sorted))

        return records

    def add_incremental_data(self, games_df: pd.DataFrame, players_df: pd.DataFrame) -> None:
        """Add new game and player data efficiently while maintaining game order."""
        # Process all team records at once
        team_records = self._create_team_records(games_df)

        # Update team data
        for team_id, team_games in team_records:
            if team_id in self.team_data:
                self.team_data[team_id] = pd.concat([self.team_data[team_id], team_games], ignore_index=True)
            else:
                self.team_data[team_id] = team_games

        # Process players efficiently
        for player_id, player_games in players_df.groupby("Player"):
            player_stats = player_games.rename(columns={"Player": "PlayerID", "Team": "TeamID", "Game": "GameID"})
            player_stats = player_stats[self.player_columns]

            if player_id in self.player_data:
                self.player_data[player_id] = pd.concat([self.player_data[player_id], player_stats], ignore_index=True)
            else:
                self.player_data[player_id] = player_stats

        self._verify_data_integrity()

    def _verify_data_integrity(self) -> None:
        """Verify data integrity using vectorized operations."""
        for team_id, team_df in self.team_data.items():
            if not (team_df["TeamID"] == team_id).all():
                raise ValueError(f"TeamID mismatch in team_data for team {team_id}")

            # Verify chronological order
            if not (team_df["Date"].is_monotonic_increasing or len(team_df) == 0):
                raise ValueError(f"Games not in chronological order for team {team_id}")

        for player_id, player_df in self.player_data.items():
            if not (player_df["PlayerID"] == player_id).all():
                raise ValueError(f"PlayerID mismatch in player_data for player {player_id}")

    def get_team_data(self, team_id: str) -> pd.DataFrame:
        """Get team data, returning empty DataFrame if team not found."""
        return self.team_data.get(team_id, pd.DataFrame(columns=self.team_columns))

    def get_player_data(self, player_id: str) -> pd.DataFrame:
        """Get player data, returning empty DataFrame if player not found."""
        return self.player_data.get(player_id, pd.DataFrame(columns=self.player_columns))


# import pandas as pd
# from collections import defaultdict


# class HistoricalDatabase:
#     def __init__(self):
#         # A dictionary to hold DataFrames for each team
#         self.team_data = defaultdict(lambda: pd.DataFrame(columns=[
#             'GameID', 'Season', 'Date', 'TeamID', 'OpponentID', 'N', 'POFF', 'OddsH', 'OddsA', 'W', 'Home',
#             'SC', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'ORB', 'DRB', 'RB', 'AST', 'STL', 'BLK', 'TOV', 'PF'
#         ]))
#         # A dictionary to hold DataFrames for each player
#         self.player_data = defaultdict(lambda: pd.DataFrame(columns=[
#             'PlayerID', 'Season', 'Date', 'TeamID', 'GameID', 'MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'ORB', 'DRB', 'RB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS'
#         ]))

#     def add_incremental_data(self, games_df, players_df):
#         # Iterate through each game in the incremental Games DataFrame
#         for row_id, row in games_df.iterrows():
#             # Extract common game information
#             game_info = {
#                 'GameID': row_id,
#                 'Season': row['Season'],
#                 'Date': row['Date'],
#                 'N': row['N'],
#                 'POFF': row['POFF'],
#                 'OddsH': row['OddsH'],
#                 'OddsA': row['OddsA'],
#             }

#             # Update Home Team DataFrame
#             home_team_data = game_info.copy()
#             home_team_data.update({
#                 'TeamID': row['HID'],
#                 'OpponentID': row['AID'],
#                 'Home': 1,
#                 'W': row['H'],
#                 'SC': row['HSC'],
#                 'FGM': row['HFGM'],
#                 'FGA': row['HFGA'],
#                 'FG3M': row['HFG3M'],
#                 'FG3A': row['HFG3A'],
#                 'FTM': row['HFTM'],
#                 'FTA': row['HFTA'],
#                 'ORB': row['HORB'],
#                 'DRB': row['HDRB'],
#                 'RB': row['HRB'],
#                 'AST': row['HAST'],
#                 'STL': row['HSTL'],
#                 'BLK': row['HBLK'],
#                 'TOV': row['HTOV'],
#                 'PF': row['HPF'],
#             })
#             self.team_data[row['HID']] = pd.concat(
#                 [self.team_data[row['HID']], pd.DataFrame([home_team_data])], ignore_index=True)

#             # Update Away Team DataFrame
#             away_team_data = game_info.copy()
#             away_team_data.update({
#                 'TeamID': row['AID'],
#                 'OpponentID': row['HID'],
#                 'Home': 0,
#                 'W': row['A'],
#                 'SC': row['ASC'],
#                 'FGM': row['AFGM'],
#                 'FGA': row['AFGA'],
#                 'FG3M': row['AFG3M'],
#                 'FG3A': row['AFG3A'],
#                 'FTM': row['AFTM'],
#                 'FTA': row['AFTA'],
#                 'ORB': row['AORB'],
#                 'DRB': row['ADRB'],
#                 'RB': row['ARB'],
#                 'AST': row['AAST'],
#                 'STL': row['ASTL'],
#                 'BLK': row['ABLK'],
#                 'TOV': row['ATOV'],
#                 'PF': row['APF'],
#             })
#             self.team_data[row['AID']] = pd.concat(
#                 [self.team_data[row['AID']], pd.DataFrame([away_team_data])], ignore_index=True)

#         for team_id, team_df in self.team_data.items():
#             assert (team_df['TeamID'] == team_id).all(), f"TeamID mismatch in team_data for team {team_id}"
#         # Iterate through each player in the incremental Players DataFrame
#         for row_id, row in players_df.iterrows():
#             # Extract player-specific information
#             player_data = {
#                 'PlayerID': row['Player'],
#                 'Season': row['Season'],
#                 'Date': row['Date'],
#                 'TeamID': row['Team'],
#                 'GameID': row['Game'],
#                 'MIN': row['MIN'],
#                 'FGM': row['FGM'],
#                 'FGA': row['FGA'],
#                 'FG3M': row['FG3M'],
#                 'FG3A': row['FG3A'],
#                 'FTM': row['FTM'],
#                 'FTA': row['FTA'],
#                 'ORB': row['ORB'],
#                 'DRB': row['DRB'],
#                 'RB': row['RB'],
#                 'AST': row['AST'],
#                 'STL': row['STL'],
#                 'BLK': row['BLK'],
#                 'TOV': row['TOV'],
#                 'PF': row['PF'],
#                 'PTS': row['PTS'],
#             }
#             self.player_data[row['Player']] = pd.concat(
#                 [self.player_data[row['Player']], pd.DataFrame([player_data])], ignore_index=True)

#         for player_id, player_df in self.player_data.items():
#             assert (player_df['PlayerID'] == player_id).all(), f"PlayerID mismatch in player_data for player {player_id}"

#     def get_team_data(self, team_id):
#         # Return the DataFrame for the specified team ID
#         return self.team_data[team_id]

#     def get_player_data(self, player_id):
#         # Return the DataFrame for the specified player ID
#         return self.player_data[player_id]


# Example usage
if __name__ == "__main__":
    db = HistoricalDatabase()

    # Example Games DataFrame
    games_data = {
        "ID": [15, 16],
        "Season": [1, 1],
        "Date": ["1975-11-08", "1975-11-08"],
        "HID": [22, 12],
        "AID": [41, 24],
        "N": [0, 0],
        "POFF": [0, 0],
        "OddsH": [1.93, 1.42],
        "OddsA": [1.92, 3.05],
        "H": [1, 1],
        "A": [0, 0],
        "HSC": [111, 119],
        "ASC": [105, 110],
        "HFGM": [46, 50],
        "AFGM": [41, 40],
        "HFGA": [93, 111],
        "AFGA": [85, 93],
        "HFG3M": [3, 1],
        "AFG3M": [0, 3],
        "HFG3A": [7, 7],
        "AFG3A": [4, 7],
        "HFTM": [16, 18],
        "AFTM": [23, 27],
        "HFTA": [23, 29],
        "AFTA": [28, 41],
        "HORB": [11, 28],
        "AORB": [12, 18],
        "HDRB": [30, 38],
        "ADRB": [33, 31],
        "HRB": [41, 66],
        "ARB": [45, 49],
        "HAST": [27, 29],
        "AAST": [23, 26],
        "HSTL": [12, 13],
        "ASTL": [4, 8],
        "HBLK": [2, 16],
        "ABLK": [7, 9],
        "HTOV": [11, 12],
        "ATOV": [19, 19],
        "HPF": [24, 31],
        "APF": [20, 23],
    }
    games_df = pd.DataFrame(games_data)

    # Example Players DataFrame
    players_data = {
        "ID": [198, 199],
        "Season": [1, 1],
        "Date": ["1975-11-08", "1975-11-08"],
        "Player": [3048, 4000],
        "Team": [12, 24],
        "Game": [16, 16],
        "MIN": [16.0, 33.0],
        "FGM": [2, 11],
        "FGA": [3.0, 21.0],
        "FG3M": [0.0, 0.0],
        "FG3A": [0.0, 1.0],
        "FTM": [1, 3],
        "FTA": [2.0, 3.0],
        "ORB": [0.0, 3.0],
        "DRB": [3.0, 2.0],
        "RB": [3.0, 5.0],
        "AST": [1.0, 1.0],
        "STL": [0.0, 0.0],
        "BLK": [0.0, 1.0],
        "TOV": [0.0, 1.0],
        "PF": [2.0, 1.0],
        "PTS": [5, 25],
    }
    players_df = pd.DataFrame(players_data)

    # Add incremental data
    db.add_incremental_data(games_df, players_df)

    # Get data for a specific team
    team_data = db.get_team_data(22)
    print(team_data)

    # Get data for a specific player
    player_data = db.get_player_data(3048)
    print(player_data)
