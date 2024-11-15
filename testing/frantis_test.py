import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


import pandas as pd

def get_team_matches(games_df, team1, team2):
    """
    Get the dates and scores of matches between two teams in specified order.

    Args:
        games_df (pd.DataFrame): DataFrame containing games data with columns 
                                 ['game_id', 'date', 'home_team', 'away_team', 'home_score', 'away_score'].
        team1 (str): Name or ID of the first team.
        team2 (str): Name or ID of the second team.

    Returns:
        pd.DataFrame: DataFrame with columns ['date', 'team1_score', 'team2_score'].
    """
    # Filter matches where team1 and team2 played against each other
    matches = games_df[
        ((games_df['HID'] == team1) & (games_df['AID'] == team2)) |
        ((games_df['HID'] == team2) & (games_df['AID'] == team1))
    ]

    # Prepare result with team1 and team2 scores in the correct order
    matches = matches.assign(
        team1_score=matches.apply(
            lambda row: row['HSC'] if row['HID'] == team1 else row['ASC'], axis=1
        ),
        team2_score=matches.apply(
            lambda row: row['ASC'] if row['HID'] == team1 else row['HSC'], axis=1
        )
    )

    # Select and sort the relevant columns
    result = matches[['Date', 'team1_score', 'team2_score']].sort_values(by='Date').reset_index(drop=True)

    return result

# Example usage
data = pd.read_csv('./testing/data/games.csv')

games_df = pd.DataFrame(data)

# Find matches between TeamA and TeamB
matches = get_team_matches(games_df, '1', '2')
print(matches)


