import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# Load datasets
games = pd.read_csv('./data/games.csv')
players = pd.read_csv('./data/players.csv')

# Feature Engineering and Aggregation
# Aggregate player statistics to team level per game
player_agg = players.groupby(['Season', 'Date', 'Team']).agg({
    'MIN': 'sum',
    'FGM': 'sum',
    'FGA': 'sum',
    'FG3M': 'sum',
    'FG3A': 'sum',
    'FTM': 'sum',
    'FTA': 'sum',
    'ORB': 'sum',
    'DRB': 'sum',
    'RB': 'sum',
    'AST': 'sum',
    'STL': 'sum',
    'BLK': 'sum',
    'TOV': 'sum',
    'PF': 'sum',
    'PTS': 'sum'
}).reset_index()
print(f"{games.shape=}")
# Merge aggregated player stats back to games data
games = games.merge(player_agg, left_on=['Season', 'Date', 'HID'], right_on=[
                    'Season', 'Date', 'Team'], suffixes=('', '_H'))
games = games.merge(player_agg, left_on=['Season', 'Date', 'AID'], right_on=[
                    'Season', 'Date', 'Team'], suffixes=('', '_A'))
print(f"{games.shape=}")

# Drop redundant columns if they exist
for col in ['Team_H', 'Team_A']:
    if col in games.columns:
        games = games.drop(columns=[col])

# Define target variable (1 if Home team wins, 0 otherwise)
# games['target'] = games['H']

# Feature Selection - Selecting key features for modeling
features = [  # 'HSC', 'ASC',
    'N', 'POFF', 'HFGM', 'AFGM', 'HFGA', 'AFGA', 'HFG3M', 'AFG3M',
    'HFG3A', 'AFG3A', 'HFTM', 'AFTM', 'HFTA', 'AFTA', 'HORB', 'AORB',
    'HDRB', 'ADRB', 'HRB', 'ARB', 'HAST', 'AAST', 'HSTL', 'ASTL',
    'HBLK', 'ABLK', 'HTOV', 'ATOV', 'HPF', 'APF',
    # 'MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'ORB', 'DRB', 'RB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS',
    # 'MIN_A', 'FGM_A', 'FGA_A', 'FG3M_A', 'FG3A_A', 'FTM_A', 'FTA_A', 'ORB_A', 'DRB_A', 'RB_A', 'AST_A', 'STL_A', 'BLK_A', 'TOV_A', 'PF_A', 'PTS_A'
]

X = games[features]
y = games['H']
print(f"{X.shape=}")
print(f"{y.shape=}")
print(f"{X.columns=}")
print(f"{X.head()=}")
print(f"{y.head()=}")
input("Press Enter to continue...")
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

# Predict probabilities for the test set
y_pred_proba = logreg.predict_proba(X_test)[:, 1]

# Evaluate model performance
accuracy = accuracy_score(y_test, logreg.predict(X_test))
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f'Accuracy: {accuracy:.2f}')
print(f'ROC AUC: {roc_auc:.2f}')

# Feature Importance using Permutation Importance
perm_importance = permutation_importance(logreg, X_test, y_test, n_repeats=10, random_state=42)
feature_importances = pd.Series(perm_importance.importances_mean, index=features)

# Plotting Feature Importances
feature_importances.sort_values(ascending=False).plot(kind='bar', figsize=(15, 7))
plt.title("Feature Importance based on Permutation Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()

# Calculate probabilities of Home and Away win for use with Kelly's criterion
# Adding predictions back to the original dataset for betting purposes
games['home_win_prob'] = logreg.predict_proba(X)[:, 1]
games['away_win_prob'] = 1 - games['home_win_prob']

# Display a few rows to understand the results
print(games[['Season', 'Date', 'HID', 'AID', 'home_win_prob', 'away_win_prob']].head())
