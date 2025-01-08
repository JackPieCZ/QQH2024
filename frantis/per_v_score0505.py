#Finding best corelation with PER and Enemy_PTS
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
games_csv = pd.read_csv(r'C:\Users\krupi\Documents\msem1\san\semestralka\QQH2024\data\games.csv')
players_csv = pd.read_csv(r'C:\Users\krupi\Documents\msem1\san\semestralka\QQH2024\data\players.csv')

# Merge players data with game data
player_stats = pd.merge(players_csv, games_csv, left_on='Game', right_on='ID', how='inner')

# Calculate additional stats
player_stats['Missed_FG'] = player_stats['FGA'] - player_stats['FGM']
player_stats['Missed_FT'] = player_stats['FTA'] - player_stats['FTM']

# Calculate PER
player_stats['PER'] = (player_stats['PTS'] + player_stats['AST'] + player_stats['RB'] + 
                       player_stats['STL'] + player_stats['BLK'] - 
                       player_stats['Missed_FG'] - player_stats['Missed_FT'] - player_stats['TOV']) / player_stats['MIN']

# Extract home team points
home_team_points = games_csv[['ID', 'HID', 'HSC', 'AID', 'ASC']].rename(
    columns={'HID': 'Team', 'HSC': 'Team_PTS', 'AID': 'Enemy', 'ASC': 'Enemy_PTS', 'ID': 'Game'})

# Extract away team points
away_team_points = games_csv[['ID', 'AID', 'ASC', 'HID', 'HSC']].rename(
    columns={'AID': 'Team', 'ASC': 'Team_PTS', 'HID': 'Enemy', 'HSC': 'Enemy_PTS', 'ID': 'Game'})

# Combine home and away team data
team_points = pd.concat([home_team_points, away_team_points])

# Determine if the team won
team_points['Won'] = team_points['Team_PTS'] > team_points['Enemy_PTS']

# Merge player stats with team points
merged_data = pd.merge(player_stats, team_points, left_on=['Team', 'Game'], right_on=['Team', 'Game'], how='inner')

# Group data by Player and Season
grouped_data = merged_data.groupby(['Player', 'Season_x'])

# Calculate the correlation of PER with key metrics for each player in each season
correlation_results = []
for (player, season), group in grouped_data:
    if len(group) > 0:
        corr_matrix = group[['PER', 'Team_PTS', 'Enemy_PTS', 'Won']].corr()
        corr_with_per = corr_matrix.loc['PER']
        correlation_results.append((player, season, corr_with_per['Team_PTS'], corr_with_per['Enemy_PTS'], corr_with_per['Won']))

# Convert correlation results to a DataFrame
correlation_df = pd.DataFrame(correlation_results, columns=['Player', 'Season', 'Team_PTS_CORR', 'Enemy_PTS_CORR', 'Won_CORR'])

# Filter players who played at least 4 seasons
seasons_per_player = correlation_df.groupby('Player')['Season'].nunique()
eligible_players = seasons_per_player[seasons_per_player >= 5].index
eligible_correlation_df = correlation_df[correlation_df['Player'].isin(eligible_players)]

# Find the player with the highest negative correlation to Enemy_PTS (i.e., most negative)
average_correlation = eligible_correlation_df.groupby('Player')[['Team_PTS_CORR', 'Enemy_PTS_CORR', 'Won_CORR']].mean()

# Find the player with the lowest correlation with Enemy_PTS (i.e., most negative)
best_player = average_correlation['Enemy_PTS_CORR'].idxmin()  # Take the index of the min correlation with Enemy_PTS

# Get the best player’s data
best_player_data = merged_data[merged_data['Player'] == best_player]

# Print the best player’s ID
print(f"Best player ID based on the most negative correlation with Enemy_PTS: {best_player}")

# Calculate and display the correlation matrix for the best player
correlation_best_player = best_player_data[['PER', 'Team_PTS', 'Enemy_PTS', 'Won']].corr()
print("Correlation matrix for the best player:")
print(correlation_best_player)

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

sns.regplot(
    data=best_player_data,
    x='Team_PTS',
    y='PER',
    scatter_kws={'alpha': 0.5},
    line_kws={'color': 'red', 'lw': 2},
    ax=axes[0]
)
axes[0].set_title("Best Player: Team_PTS vs PER")
axes[0].set_xlabel("Team Points")
axes[0].set_ylabel("Player Efficiency Rating (PER)")

sns.regplot(
    data=best_player_data,
    x='Enemy_PTS',
    y='PER',
    scatter_kws={'alpha': 0.5},
    line_kws={'color': 'blue', 'lw': 2},
    ax=axes[1]
)
axes[1].set_title("Best Player: Enemy_PTS vs PER")
axes[1].set_xlabel("Enemy Points")
axes[1].set_ylabel("Player Efficiency Rating (PER)")

sns.regplot(
    data=best_player_data,
    x='Won',
    y='PER',
    scatter_kws={'alpha': 0.5},
    line_kws={'color': 'green', 'lw': 2},
    ax=axes[2]
)
axes[2].set_title("Best Player: Won vs PER")
axes[2].set_xlabel("Won (1=True, 0=False)")
axes[2].set_ylabel("Player Efficiency Rating (PER)")

plt.tight_layout()
plt.show()

# Visualization for each season
seasons = best_player_data['Season_x'].unique()
fig, axes = plt.subplots(len(seasons), 3, figsize=(18, 6 * len(seasons)))

for i, season in enumerate(seasons):
    season_data = best_player_data[best_player_data['Season_x'] == season]
    
    # Plot PER vs Team_PTS
    sns.regplot(
        data=season_data,
        x='Team_PTS',
        y='PER',
        scatter_kws={'alpha': 0.5},
        line_kws={'color': 'red', 'lw': 2},
        ax=axes[i, 0] if len(seasons) > 1 else axes[0]
    )
    axes[i, 0].set_title(f"Season {season}: Team_PTS vs PER")
    axes[i, 0].set_xlabel("Team Points")
    axes[i, 0].set_ylabel("Player Efficiency Rating (PER)")

    # Plot PER vs Enemy_PTS
    sns.regplot(
        data=season_data,
        x='Enemy_PTS',
        y='PER',
        scatter_kws={'alpha': 0.5},
        line_kws={'color': 'blue', 'lw': 2},
        ax=axes[i, 1] if len(seasons) > 1 else axes[1]
    )
    axes[i, 1].set_title(f"Season {season}: Enemy_PTS vs PER")
    axes[i, 1].set_xlabel("Enemy Points")
    axes[i, 1].set_ylabel("Player Efficiency Rating (PER)")

    # Plot PER vs Won
    sns.regplot(
        data=season_data,
        x='Won',
        y='PER',
        scatter_kws={'alpha': 0.5},
        line_kws={'color': 'green', 'lw': 2},
        ax=axes[i, 2] if len(seasons) > 1 else axes[2]
    )
    axes[i, 2].set_title(f"Season {season}: Won vs PER")
    axes[i, 2].set_xlabel("Won (1=True, 0=False)")
    axes[i, 2].set_ylabel("Player Efficiency Rating (PER)")

# Adjust layout for season plots
plt.tight_layout()
plt.show()
