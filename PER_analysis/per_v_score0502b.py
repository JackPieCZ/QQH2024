#Corelation of top players by PER
#Corelation of PER groups
#Best corelated with Team_PTS
#Best corelation with Enemy_PTS
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
games_csv = pd.read_csv(r'C:\Users\krupi\Documents\msem1\san\semestralka\QQH2024\data\games.csv')
players_csv = pd.read_csv(r'C:\Users\krupi\Documents\msem1\san\semestralka\QQH2024\data\players.csv')

# Merge players data with game data
player_stats = pd.merge(players_csv, games_csv, left_on='Game', right_on='ID', how='inner')
player_stats = player_stats[player_stats['Season_x'] == 1]
#print(player_stats)
# Calculate PER components for each player
player_stats['Missed_FG'] = player_stats['FGA'] - player_stats['FGM']
player_stats['Missed_FT'] = player_stats['FTA'] - player_stats['FTM']

player_stats['PER'] = (player_stats['PTS'] + player_stats['AST'] + player_stats['RB'] + 
                       player_stats['STL'] + player_stats['BLK'] - 
                       player_stats['Missed_FG'] - player_stats['Missed_FT'] - player_stats['TOV']) / player_stats['MIN']

# Bin PER into groups
bins = [-np.inf, 0, 0.25, 0.5, 0.75, 1, np.inf]
labels = ['<0', '0-0.25', '0.25-0.5', '0.5-0.75', '0.75-1', '>1']
player_stats['PER_group'] = pd.cut(player_stats['PER'], bins=bins, labels=labels)

# Extract home team points
home_team_points = games_csv[['ID', 'HID', 'HSC', 'AID', 'ASC']].rename(columns={'HID': 'Team', 'HSC': 'Team_PTS', 'AID': 'Enemy', 'ASC': 'Enemy_PTS', 'ID': 'Game'})

# Extract away team points
away_team_points = games_csv[['ID', 'AID', 'ASC', 'HID', 'HSC']].rename(columns={'AID': 'Team', 'ASC': 'Team_PTS', 'HID': 'Enemy', 'HSC': 'Enemy_PTS', 'ID': 'Game'})

# Combine home and away team data
team_points = pd.concat([home_team_points, away_team_points])

# Determine if the team won
team_points['Won'] = team_points['Team_PTS'] > team_points['Enemy_PTS']

# Sort by Team and Game
team_points = team_points.sort_values(by=['Team', 'Game']).reset_index(drop=True)

# Print the result
#print(team_points)

# Merge player stats with team points
merged_data = pd.merge(player_stats, team_points, left_on=['Team', 'Game'], right_on=['Team', 'Game'], how='inner')

# Calculate correlation between PER and Team_PTS, Enemy_PTS, and Won
correlation = merged_data[['PER', 'Team_PTS', 'Enemy_PTS', 'Won']].corr()

# Compute the average PER, Points, and Minutes for each player
player_summary = merged_data.groupby('Player').agg(
    Average_PER=('PER', 'mean'),
    Average_PTS=('PTS', 'mean'),
    Average_MIN=('MIN', 'mean')
).reset_index()

# Order the summary by Average PER in descending order
player_summary = player_summary.sort_values(by='Average_PER', ascending=False)

# Save the player summary to a CSV file
player_summary.to_csv(
    r'C:\Users\krupi\Documents\msem1\san\semestralka\QQH2024\frantis\Player_Summary.csv',
    index=False
)

filtered_summary = player_summary[player_summary['Average_MIN'] > 10]

# Order by Average_PER in descending order and select the top 5
top_5_players = filtered_summary.sort_values(by='Average_PER', ascending=False).head(5)

# Print the top 5 players
print(top_5_players)



# Visualization for a selected player (for example, Player 12)
selected_player_id = 1240  # Change this to the desired player ID
selected_player_data = merged_data[merged_data['Player'] == selected_player_id]

player_summary = merged_data.groupby('Player').agg(
    Average_PER=('PER', 'mean'),
    Average_PTS=('PTS', 'mean'),
    Average_MIN=('MIN', 'mean')
).reset_index()
# Filter merged_data for the top 5 players
top_5_players = player_summary[player_summary['Average_MIN'] > 15].sort_values(by='Average_PER', ascending=False).head(5)
#top_5_player_data = merged_data[merged_data['Player'].isin(top_5_correlated_players['Player'])] #top 5 correlated players
top_5_player_data = merged_data[merged_data['Player'].isin(top_5_players['Player'])]


# Initialize the figure with a 5x3 grid (5 rows and 3 columns)
fig, axes = plt.subplots(5, 3, figsize=(15, 15))

# Colors for each player
colors = ['blue', 'orange', 'green', 'red', 'purple']

# Loop through the top 5 players and plot the data for each one
for i, player in enumerate(top_5_players['Player']):
    player_data = top_5_player_data[top_5_player_data['Player'] == player]
    
    # Team_PTS vs PER
    sns.regplot(
        data=player_data, 
        x='Team_PTS', 
        y='PER', 
        scatter_kws={'alpha': 0.5}, 
        line_kws={'label': f'{player} (Team_PTS)', 'color': colors[i], 'lw': 2}, 
        color=colors[i],
        ax=axes[i, 0]  # Plot in the first column (Team_PTS)
    )
    axes[i, 0].set_title(f"{player} - Team_PTS vs PER")
    axes[i, 0].set_xlabel("Team Points")
    axes[i, 0].set_ylabel("Player Efficiency Rating (PER)")

    # Enemy_PTS vs PER
    sns.regplot(
        data=player_data, 
        x='Enemy_PTS', 
        y='PER', 
        scatter_kws={'alpha': 0.5}, 
        line_kws={'label': f'{player} (Enemy_PTS)', 'color': colors[i], 'lw': 2, 'ls': '--'}, 
        color=colors[i],
        ax=axes[i, 1]  # Plot in the second column (Enemy_PTS)
    )
    axes[i, 1].set_title(f"{player} - Enemy_PTS vs PER")
    axes[i, 1].set_xlabel("Enemy Points")
    axes[i, 1].set_ylabel("Player Efficiency Rating (PER)")

    # Won vs PER
    sns.regplot(
        data=player_data, 
        x='Won', 
        y='PER', 
        scatter_kws={'alpha': 0.5}, 
        line_kws={'label': f'{player} (Won)', 'color': colors[i], 'lw': 2, 'ls': ':'}, 
        color=colors[i],
        ax=axes[i, 2]  # Plot in the third column (Won)
    )
    axes[i, 2].set_title(f"{player} - Won vs PER")
    axes[i, 2].set_xlabel("Won")
    axes[i, 2].set_ylabel("Player Efficiency Rating (PER)")

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

# Calculate correlation for each PER group
grouped_data = merged_data.groupby('PER_group')[['PER', 'Team_PTS', 'Enemy_PTS', 'Won']].corr()

# Print the correlation results for each PER group
#print(grouped_data)

# Number of unique PER groups
# Define the specific PER groups you want to focus on
specific_groups = ['0-0.25', '0.25-0.5', '0.5-0.75', '0.75-1']

# Filter the data for the specific PER groups
filtered_data = merged_data[merged_data['PER_group'].isin(specific_groups)]

# Define the number of rows and columns for the subplots based on the number of groups
num_rows = 2  # Adjust the number of rows
num_columns = 2  # Adjust the number of columns

# Create subplots for each specific PER group
for i, group in enumerate(specific_groups):
    plt.subplot(num_rows, num_columns, i + 1)
    
    # Filter data for the current PER group
    group_data = filtered_data[filtered_data['PER_group'] == group]
    
    # Calculate and plot correlation
    corr = group_data[['PER', 'Team_PTS', 'Enemy_PTS', 'Won']].corr()
    
    # Heatmap of the correlation matrix
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, 
                xticklabels=corr.columns, yticklabels=corr.columns)
    
    plt.title(f"Correlation for PER Group: {group}")
    plt.tight_layout()

# Show the plot
plt.show()

#----------------------------------------------------------------------------
# Filter for players with Average_MIN > 20
filtered_data = merged_data.groupby('Player').agg(
    Average_MIN=('MIN', 'mean')
).reset_index()
filtered_players = filtered_data[filtered_data['Average_MIN'] > 20]['Player']

filtered_merged_data = merged_data[merged_data['Player'].isin(filtered_players)]

# Calculate correlation between PER and Team_PTS for filtered players
team_pts_correlation = filtered_merged_data.groupby('Player').apply(
    lambda group: group[['PER', 'Team_PTS']].corr().iloc[0, 1]
).reset_index(name='Correlation_PER_Team_PTS')

# Add average statistics for filtered players
player_summary = filtered_merged_data.groupby('Player').agg(
    Average_PER=('PER', 'mean'),
    Average_PTS=('PTS', 'mean'),
    Average_MIN=('MIN', 'mean')
).reset_index()

# Merge the correlations with the player summary
team_pts_correlation = team_pts_correlation.merge(player_summary, on='Player')

# Rank players by absolute value of the correlation
team_pts_correlation['Absolute_Correlation'] = team_pts_correlation['Correlation_PER_Team_PTS'].abs()
top_5_team_pts_players = team_pts_correlation.sort_values(
    by='Absolute_Correlation', ascending=False
).head(5)


# Print statistics for the top 5 players by Team_PTS correlation
print("Top 5 Players by Team_PTS Correlation:")
print(top_5_team_pts_players)

# Repeat the same process for Enemy_PTS correlation
enemy_pts_correlation = filtered_merged_data.groupby('Player').apply(
    lambda group: group[['PER', 'Enemy_PTS']].corr().iloc[0, 1]
).reset_index(name='Correlation_PER_Enemy_PTS')

# Merge with player summary
enemy_pts_correlation = enemy_pts_correlation.merge(player_summary, on='Player')

# Rank players by absolute value of the correlation
enemy_pts_correlation['Absolute_Correlation'] = enemy_pts_correlation['Correlation_PER_Enemy_PTS'].abs()
top_5_enemy_pts_players = enemy_pts_correlation.sort_values(
    by='Correlation_PER_Enemy_PTS', ascending=False
).tail(5)

# Save the full leaderboard for Enemy_PTS correlation, sorted by Absolute_Correlation
player_correlation_enemy_pts = enemy_pts_correlation.sort_values(by='Absolute_Correlation', ascending=False)
player_correlation_enemy_pts.to_csv(
    r'C:\Users\krupi\Documents\msem1\san\semestralka\QQH2024\frantis\Full_Enemy_PTS_Correlation.csv',
    index=False
)

# Print statistics for the top 5 players by Enemy_PTS correlation
print("Top 5 Players by Enemy_PTS Correlation:")
print(top_5_enemy_pts_players)

# Visualization for top 5 players by Team_PTS correlation
top_5_team_pts_players_data = filtered_merged_data[filtered_merged_data['Player'].isin(top_5_team_pts_players['Player'])]

fig, axes = plt.subplots(5, 3, figsize=(15, 15))
colors = ['blue', 'orange', 'green', 'red', 'purple']

for i, player in enumerate(top_5_team_pts_players['Player']):
    player_data = top_5_team_pts_players_data[top_5_team_pts_players_data['Player'] == player]
    
    sns.regplot(data=player_data, x='Team_PTS', y='PER', ax=axes[i, 0], scatter_kws={'alpha': 0.5}, 
                line_kws={'color': colors[i], 'lw': 2})
    axes[i, 0].set_title(f"{player} - Team_PTS vs PER")
    
    sns.regplot(data=player_data, x='Enemy_PTS', y='PER', ax=axes[i, 1], scatter_kws={'alpha': 0.5}, 
                line_kws={'color': colors[i], 'lw': 2})
    axes[i, 1].set_title(f"{player} - Enemy_PTS vs PER")
    
    sns.regplot(data=player_data, x='Won', y='PER', ax=axes[i, 2], scatter_kws={'alpha': 0.5}, 
                line_kws={'color': colors[i], 'lw': 2})
    axes[i, 2].set_title(f"{player} - Won vs PER")

plt.tight_layout()
plt.show()


#Enemy_PTS corelation

# Visualization for top 5 players by Enemy_PTS correlation
top_5_enemy_pts_players_data = filtered_merged_data[filtered_merged_data['Player'].isin(top_5_enemy_pts_players['Player'])]

fig, axes = plt.subplots(5, 3, figsize=(15, 15))
colors = ['blue', 'orange', 'green', 'red', 'purple']

for i, player in enumerate(top_5_enemy_pts_players['Player']):
    player_data = top_5_enemy_pts_players_data[top_5_enemy_pts_players_data['Player'] == player]
    
    # Enemy_PTS vs PER
    sns.regplot(data=player_data, x='Enemy_PTS', y='PER', ax=axes[i, 0], scatter_kws={'alpha': 0.5}, 
                line_kws={'color': colors[i], 'lw': 2})
    axes[i, 0].set_title(f"{player} - Enemy_PTS vs PER")
    axes[i, 0].set_xlabel("Enemy Points")
    axes[i, 0].set_ylabel("Player Efficiency Rating (PER)")

    # Team_PTS vs PER
    sns.regplot(data=player_data, x='Team_PTS', y='PER', ax=axes[i, 1], scatter_kws={'alpha': 0.5}, 
                line_kws={'color': colors[i], 'lw': 2})
    axes[i, 1].set_title(f"{player} - Team_PTS vs PER")
    axes[i, 1].set_xlabel("Team Points")
    axes[i, 1].set_ylabel("Player Efficiency Rating (PER)")

    # Won vs PER
    sns.regplot(data=player_data, x='Won', y='PER', ax=axes[i, 2], scatter_kws={'alpha': 0.5}, 
                line_kws={'color': colors[i], 'lw': 2})
    axes[i, 2].set_title(f"{player} - Won vs PER")
    axes[i, 2].set_xlabel("Won")
    axes[i, 2].set_ylabel("Player Efficiency Rating (PER)")

plt.tight_layout()
plt.show()

#tito hraci maji sice hazkou negativni korelaci
#rucne jsem se u nich dival na prumer za vsechny sezony a korelace se uz nedela