#Statistics for one player (player599)
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
    columns={'HID': 'Team', 'HSC': 'Team_PTS', 'AID': 'Enemy', 'ASC': 'Enemy_PTS', 'ID': 'Game'}
)

# Extract away team points
away_team_points = games_csv[['ID', 'AID', 'ASC', 'HID', 'HSC']].rename(
    columns={'AID': 'Team', 'ASC': 'Team_PTS', 'HID': 'Enemy', 'HSC': 'Enemy_PTS', 'ID': 'Game'}
)

# Combine home and away team data
team_points = pd.concat([home_team_points, away_team_points])

# Determine if the team won
team_points['Won'] = team_points['Team_PTS'] > team_points['Enemy_PTS']

# Merge player stats with team points
merged_data = pd.merge(player_stats, team_points, left_on=['Team', 'Game'], right_on=['Team', 'Game'], how='inner')

# Filter the merged_data for Player 599
player_599_data = merged_data[merged_data['Player'] == 599]

# Ensure player_599_data is not empty
if player_599_data.empty:
    print("No data available for Player 599.")
else:
    # Calculate correlation matrix for Player 599
    correlation_599 = player_599_data[['PER', 'Team_PTS', 'Enemy_PTS', 'Won']].corr()
    print("Correlation matrix for Player 599:")
    print(correlation_599)

    # Summary visualization across all seasons
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot PER vs Team_PTS
    sns.regplot(
        data=player_599_data,
        x='Team_PTS',
        y='PER',
        scatter_kws={'alpha': 0.5},
        line_kws={'color': 'red', 'lw': 2},
        ax=axes[0]
    )
    axes[0].set_title("Player 599: Team_PTS vs PER")
    axes[0].set_xlabel("Team Points")
    axes[0].set_ylabel("Player Efficiency Rating (PER)")

    # Plot PER vs Enemy_PTS
    sns.regplot(
        data=player_599_data,
        x='Enemy_PTS',
        y='PER',
        scatter_kws={'alpha': 0.5},
        line_kws={'color': 'blue', 'lw': 2},
        ax=axes[1]
    )
    axes[1].set_title("Player 599: Enemy_PTS vs PER")
    axes[1].set_xlabel("Enemy Points")
    axes[1].set_ylabel("Player Efficiency Rating (PER)")

    # Plot PER vs Won
    sns.regplot(
        data=player_599_data,
        x='Won',
        y='PER',
        scatter_kws={'alpha': 0.5},
        line_kws={'color': 'green', 'lw': 2},
        ax=axes[2]
    )
    axes[2].set_title("Player 599: Won vs PER")
    axes[2].set_xlabel("Won (1=True, 0=False)")
    axes[2].set_ylabel("Player Efficiency Rating (PER)")

    # Adjust layout for summary
    plt.tight_layout()
    plt.show()

    # Visualization for each season
    seasons = player_599_data['Season_x'].unique()
    fig, axes = plt.subplots(len(seasons), 3, figsize=(18, 6 * len(seasons)))

    for i, season in enumerate(seasons):
        season_data = player_599_data[player_599_data['Season_x'] == season]
        
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
        axes[i, 0].set_ylabel("PER")

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
        axes[i, 1].set_ylabel("PER")

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
        axes[i, 2].set_ylabel("PER")

    # Adjust layout for season plots
    plt.tight_layout()
    plt.show()
