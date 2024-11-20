from torch import tensor, nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input layer with 408 features
        self.layer1 = nn.Linear(408, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, 32)
        self.layer5 = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.xavier_uniform_(self.layer3.weight)
        nn.init.xavier_uniform_(self.layer4.weight)
        nn.init.xavier_uniform_(self.layer5.weight)

    def forward(self, x):
        # Forward pass through the network
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        x = self.sigmoid(self.layer5(x))
        return x


def calculate_win_probs_kuba(summary, opp, games_inc, players_inc, database):
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
    current_season = opp['Season']
    match_date = opp['Date']
    home_ID = opp['HID']
    away_ID = opp['AID']
    neutral_ground = opp['N']
    playoff_game = opp['POFF']
    oddsH = opp['OddsH']
    oddsA = opp['OddsA']

    # Example use of summary
    bankroll = summary['Bankroll']
    current_date = summary['Date']
    min_bet = summary['Min_bet']
    max_bet = summary['Max_bet']

    # Example use of database
    home_team_games_stats = database.get_team_data(home_ID)
    # print(f"Last two games of home team:\n {home_team_games_stats.tail(2)}")
    away_team_game_stats = database.get_team_data(away_ID)

    player3048_stats = database.get_player_data(3048)
    # print(f"Last two games of player 3048:\n {player3048_stats.tail(2)}")
    home_win_prob = 0.5
    away_win_prob = 0.5
    # print(f"Calculated win probabilities: {home_win_prob}, {away_win_prob}")
    # input("Press Enter to confirm and continue...")
    return home_win_prob, away_win_prob


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
    home_data = database.get_team_data(home_ID).tail(6).drop(
        columns=['GameID', 'Season', 'Date', 'TeamID', 'OpponentID', 'TeamOdds', 'OpponentOdds'], inplace=False)
    away_data = database.get_team_data(away_ID).tail(6).drop(
        columns=['GameID', 'Season', 'Date', 'TeamID', 'OpponentID', 'TeamOdds', 'OpponentOdds'], inplace=False)
    # print(away_team_game_stats)

    if not home_data.empty and not away_data.empty:
        home_team_games_stats = home_data[::-1].values.tolist()
        away_team_game_stats = away_data[::-1].values.tolist()
        all_data = []
        for game_data in home_team_games_stats:
            all_data.extend(game_data)
        for game_data in away_team_game_stats:
            all_data.extend(game_data)
        home_win_prob = model(tensor(all_data).float())
        away_win_prob = 1 - home_win_prob

    return home_win_prob, away_win_prob

