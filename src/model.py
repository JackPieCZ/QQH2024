import numpy as np
import pandas as pd


class Model:

    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: tuple[pd.DataFrame, pd.DataFrame]):
        min_bet = summary.iloc[0]["Min_bet"]
        N = len(opps)
        bets = np.zeros((N, 2))
        bets[np.arange(N), np.random.choice([0, 1])] = min_bet
        bets = pd.DataFrame(data=bets, columns=["BetH", "BetA"], index=opps.index)
        return bets


"""
conda create -n qqh python=3.12.4 -y
conda activate qqh
conda install -c conda-forge -c pytorch -c pyg numpy pandas py-xgboost-cpu scikit-learn scipy statsmodels pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 cpuonly pyg -y
"""
