import os
import pandas as pd

games_df = pd.read_csv('data/game/bbref_game.csv')

games_df = games_df[games_df['SEASON'].apply(lambda x: int(x[-2:]) > 19)]

games_df.to_csv('data/game/bbref_game_mini.csv', index=False)