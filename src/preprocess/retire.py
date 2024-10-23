import pandas as pd
import os

player_dir = 'data/player/box_season_mean'
for f in os.listdir(player_dir):
    df = pd.read_csv(f'{player_dir}/{f}')
    seasons = [int(s[-2:]) for s in df['SEASON'].unique()]
    for s, next_s in zip(seasons, seasons[1:]):
        if next_s - s > 1:
            print(f'suspect: {f} played in \'{s} and \'{next_s} but not in between.')