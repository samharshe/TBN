import pandas as pd
import os

dir = 'data/team/box_raw_with_game_type'
team_files = os.listdir(dir)

for team_file in team_files:
    team_df = pd.read_csv(f'{dir}/{team_file}')
    team_df = team_df[team_df['GAME_TYPE'] == 'regular_season']

    numeric_columns = team_df.select_dtypes(include=['float64', 'int64']).columns
    season_means = team_df.groupby('SEASON')[numeric_columns].mean().round(3)

    season_means.to_csv(f'data/team/box_regular_season_mean/{team_file}')