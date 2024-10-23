import os
import pandas as pd

data_dir = 'REAL_RECOVERED_DATA'
player_dfs = {}

for filename in os.listdir(data_dir):
    if filename.endswith('.csv'):
        player_name = filename[:-4]
        print(f'PREPROCESSING DATA FOR {player_name}')
        player_df = pd.read_csv(os.path.join(data_dir, filename))
        player_df['2PT_FGA'] = player_df['FGA'] - player_df['3PA']
        player_df['2PT_FGM'] = player_df['FG'] - player_df['3PM']
        player_df['2PT_FG%'] = (player_df['2PT_FGM'] / player_df['2PT_FGA']).round(3)
        player_df.rename(columns={'3PA': '3PT_FGA', '3PM': '3PT_FGM', '3P%': '3PT_FG%'}, inplace=True)
        first_columns = ['DATE','OPPONENT','HOME_TEAM','FG','FGA','FG%','2PT_FGM','2PT_FGA','2PT_FG%']
        player_df = player_df[first_columns + [col for col in player_df.columns if col not in first_columns]]
        player_df.to_csv(os.path.join(data_dir + '_2PT', f'{player_name}.csv'), index=False)