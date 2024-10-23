import os
import pandas as pd

data_dir = 'REAL_RECOVERED_DATA_2PT'
player_dfs = {}

for filename in os.listdir(data_dir):
    if filename.endswith('.csv'):
        player_name = filename[:-4]
        print(f'PREPROCESSING DATA FOR {player_name}')
        player_df = pd.read_csv(os.path.join(data_dir, filename))
        player_df.rename(columns={'FG': 'FGM', 'FT': 'FTM', '3P%': '3PT_FG%'}, inplace=True)
        first_columns = ['SEASON','DATE','TEAM','OPPONENT','HOME_TEAM','MP','FGM','FGA','FG%','2PT_FGM','2PT_FGA','2PT_FG%','3PT_FGM','3PT_FGA','3PT_FG%','FTM','FTA','FT%','ORB','DRB','TRB','AST','STL','BLK','TOV','PF','PTS','GMSC','+/-','TS%','EFG%','3PAR','FTR','ORB%','DRB%','TRB%','AST%','STL%','BLK%','TOV%','USG%','ORTG','DRTG','BPM']
        player_df = player_df[first_columns + [col for col in player_df.columns if col not in first_columns]]
        player_df.to_csv(os.path.join(data_dir, f'{player_name}.csv'), index=False)