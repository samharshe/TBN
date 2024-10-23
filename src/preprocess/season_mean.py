import pandas as pd
import os

player_dir = 'raw/player'
for player_name in os.listdir(player_dir):
    player_df = pd.read_csv(os.path.join(player_dir, player_name))
    player_df.drop(columns=['DATE', 'TEAM', 'OPPONENT', 'HOME_TEAM'], inplace=True)
    season_mean = player_df.groupby('SEASON').mean()
    season_mean['FG%'] = season_mean['FG'] / season_mean['FGA']
    season_mean['3P%'] = season_mean['3PM'] / season_mean['3PA']
    season_mean['FT%'] = season_mean['FT'] / season_mean['FTA']
    
    season_mean['ORB%'] = player_df[player_df['MP'] != 0].groupby('SEASON').mean()['ORB%']
    season_mean['DRB%'] = player_df[player_df['MP'] != 0].groupby('SEASON').mean()['DRB%']
    season_mean['TRB%'] = player_df[player_df['MP'] != 0].groupby('SEASON').mean()['TRB%']
    season_mean['AST%'] = player_df[player_df['MP'] != 0].groupby('SEASON').mean()['AST%']
    season_mean['STL%'] = player_df[player_df['MP'] != 0].groupby('SEASON').mean()['STL%']
    season_mean['BLK%'] = player_df[player_df['MP'] != 0].groupby('SEASON').mean()['BLK%']
    season_mean['TOV%'] = player_df[player_df['MP'] != 0].groupby('SEASON').mean()['TOV%']
    season_mean['USG%'] = player_df[player_df['MP'] != 0].groupby('SEASON').mean()['USG%']
    season_mean['DRTG'] = player_df[player_df['MP'] != 0].groupby('SEASON').mean()['DRTG']
    season_mean['ORTG'] = player_df[player_df['MP'] != 0].groupby('SEASON').mean()['ORTG']
    
    season_mean['3PAR'] = player_df[player_df['FGA'] != 0].groupby('SEASON').mean()['3PAR']
    season_mean['FTR'] = player_df[player_df['FGA'] != 0].groupby('SEASON').mean()['FTR']

    season_mean['TS%'] = player_df[(player_df['FGA'] != 0) & (player_df['FTA'] != 0)].groupby('SEASON').mean()['TS%'] 
    season_mean['EFG%'] = player_df[(player_df['FGA'] != 0) & (player_df['FTA'] != 0)].groupby('SEASON').mean()['EFG%']
    season_mean = season_mean.round(3)
    
    season_mean.fillna(0, inplace=True)
    season_mean.to_csv(f'processed/player/bbref_box/season_mean/{player_name[:-4]}.csv')