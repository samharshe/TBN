import os, warnings
warnings.filterwarnings('ignore')
import pandas as pd

def get_means(df):
    df.drop(columns=['SEASON', 'DATE', 'TEAM', 'OPPONENT', 'HOME_TEAM'], inplace=True)
    empty_row = pd.DataFrame([0] * len(df.columns), index=df.columns).T
    df = (pd.concat([df, empty_row], ignore_index=True)).fillna(0)
    df_play = df[df['MP'] > 0]
    df_play = df_play.mean()
    df = df.mean()
    df['FG%'] = df['FGM'] / df['FGA']
    df['2PT_FG%'] = df['2PT_FGM'] / df['2PT_FGA']
    df['3PT_FG%'] = df['3PT_FGM'] / df['3PT_FGA']
    df['FT%'] = df['FTM'] / df['FTA']
    df['TS%'] = df['PTS'] / (2*(df['FGA'] + 0.44*df['FTA']))
    df['EFG%'] = (df['FGM'] + (0.5*df['3PT_FGM'])) / (df['FGA'])
    df['3PAR'] = df['3PT_FGA'] / df['FGA']
    df['FTR'] = df['FTM'] / df['FGA']
    play_columns = ['ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'ORTG', 'DRTG', 'BPM']
    df[play_columns] = df_play[play_columns]
    df.fillna(0, inplace=True)
    df = df.round(3)
    df = df.to_frame()
    df = df.T
    return df
    
player_dir = 'data/raw/player'
player_dir_list = os.listdir(player_dir)
n_players = len(player_dir_list)
for index, player in enumerate(player_dir_list):
    print(f'processing {player}, file {index} of {n_players}')
    path = os.path.join(player_dir, player)
    player_df = pd.read_csv(path)
    seasons = player_df['SEASON'].unique()
    means = pd.DataFrame()
    for season in seasons:
        season_df = player_df[player_df['SEASON'] == season]
        season_means_df = get_means(season_df)
        season_means_df['SEASON'] = season
        season_means_df = season_means_df[['SEASON'] + [col for col in season_means_df.columns if col not in ['SEASON']]]
        means = pd.concat([means, season_means_df], ignore_index=True)
    means.to_csv('data/processed/player/bbref_box/season_mean/' + player, index=False)