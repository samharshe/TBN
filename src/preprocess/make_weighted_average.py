import os, warnings
warnings.filterwarnings('ignore')
import pandas as pd
from tqdm import tqdm

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
    
players_dir = 'data/raw/player'
players_dir_list = os.listdir(players_dir)
n_players = len(players_dir_list)
for index, player_name in tqdm(enumerate(players_dir_list)):
    print(f'processing {player_name}, file {index} of {n_players}')
    path = os.path.join(players_dir, player_name)
    player_df = pd.read_csv(path)
    seasons = player_df['SEASON'].unique()
    
    first_season = seasons[0]
    first_season_df = player_df[player_df['SEASON'] == first_season]
    for index, row in first_season_df.iterrows():
        first_season_to_date_df = first_season_df.iloc[:max(index, 0)]
        first_season_means_df = get_means(first_season_to_date_df)
        player_df.loc[index, first_season_means_df.columns] = first_season_means_df.iloc[0]
    
    season_means_df = pd.read_csv(f'data/processed/player/bbref_box/season_mean/{player_name}')
    for season in seasons[1:]:
        current_season_df = player_df[player_df['SEASON'] == season]
        previous_season_means_df = season_means_df.iloc[season_means_df.index[season_means_df['SEASON'] == season] - 1].drop(columns=['SEASON'])
        for index, row in current_season_df.iterrows():
            current_season_to_date_df = current_season_df.loc[:max(index, 0)]
            current_season_means_df = get_means(current_season_to_date_df)
            
            n_games = current_season_to_date_df.shape[0]
            
            weighted_average_df = pd.DataFrame((20*previous_season_means_df.values + n_games*current_season_means_df.values)/(20+n_games)).round(3)
            weighted_average_df.columns = current_season_means_df.columns
            
            player_df.loc[index, weighted_average_df.columns] = weighted_average_df.iloc[0]
    
    player_df.to_csv('data/processed/player/bbref_box/weighted_average/' + player_name, index=False)