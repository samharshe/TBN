### clearly, this was first a Jupyter notebook.
### I wanted to convert everything to .py scripts before relying on them throughout this repo so that they could be more easily recreated, but this was way too messy
### I am exporting to .py strictly for aesthetic continuity here. lots of information is lost in the conversion. sry.

import pandas as pd
import os, warnings, pickle
from tqdm import tqdm
warnings.filterwarnings('ignore')

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

to_process_dir = 'data/player/box_raw'
processed_dir = 'data/player/box_season_mean'
to_process_list = list(set(os.listdir(to_process_dir)) - set(os.listdir(processed_dir)))
n_players = len(to_process_list)
for index, player in enumerate(to_process_list):
    print(f'processing {player}, file {index} of {n_players}')
    path = os.path.join(to_process_dir, player)
    player_df = pd.read_csv(path)
    seasons = player_df['SEASON'].unique()
    means = pd.DataFrame()
    for season in seasons:
        season_df = player_df[player_df['SEASON'] == season]
        season_means_df = get_means(season_df)
        season_means_df['SEASON'] = season
        season_means_df = season_means_df[['SEASON'] + [col for col in season_means_df.columns if col not in ['SEASON']]]
        means = pd.concat([means, season_means_df], ignore_index=True)
    means.to_csv(processed_dir + '/' + player, index=False)

buggy_players = [
    'Dorell Wright',
    'Earl Watson',
    'Jamaal Franklin',
    'Joel Freeland',
    'Jon Leuer',
    'Kosta Koufos',
    'Mike Miller',
    'Al Harrington',
    'Rodney White',
    'Vladimir Radmanović',
    'Nikoloz Tskitishvili',
    'Chris Wilcox',
    'Matt Barnes',
    'Šarūnas Jasikevičius',
    'Tim Thomas',
    'Ruben Patterson',
    'Charles Smith',
    'Malik Allen',
    'Shawn Marion',
    'Antoine Wright',
    'Kirk Snyder',
    'Josh Powell',
    'Bobby Jones',
    'Stephen Jackson',
    'Mike James',
    'Jason Kidd'
]

buggy_players = [
    'Sam Cassell',
    'Tony Mitchell 1',
    'Tony Mitchell 2',
]

raw_dir = 'data/player/box_raw'
processed_dir = 'data/player/box_season_mean'
n_players = len(buggy_players)
for index, player in enumerate(buggy_players):
    print(f'processing {player}, file {index} of {n_players}')
    path = os.path.join(raw_dir, player +'.csv')
    player_df = pd.read_csv(path)
    seasons = player_df['SEASON'].unique()
    means = pd.DataFrame()
    for season in seasons:
        season_df = player_df[player_df['SEASON'] == season]
        season_means_df = get_means(season_df)
        season_means_df['SEASON'] = season
        season_means_df = season_means_df[['SEASON'] + [col for col in season_means_df.columns if col not in ['SEASON']]]
        means = pd.concat([means, season_means_df], ignore_index=True)
    means.to_csv(processed_dir + '/' + player + '.csv', index=False)

buggy_players = [
    'Dorell Wright',
    'Earl Watson',
    'Jamaal Franklin',
    'Joel Freeland',
    'Jon Leuer',
    'Kosta Koufos',
    'Mike Miller',
    'Al Harrington',
    'Rodney White',
    'Vladimir Radmanović',
    'Nikoloz Tskitishvili',
    'Chris Wilcox',
    'Matt Barnes',
    'Šarūnas Jasikevičius',
    'Tim Thomas',
    'Ruben Patterson',
    'Charles Smith',
    'Malik Allen',
    'Shawn Marion',
    'Antoine Wright',
    'Kirk Snyder',
    'Josh Powell',
    'Bobby Jones',
    'Stephen Jackson',
    'Mike James',
    'Jason Kidd',
    'Marcus Williams 2',
    'Marcus Williams 1',
    'Brandon Williams 1',
    'Brandon Williams 2',
    'Chris Johnson 2',
    'Patrick Ewing 1',
    'Tony Mitchell 2',
    'Patrick Ewing 2',
    'Chris Johnson 1',
    'Tony Mitchell 1',
]

players_dir = 'data/player/box_raw'
n_players = len(buggy_players)
for index, player_name in tqdm(enumerate(buggy_players)):
    print(f'processing {player_name}, file {index} of {n_players}')
    path = os.path.join(players_dir, player_name + '.csv')
    player_df = pd.read_csv(path)
    seasons = player_df['SEASON'].unique()
    
    first_season = seasons[0]
    first_season_df = player_df[player_df['SEASON'] == first_season]
    for index, row in first_season_df.iterrows():
        first_season_to_date_df = first_season_df.iloc[:max(index, 0)]
        first_season_means_df = get_means(first_season_to_date_df)
        player_df.loc[index, first_season_means_df.columns] = first_season_means_df.iloc[0]
    
    season_means_df = pd.read_csv(f'data/player/box_season_mean/{player_name}.csv')
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
    
    player_df.to_csv(f'data/player/box_weighted_average/{player_name}.csv', index=False)

def get_means(df):
    df.drop(columns=['SEASON', 'DATE', 'OPPONENT', 'HOME_TEAM'], inplace=True)
    empty_row = pd.DataFrame([0] * len(df.columns), index=df.columns).T
    df = (pd.concat([df, empty_row], ignore_index=True)).fillna(0)
    df = df.mean().to_frame().T
    df['FG%'] = df['FGM'] / df['FGA']
    df['2PT_FG%'] = df['2PT_FGM'] / df['2PT_FGA']
    df['3PT_FG%'] = df['3PT_FGM'] / df['3PT_FGA']
    df['FT%'] = df['FTM'] / df['FTA']
    df['TS%'] = df['PTS'] / (2*(df['FGA'] + 0.44*df['FTA']))
    df['EFG%'] = (df['FGM'] + (0.5*df['3PT_FGM'])) / (df['FGA'])
    df['3PAR'] = df['3PT_FGA'] / df['FGA']
    df['FTR'] = df['FTM'] / df['FGA']
    df = df.fillna(0)
    df = df.round(3)
    return df

to_process_dir = 'data/team/box_raw'
processed_dir = 'data/team/box_season_mean'
code = 'JAZ'
print(f'processing {code}')
path = os.path.join(to_process_dir, code + '.csv')
team_df = pd.read_csv(path)
seasons = team_df['SEASON'].unique()
means = pd.DataFrame()
for season in seasons:
    season_df = team_df[team_df['SEASON'] == season]
    season_means_df = get_means(season_df)
    season_means_df['SEASON'] = season
    season_means_df = season_means_df[['SEASON'] + [col for col in season_means_df.columns if col not in ['SEASON']]]
    means = pd.concat([means, season_means_df], ignore_index=True)
means.to_csv(processed_dir + '/' + code + '.csv', index=False)

print(f'processing {code}')
path = os.path.join(to_process_dir, code + '.csv')
team_df = pd.read_csv(path)
seasons = team_df['SEASON'].unique()

first_season = seasons[0]
first_season_df = player_df[player_df['SEASON'] == first_season]
for index, row in first_season_df.iterrows():
    first_season_to_date_df = first_season_df.iloc[:max(index, 0)]
    first_season_means_df = get_means(first_season_to_date_df)
    player_df.loc[index, first_season_means_df.columns] = first_season_means_df.iloc[0]

season_means_df = pd.read_csv(f'data/team/box_season_mean/{code}.csv')
seasons = season_means_df['SEASON'].unique()
for season in seasons[1:]:
    current_season_df = team_df[team_df['SEASON'] == season]
    previous_season_means_df = season_means_df.iloc[season_means_df.index[season_means_df['SEASON'] == season] - 1].drop(columns=['SEASON'])
    for index, row in current_season_df.iterrows():
        current_season_to_date_df = current_season_df.loc[:max(index, 0)]
        current_season_means_df = get_means(current_season_to_date_df)
        
        n_games = current_season_to_date_df.shape[0]
        
        weighted_average_df = pd.DataFrame((20*previous_season_means_df.values + n_games*current_season_means_df.values)/(20+n_games)).round(3)
        weighted_average_df.columns = current_season_means_df.columns
        
        player_df.loc[index, weighted_average_df.columns] = weighted_average_df.iloc[0]

team_df.to_csv('data/team/box_weighted_average/' + code, index=False)

team_player_dict = pickle.load(open('data/team_player_dict.pkl', 'rb'))

for player in buggy_players:
    player_df = pd.read_csv(f'data/player/box_raw/{player}.csv')
    for date in player_df['DATE'].unique():
        team = player_df[player_df['DATE'] == date]['TEAM'].iloc[0]