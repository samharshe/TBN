import os, datetime
import pandas as pd
import datetime
from datetime import timedelta, datetime

def num_games_in_previous_n_days(date, team, n):
    season = get_season(date)
    date = datetime.strptime(date, '%Y-%m-%d') # to do subtraction with timedelta, which is easier
    games_in_previous_n_days = games_df[((games_df['HOME_TEAM'] == team) | (games_df['AWAY_TEAM'] == team)) & ((str(date - timedelta(days=n)) < games_df['DATE']) & (games_df['DATE'] <= str(date)))]
    return len(games_in_previous_n_days)

def get_rest(date, team):
    season = get_season(date)
    if num_games_in_previous_n_days(date, team, 5) == 4:
        return('4in5')
    elif num_games_in_previous_n_days(date, team, 4) == 3:
        if num_games_in_previous_n_days(date, team, 2) == 2:
            return '3in4_2in2'
        else:
            return '3in4_1in2'
    elif num_games_in_previous_n_days(date, team, 2) == 2:
        return '2in2'
    elif num_games_in_previous_n_days(date, team, 3) == 2:
        return '2in3'
    elif num_games_in_previous_n_days(date, team, 4) == 2:
        return '2in4'
    else:
        return '2in5+'
    
def get_game_number(date, team):
    season = get_season(date)
    date = datetime.strptime(date, '%Y-%m-%d')
    team_games = len(games_df[(games_df['SEASON'] == season) & ((games_df['HOME_TEAM'] == team) | (games_df['AWAY_TEAM'] == team)) & (games_df['DATE'] < str(date))])
    return team_games

def get_season(date):
    date = datetime.strptime(date, '%Y-%m-%d')

    if date <= datetime.strptime(f"{date.year}-10-11", '%Y-%m-%d'):
        return f"{date.year-1}-{str(date.year)[2:]}"
    else:
        return f"{date.year}-{str(date.year+1)[2:]}"

def get_team_dfs_dict():
    team_dfs_dict = {}
    data_dir = 'data/team/raw/bbref_box/'
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            team_name = filename[:-4]
            
            file_path = os.path.join(data_dir, filename)
            
            df = pd.read_csv(file_path)
            
            team_dfs_dict[team_name] = df
            
    print(f'loaded dict for all {len(team_dfs_dict)} teams.')
    
    return team_dfs_dict
    
def get_games_df(team_dfs_dict):
    game_data = []
    for team, df in team_dfs_dict.items():
        for _, row in df.iterrows():
            game = {}
            game['DATE'] = row['DATE']
            opponent_df = team_dfs_dict[row['OPPONENT']]
            opponent_row = opponent_df[opponent_df['DATE'] == game['DATE']].iloc[0]
            
            if row['HOME_TEAM'] == team:
                prefix = 'HOME_'
                opponent_prefix = 'AWAY_'
            else:
                prefix = 'AWAY_'
                opponent_prefix = 'HOME_'
            
            for col in df.columns:
                if col not in ['DATE', 'OPPONENT', 'HOME_TEAM']:
                    game[f'{prefix}{col}'] = row[col]
                    game[f'{opponent_prefix}{col}'] = opponent_row[col]
                    
            game[f'{opponent_prefix}TEAM'] = row['OPPONENT']
            game[f'{prefix}TEAM'] = team
            
            game_data.append(game)
            
    games_df = pd.DataFrame(game_data)

    games_df = games_df.sort_values('DATE')
    games_df = games_df.drop_duplicates()
    games_df = games_df.reset_index(drop=True)
    
    print(f'combined all team data into single DataFrame of length {len(games_df)}.')
    
    return games_df
    
def main():
    team_dfs_dict = get_team_dfs_dict()
    global games_df
    games_df = get_games_df(team_dfs_dict)

    games_df['SEASON'] = games_df['DATE'].apply(get_season)
    print('added SEASON column.')

    for team in ["HOME", "AWAY"]:
        games_df[f'{team}_GAME_NUMBER'] = games_df.apply(lambda row: get_game_number(row['DATE'], row[f'{team}_TEAM']), axis=1)
        print(f'added {team}_GAME_NUMBER column.')
        games_df[f'{team}_REST'] = games_df.apply(lambda row: get_rest(row['DATE'], row[f'{team}_TEAM']), axis=1)
        print(f'added {team}_REST column.')

    columns = ['SEASON', 'DATE', 'HOME_TEAM', 'AWAY_TEAM', 'HOME_PTS', 'AWAY_PTS']
    games_df = games_df[columns + [col for col in games_df.columns if col not in columns]]
    print('reordered columns.')
    
    games_df.to_csv('data/team/preprocessed/bbref_box/bbref_box2.csv', index=False)
    print('saved to CSV.')
    
if __name__ == '__main__':
    main()