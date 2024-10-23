import os, warnings, datetime
warnings.filterwarnings('ignore')
import pandas as pd

def find_season(input_date: str):
    input_date = datetime.datetime.strptime(input_date, '%Y-%m-%d')
    season_end_dates = {
        '2000-01': '2001-06-15',
        '2001-02': '2002-06-12',
        '2002-03': '2003-06-15',
        '2003-04': '2004-06-15',
        '2004-05': '2005-06-23',
        '2005-06': '2006-06-20',
        '2006-07': '2007-06-14',
        '2007-08': '2008-06-17',
        '2008-09': '2009-06-14',
        '2009-10': '2010-06-17',
        '2010-11': '2011-06-12',
        '2011-12': '2012-06-21',
        '2012-13': '2013-06-20',
        '2013-14': '2014-06-15',
        '2014-15': '2015-06-16',
        '2015-16': '2016-06-19',
        '2016-17': '2017-06-12',
        '2017-18': '2018-06-08',
        '2018-19': '2019-06-13',
        '2019-20': '2020-10-11',
        '2020-21': '2021-07-20',
        '2021-22': '2022-06-16',
        '2022-23': '2023-06-12',
        '2023-24': '2024-06-17'
    }
    for key, value in season_end_dates.items():
        if datetime.datetime.strptime(value, '%Y-%m-%d') >= input_date:
            return key
    raise ValueError(f'no season end date found for {input_date}')

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

teams_dir = 'raw/team/bbref_team'

for team_name in os.listdir(teams_dir):
    print(f'processing {team_name}')
    path = os.path.join(teams_dir, team_name)
    team_df = pd.read_csv(path)
    final_df = pd.DataFrame()
    
    for season in team_df['SEASON'].unique():
        print(f'season {season}')
        current_season_df = team_df[team_df['SEASON'] == season]
        current_season_means_df = get_means(current_season_df)
        current_season_means_df['SEASON'] = season
        cols = ['SEASON'] + [col for col in current_season_means_df.columns if col != 'SEASON']
        current_season_means_df = current_season_means_df[cols]
        final_df = pd.concat([final_df, current_season_means_df], ignore_index=True)
    
    final_df.to_csv(f'processed/team/bbref_box/season_mean/{team_name}', index=False)

teams_dir = 'raw/team/bbref_team'
for team_name in os.listdir(teams_dir):
    print(f'processing {team_name}')
    path = os.path.join(teams_dir, team_name)
    team_df = pd.read_csv(path)
    seasons = team_df['SEASON'].unique()
    
    first_season = seasons[0]
    first_season_df = team_df[team_df['SEASON'] == first_season]
    for index, row in first_season_df.iterrows():
        first_season_to_date_df = first_season_df.iloc[:max(index, 0)]
        first_season_means_df = get_means(first_season_to_date_df)
        team_df.loc[index, first_season_means_df.columns] = first_season_means_df.iloc[0]
    
    season_means_df = pd.read_csv(f'processed/team/bbref_box/season_mean/{team_name}')
    for season in seasons[1:]:
        current_season_df = team_df[team_df['SEASON'] == season]
        previous_season_means_df = season_means_df.iloc[season_means_df.index[season_means_df['SEASON'] == season] - 1].drop(columns=['SEASON'])
        for index, row in current_season_df.iterrows():
            current_season_to_date_df = current_season_df.loc[:max(index, 0)]
            current_season_means_df = get_means(current_season_to_date_df)
            
            n_games = current_season_to_date_df.shape[0]
            
            weighted_average_df = pd.DataFrame((20*previous_season_means_df.values + n_games*current_season_means_df.values)/(20+n_games)).round(3)
            weighted_average_df.columns = current_season_means_df.columns
            
            team_df.loc[index, weighted_average_df.columns] = weighted_average_df.iloc[0]
    
    team_df.to_csv('processed/team/bbref_box/weighted_average/' + team_name, index=False)