import pandas as pd
import os, datetime

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

player_dir = 'REAL_RECOVERED_DATA'
for player in [p for p in os.listdir(player_dir) if p != '.DS_Store']:
    player_path = os.path.join(player_dir, player)
    player_df = pd.read_csv(player_path)
    player_df['SEASON'] = player_df['DATE'].apply(find_season)
    columns = ['SEASON', 'DATE']
    player_df = player_df[columns + [col for col in player_df.columns if col not in columns]]
    player_df.to_csv(f'REAL_RECOVERED_DATA2/{player}.csv', index=False)