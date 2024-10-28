import pandas as pd

games_df = pd.read_csv('data/game/bbref_game.csv')

playoff_start_dates = [
    '2024-04-20',
    '2023-04-15', 
    '2022-04-16',
    '2021-05-22',
    '2020-08-18',
    '2019-04-13',
    '2018-04-14',
    '2017-04-15',
    '2016-04-16',
    '2015-04-18',
    '2014-04-19',
    '2013-04-20',
    '2012-04-28',
    '2011-04-16',
    '2010-04-17',
    '2009-04-18',
    '2008-04-19',
    '2007-04-21',
    '2006-04-22',
    '2005-04-23',
    '2004-04-17',
    '2003-04-19',
    '2002-04-20',
    '2001-04-21',
    '2000-04-22'
]

# Convert DATE column to datetime for comparison
games_df['DATE'] = pd.to_datetime(games_df['DATE'])

# Create a mapping of season to playoff start date
playoff_dates = {}
for date in playoff_start_dates:
    year = int(date[:4])
    # Map to season format (e.g. 2024 -> '2023-24')
    season = f'{year-1}-{str(year)[2:]}' 
    playoff_dates[season] = pd.to_datetime(date)

# Determine game type based on date comparison
def get_game_type(row):
    playoff_date = playoff_dates[row['SEASON']]
    return 'playoff' if row['DATE'] >= playoff_date else 'regular_season'

games_df['GAME_TYPE'] = games_df.apply(get_game_type, axis=1)
games_df.to_csv('data/game/bbref_game_with_game_type.csv', index=False)
