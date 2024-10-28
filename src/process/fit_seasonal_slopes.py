import pandas as pd

df = pd.read_csv('data/game/box_raw/reshaped_regular_season_games.csv')
df = df[df['SEASON'] != '2019-20']
new_df = df.copy()

for year in new_df['SEASON'].unique():
    year_df = new_df[new_df['SEASON'] == year]
    for stat in ['PTS', 'PACE', 'FGM', 'FGA', '3PT_FGM', '3PT_FGA', 'FTM', 'FTA', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', '3PAR', 'FTR', 'ORTG', 'DRTG', '2PT_FGM', '2PT_FGA']:
        # For each stat, fit a line to see if game number predicts the stat value
        X = year_df['GAME_NUMBER'].values.reshape(-1, 1)
        y = year_df[stat].values
        
        # fit linear regression
        m = ((X.ravel() * y).mean() - X.mean() * y.mean()) / ((X.ravel()**2).mean() - X.mean()**2)
        print(f'correlation of GAME_NUMBER and {stat} in {year}: {m}')
        
        # Adjust the stat by subtracting the line to remove correlation with game number
        adjusted = y - (m * X.ravel())
        
        # Store back in dataframe 
        new_df.loc[(df['SEASON'] == year), stat] = adjusted

new_df.to_csv('data/game/box_raw/reshaped_regular_season_games_adjusted.csv', index=False)