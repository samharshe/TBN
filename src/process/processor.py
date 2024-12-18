import numpy as np
import pandas as pd

STATS_TO_ADJUST = ['PTS', 'PACE', 'FGM', 'FGA', '3PT_FGM', '3PT_FGA', 'FTM', 'FTA', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'ORTG', 'DRTG', '2PT_FGM', '2PT_FGA']

class BaseModel:
    """base class similar to torch.nn.Module but for Bayesian models"""
    def __init__(self):
        self.models = [
            FitSeasonalSlopes(),
            HomeAdjustment(),
            RestAdjustment(),
            OpponentAdjustment(),
        ]

    def forward(self, df):
        new_df = df.copy()
        for model in self.models:
            new_df = model.forward(new_df)
        
        return new_df

    def backward(self, df):
        new_df = df.copy()
        for model in self.models[::-1]:
            new_df = model.backward(new_df)
        
        return new_df


class FitSeasonalSlopes():
    def __init__(self):
        super().__init__()
        self.ms = None

    def get_parameters(self):
        return self.ms
        
    def forward(self, df):
        ms = {}
        for season in df['SEASON'].unique():
            print(f'fitting seasonal slopes for {season}')
            ms[season] = {}
            season_df = df[df['SEASON'] == season]
            for stat in STATS_TO_ADJUST:
                X = season_df['GAME_NUMBER'].values.reshape(-1, 1)
                y = season_df[stat].values
                m = ((X.ravel() * y).mean() - X.mean() * y.mean()) / ((X.ravel()**2).mean() - X.mean()**2)
                ms[season][stat] = m

        new_df = df.copy()
        for season in df['SEASON'].unique():
            season_df = df[df['SEASON'] == season]
            for stat in STATS_TO_ADJUST:
                X = season_df['GAME_NUMBER'].values.reshape(-1, 1)
                y = season_df[stat].values
                adjusted = y - (ms[season][stat] * X.ravel())
                new_df.loc[(df['SEASON'] == season), stat] = adjusted

        self.ms = ms
    
        new_df = new_df.round(5)
        return new_df

    def backward(self, full_df):
        print('applying seasonal slopes backward')

        ms = self.ms
        new_full_df = full_df.copy()


        def backward_row(row):
            game_number = row['GAME_NUMBER']
            season = row['SEASON']
            for stat in STATS_TO_ADJUST:
                unadjusted_stat = row[stat]
                adjusted_stat = unadjusted_stat + (ms[season][stat] * game_number)
                row[stat] = adjusted_stat
            return row

        new_full_df = new_full_df.apply(backward_row, axis=1)

        new_full_df = new_full_df.round(5)
        return new_full_df

class HomeAdjustment():
    def __init__(self):
        super().__init__()
        self.home_advantages = {}

    def forward(self, df):
        new_df = df.copy()

        for season in new_df['SEASON'].unique():
            self.home_advantages[season] = {}
            print(f'calculating home adjustment for {season}')
            season_df = new_df[new_df['SEASON'] == season]
            home_mean_df = season_df[season_df['IS_HOME'] == 1][['TEAM'] + STATS_TO_ADJUST].groupby('TEAM').mean()
            mean_df = season_df[['TEAM'] + STATS_TO_ADJUST].groupby('TEAM').mean()
            diff_df = (home_mean_df[STATS_TO_ADJUST] - mean_df[STATS_TO_ADJUST]).reset_index()
            diff_df = james_stein_shrinkage(df=diff_df, cols=STATS_TO_ADJUST)
            self.home_advantages[season] = diff_df

            def forward_row(row):
                home_advantage = diff_df[diff_df['TEAM'] == row['TEAM']][STATS_TO_ADJUST].iloc[0]
                adjustment = 2 * (float(row['IS_HOME']) - 0.5) * home_advantage
                adjusted_game = row[STATS_TO_ADJUST] - adjustment
                for stat in STATS_TO_ADJUST:
                    row[stat] = adjusted_game[stat]
                return row

            season_df = season_df.apply(forward_row, axis=1)
            new_df.loc[new_df['SEASON'] == season, STATS_TO_ADJUST] = season_df[STATS_TO_ADJUST]

        return new_df
     
    def backward(self, df):
        print('applying home adjustment backward')

        new_df = df.copy()
        home_advantages = self.home_advantages

        def backward_row(row):
            season = row['SEASON']
            team = row['TEAM']
            diff_df = home_advantages[season]
            home_advantage = diff_df[diff_df['TEAM'] == team][STATS_TO_ADJUST].iloc[0]
            adjustment = 2 * (float(row['IS_HOME']) - 0.5) * home_advantage
            adjusted_game = row[STATS_TO_ADJUST] + adjustment
            for stat in STATS_TO_ADJUST:
                row[stat] = adjusted_game[stat]
            return row

        new_df = new_df.apply(backward_row, axis=1)

        return new_df

class RestAdjustment():
    def __init__(self):
        super().__init__()
        self.rest_adjustments = {}
        
    def forward(self, df):
        new_df = df.copy()

        for season in new_df['SEASON'].unique():
            print(f'calculating rest adjustment for {season}')
            self.rest_adjustments[season] = {}
            season_df = new_df[new_df['SEASON'] == season]

            for rest in new_df['REST'].unique():
                print(f'calculating rest adjustment for {rest}')
                rest_df = season_df[season_df['REST'] == rest][['TEAM'] + STATS_TO_ADJUST].groupby('TEAM').mean()
                mean_df = season_df[['TEAM'] + STATS_TO_ADJUST].groupby('TEAM').mean()

                diff_df = (rest_df[STATS_TO_ADJUST] - mean_df[STATS_TO_ADJUST]).reset_index() # diff_df gives improvement relative to mean for a given rest value
                diff_df = epd_shrinkage(df=diff_df, cols=STATS_TO_ADJUST, p=0.01) # shrunken toward the league mean

                self.rest_adjustments[season][rest] = diff_df
                def forward_row(row):
                    for stat in STATS_TO_ADJUST:
                        row[stat] = row[stat] - diff_df[diff_df['TEAM'] == row['TEAM']][stat].iloc[0]
                    return row
                
                row_mask = season_df['REST'] == rest
                season_df.loc[row_mask] = season_df[row_mask].apply(forward_row, axis=1)

            new_df.loc[new_df['SEASON'] == season, STATS_TO_ADJUST] = season_df[STATS_TO_ADJUST]

        return new_df
    
    def backward(self, df):
        print('applying rest adjustment backward')

        new_df = df.copy()
        rest_adjustments = self.rest_adjustments

        def backward_row(row):
            diff_df = rest_adjustments[row['SEASON']][row['REST']]

            for stat in STATS_TO_ADJUST:
                row[stat] = row[stat] + diff_df[diff_df['TEAM'] == row['TEAM']][stat].iloc[0]

            return row

        new_df = new_df.apply(backward_row, axis=1)

        return new_df

class OpponentAdjustment():
    def __init__(self):
        super().__init__()
        self.opponent_effects = {}

    def forward(self, df):
        new_df = df.copy()
        
        for season in new_df['SEASON'].unique():
            print(f'calculating opponent adjustment for {season}')
            self.opponent_effects[season] = pd.DataFrame(0, index=new_df[new_df['SEASON'] == season]['OPPONENT'].unique(), columns=STATS_TO_ADJUST)
            iterations = 0
            while True:
                season_df = new_df[new_df['SEASON'] == season].copy()
                teams = list(season_df['TEAM'].unique())
                opponent_excluded_dict = {}
                for team in teams:
                    opponent_excluded_dict[team] = {}
                    for opponent in teams:
                        if team != opponent:
                            opponent_excluded_dict[team][opponent] = season_df[(season_df['TEAM'] == team) & (season_df['OPPONENT'] != opponent)][STATS_TO_ADJUST].mean()

                relative_performance = season_df.copy()
                def relative_performance_row(row):
                    team = row['TEAM']
                    opponent = row['OPPONENT']
                    game_relative_performance = row[STATS_TO_ADJUST] - opponent_excluded_dict[team][opponent]
                    for stat in STATS_TO_ADJUST:
                        row[stat] = game_relative_performance[stat]
                    return row

                relative_performance = relative_performance.apply(relative_performance_row, axis=1)

                opponent_effect = relative_performance.groupby('OPPONENT')[STATS_TO_ADJUST].mean()
                opponent_effect = james_stein_shrinkage(opponent_effect, STATS_TO_ADJUST)
                self.opponent_effects[season] += opponent_effect
                opponent_effect_magnitude = np.sqrt((opponent_effect ** 2).sum(axis=1)).mean()
                print(f'{season} opponent effect magnitude for iteration {iterations}: {opponent_effect_magnitude}')

                if opponent_effect_magnitude < 0.1 or iterations > 10:
                    break

                def forward_row(row):
                    opponent = row['OPPONENT']
                    adjustment = opponent_effect.loc[opponent]
                    adjusted_game = row[STATS_TO_ADJUST] - adjustment
                    for stat in STATS_TO_ADJUST:
                        row[stat] = adjusted_game[stat]
                    return row

                season_df = season_df.apply(forward_row, axis=1)
                new_df.loc[new_df['SEASON'] == season, STATS_TO_ADJUST] = season_df[STATS_TO_ADJUST]

                iterations += 1

        return new_df
    
    def backward(self, df):
        print('applying opponent adjustment backward')

        new_df = df.copy()
        opponent_effects = self.opponent_effects

        def backward_row(row):
            season = row['SEASON']
            opponent = row['OPPONENT']
            adjustment = opponent_effects[season].loc[opponent]
            adjusted_game = row[STATS_TO_ADJUST] + adjustment
            for stat in STATS_TO_ADJUST:
                row[stat] = adjusted_game[stat]
            return row

        new_df = new_df.apply(backward_row, axis=1)

        return new_df

def james_stein_shrinkage(df, cols):
    new_df = df.copy()

    for col in cols:
        values = new_df[col].values
        n = len(values)
        mean = values.mean()
        var = values.var(ddof=1)
        shrinkage = max(0, 1 - (n-3) * var / ((values - mean)**2).sum())
        new_df[col] = mean + (1 - shrinkage) * (values - mean)
        
    return new_df

def epd_shrinkage(df, cols, p=0.5):
    new_df = df.copy()

    for col in cols:
        values = new_df[col].values
        column_mean = np.mean(values)
        column_scale = np.std(values)
        
        deviations = np.abs(values - column_mean)
        weights = 1 / (1 + (deviations/column_scale)**(p-1))
        
        shrunk_values = column_mean + weights * (values - column_mean)
        
        new_df[col] = shrunk_values
    
    return new_df

def make_mean_filled_df(df):
    new_df = df.copy()
    mean_df = df.groupby(['SEASON', 'TEAM'])[STATS_TO_ADJUST].mean()

    def fill_mean(row):
        for col in STATS_TO_ADJUST:
            row[col] = mean_df.loc[(row['SEASON'], row['TEAM']), col]
        return row
    
    new_df = new_df.apply(fill_mean, axis=1)
    
    return new_df

def make_training_df(processed_df, original_df):
    new_df = processed_df.copy()

    def add_y(row):
        y = original_df[(original_df['DATE'] == row['DATE']) & (original_df['TEAM'] == row['TEAM'])]['PTS'].values[0]
        row['Y'] = y
        return row
    
    new_df = new_df.apply(add_y, axis=1)

    halved_df = pd.DataFrame()
    row_list = []

    for i in range(0, len(new_df), 2):
        home_row = new_df.iloc[i]
        away_row = new_df.iloc[i+1]

        season = home_row['SEASON']
        date = home_row['DATE']

        cols_to_exclude = ['SEASON', 'DATE', 'OPPONENT', 'IS_HOME']
        home_row = home_row[home_row.index.difference(cols_to_exclude)]
        away_row = away_row[away_row.index.difference(cols_to_exclude)]

        home_row.index = [f'HOME_{col}' for col in home_row.index]
        away_row.index = [f'AWAY_{col}' for col in away_row.index]

        combined_row = pd.Series()
        combined_row['SEASON'] = season
        combined_row['DATE'] = date
        combined_row['HOME_WIN'] = home_row['HOME_Y'] > away_row['AWAY_Y']
        combined_row = pd.concat([combined_row, home_row, away_row])
        row_list.append(combined_row)
        
    halved_df = pd.concat(row_list, axis=1).T

    return halved_df