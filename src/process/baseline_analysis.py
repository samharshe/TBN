import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

def reshape_data() -> pd.DataFrame:
    df = pd.read_csv('data/game/box_raw/bbref_game_counts.csv')
    home_games, away_games = pd.DataFrame(), pd.DataFrame()

    home_games = df[[col for col in df.columns if 'HOME_' in col]]
    home_games = home_games.rename(columns={
        col: col.replace('HOME_', '') for col in home_games.columns if 'HOME_' in col
    })
    home_games[['REST', 'OPPONENT', 'SEASON', 'DATE', 'GAME_TYPE']] = df[['HOME_REST', 'AWAY_TEAM', 'SEASON', 'DATE', 'GAME_TYPE']]
    home_games['IS_HOME'] = 1

    away_games = df[[col for col in df.columns if 'AWAY_' in col]]
    away_games = away_games.rename(columns={
        col: col.replace('AWAY_', '') for col in away_games.columns if 'AWAY_' in col
    })
    away_games[['REST', 'OPPONENT', 'SEASON', 'DATE', 'GAME_TYPE']] = df[['AWAY_REST', 'HOME_TEAM', 'SEASON', 'DATE', 'GAME_TYPE']]
    away_games['IS_HOME'] = 0

    all_games = pd.concat([home_games, away_games])
    all_games.sort_values(['SEASON', 'DATE', 'TEAM'], inplace=True)

    return all_games

def analyze_rest_effects(df, stat='PTS'):
    """
    Analyze how rest affects team performance
    """
    print(f"\n=== Rest Effects Analysis for {stat} ===\n")
    
    # Create long-format dataset for this statistic
    data = df.copy()
    # Basic statistics by rest
    rest_stats = data.groupby(['REST', 'IS_HOME'])[stat].agg(['count', 'mean', 'std'])
    print("Statistics by rest days and home/away:")
    print(rest_stats)
    
    # ANOVA test
    for is_home in [0, 1]:
        subset = data[data['IS_HOME'] == is_home]
        f_stat, p_value = stats.f_oneway(*[group[stat].values 
                                         for name, group in subset.groupby('REST')])
        location = "home" if is_home else "away"
        print(f"\nANOVA test for {location} games:")
        print(f"F-statistic: {f_stat:.3f}")
        print(f"p-value: {p_value:.3e}")
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Box plot by rest
    sns.boxplot(data=data, x='REST', y=stat, ax=axes[0,0])
    axes[0,0].set_title(f'{stat} by Rest Days')
    
    # 2. Box plot by rest and home/away
    sns.boxplot(data=data, x='REST', y=stat, hue='IS_HOME', ax=axes[0,1])
    axes[0,1].set_title(f'{stat} by Rest Days and Home/Away')
    
    # 3. Mean and confidence intervals
    rest_means = data.groupby('REST')[stat].agg(['mean', 'std', 'count'])
    rest_means['ci'] = 1.96 * rest_means['std'] / np.sqrt(rest_means['count'])
    rest_means['mean'].plot(kind='bar', yerr=rest_means['ci'], ax=axes[1,0])
    axes[1,0].set_title(f'Mean {stat} by Rest Days (with 95% CI)')
    
    # 4. Density plot by rest
    for rest_days in sorted(data['REST'].unique()):
        sns.kdeplot(data=data[data['REST']==rest_days][stat], 
                   label=f'Rest {rest_days}', ax=axes[1,1])
    axes[1,1].set_title(f'{stat} Distribution by Rest Days')
    
    plt.tight_layout()
    plt.show()
    
    return rest_stats

def analyze_temporal_trends(df, stat='PTS'):
    """
    Analyze how rest effects change over time
    """
    print(f"\n=== Temporal Trends Analysis for {stat} ===\n")
    
    data = df.copy()
    
    # League-wide trends
    yearly_stats = data.groupby('SEASON')[stat].agg(['mean', 'std'])
    print("\nYearly league averages:")
    print(yearly_stats)
    
    # Rest effects by season
    rest_by_season = data.groupby(['SEASON', 'REST'])[stat].mean().unstack()
    print("\nRest effects by season:")
    print(rest_by_season)
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. League average over time
    yearly_stats['mean'].plot(ax=axes[0,0])
    axes[0,0].set_title(f'League Average {stat} Over Time')
    axes[0,0].fill_between(yearly_stats.index,
                          yearly_stats['mean'] - yearly_stats['std'],
                          yearly_stats['mean'] + yearly_stats['std'],
                          alpha=0.2)
    
    # 2. Rest effects over time
    rest_by_season.plot(ax=axes[0,1])
    axes[0,1].set_title('Rest Effects Over Time')
    
    # 3. Home advantage over time
    home_adv = data.groupby(['SEASON', 'IS_HOME'])[stat].mean().unstack()
    home_adv[1].subtract(home_adv[0]).plot(ax=axes[1,0])
    axes[1,0].set_title('Home Court Advantage Over Time')
    
    # 4. Rest effect sizes over time
    baseline_rest = rest_by_season.iloc[:,0]  # using first rest category as baseline
    for rest_cat in rest_by_season.columns[1:]:
        effect_size = (rest_by_season[rest_cat] - baseline_rest) / \
                     data.groupby('SEASON')[stat].std()
        effect_size.plot(ax=axes[1,1], label=f'Rest {rest_cat}')
    axes[1,1].set_title('Rest Effect Sizes Over Time')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.show()
    
    return yearly_stats, rest_by_season

all_games = reshape_data()
regular_season_games = all_games[(all_games['GAME_TYPE'] == 'regular_season') & (all_games['SEASON'] != '2019-2020')]
regular_season_games = regular_season_games.drop(columns=['GAME_TYPE'])
regular_season_games.to_csv('data/game/box_raw/reshaped_regular_season_games.csv', index=False)