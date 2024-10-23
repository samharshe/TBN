import pandas as pd
import os

dir_paths = ['data/player/box_raw',
            'data/player/box_weighted_average',
            'data/team/box_raw',
            'data/team/box_weighted_average']
for dir_path in dir_paths:
    print(f'checking {dir_path} for duplicates')
    for f in os.listdir(dir_path):
        df = pd.read_csv(f'{dir_path}/{f}')
        duplicate_dates = df[df.duplicated(subset=['DATE'], keep=False)]
        
        if not duplicate_dates.empty:
            print(f"duplicates found: {f}")
            print(duplicate_dates)
    print(f'finished checking {dir_path} for duplicates')
