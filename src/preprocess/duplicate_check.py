import pandas as pd
import os

for file in os.listdir('data/player/box_weighted_average'):
    df = pd.read_csv(f'data/player/box_weighted_average/{file}')
    # Check for duplicate rows
    duplicate_rows = df[df.duplicated()]
    
    if not duplicate_rows.empty:
        print(f"Duplicate rows found in {file}:")
        print(duplicate_rows)