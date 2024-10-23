import os
import pandas as pd
import numpy as np

data_dir = 'data/player/box_weighted_average'

for player in os.listdir(data_dir):
    player_df = pd.read_csv(os.path.join(data_dir, player))
    # replace inf cells with 0
    player_df.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Save the updated DataFrame back to the CSV file
    player_df.to_csv(os.path.join(data_dir, player), index=False)
    print(f"processed {player}")