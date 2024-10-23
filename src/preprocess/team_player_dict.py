import os, pickle
import pandas as pd

team_player_dict = {}
data_dir = 'raw/player'

for file_name in os.listdir(data_dir):
    player_name = file_name[:-4]
    if file_name != '.DS_Store':
        player_df = pd.read_csv(os.path.join(data_dir, file_name))
    for _, row in player_df.iterrows():
        date = row['DATE']
        team = row['TEAM']
        if team not in team_player_dict:
            team_player_dict[team] = {}
        if date not in team_player_dict[team]:
            team_player_dict[team][date] = [player_name]
        else:
            team_player_dict[team][date].append(player_name)

with open('team_player_dict.pkl', 'wb') as file:
    pickle.dump(team_player_dict, file)