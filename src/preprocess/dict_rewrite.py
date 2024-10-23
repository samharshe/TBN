import os, pickle
import pandas as pd

team_player_dict = pickle.load(open('data/team_player_dict/team_player_dict.pkl', 'rb'))

buggy_players = [
    'Dorell Wright',
    'Earl Watson',
    'Jamaal Franklin',
    'Joel Freeland',
    'Jon Leuer',
    'Kosta Koufos',
    'Mike Miller',
    'Al Harrington',
    'Rodney White',
    'Vladimir Radmanović',
    'Nikoloz Tskitishvili',
    'Chris Wilcox',
    'Matt Barnes',
    'Šarūnas Jasikevičius',
    'Tim Thomas',
    'Ruben Patterson',
    'Charles Smith',
    'Malik Allen',
    'Shawn Marion',
    'Antoine Wright',
    'Kirk Snyder',
    'Josh Powell',
    'Bobby Jones',
    'Stephen Jackson',
    'Mike James',
    'Jason Kidd',
    'Marcus Williams 2',
    'Marcus Williams 1',
    'Brandon Williams 1',
    'Brandon Williams 2',
    'Chris Johnson 2',
    'Patrick Ewing 1',
    'Tony Mitchell 2',
    'Patrick Ewing 2',
    'Chris Johnson 1',
    'Tony Mitchell 1',
]

for player in buggy_players:
    print(f'rewriting dict for {player}')
    if any(char.isdigit() for char in player):
        name_to_remove = player[:-2]
    else:
        name_to_remove = player
    player_df = pd.read_csv(f'data/player/box_raw/{player}.csv')
    for _, row in player_df.iterrows():
        date = row['DATE']
        team = row['TEAM']
        original_list = team_player_dict[team][date]
        modified_list = [player for player in original_list if player != name_to_remove]
        modified_list.append(player)
        while modified_list.count(player) > 1:
            modified_list.remove(player)
        team_player_dict[team][date] = modified_list
pickle.dump(team_player_dict, open('data/team_player_dict/team_player_dict.pkl', 'wb'))