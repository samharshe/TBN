import os, pickle, torch, random
from typing import Tuple, Dict, Any, List
from torch.utils import data as torch_data
import pandas as pd
import torch
from src.ml import model
from IPython.display import clear_output

class Game():
    """
    class to store single game's data. really just a wrapper for dictionaries.

    parameters
    ----------
    x: Dict[str, torch.Tensor]
        input data for given game. for example, player's weighted statistics up to game.
    y: Dict[str, torch.Tensor]
        output data for given game. for example, player's statistics in game.
    metadata: Dict[str, Any]
        metadata for given game. for example, season, date, home_players, away_players.
    """
    def __init__(self, x: Dict[str, torch.Tensor], y: Dict[str, torch.Tensor], metadata: Dict[str, Any]):
        self.x = x
        self.y = y
        self.metadata = metadata
        
    def __str__(self):
        return str(self.metadata)

class GameDataset(torch_data.Dataset):
    """
    class to store a dataset of games. really just a wrapper for a list of Game objects with nice preprocessing functionality.

    parameters
    ----------
    name: str
        name of dataset. if name already exists, will load from file, so no need to specify data versions.
    player_x_version: str = None
        version of player statistics to use for inputs. 
    player_y_version: str = None
        version of player statistics to use for targets.
    team_x_version: str = None
        version of team statistics to use for inputs.
    team_y_version: str = None
        version of team statistics to use for targets.
    """
    def __init__(self, name: str, player_x_version: str = None, player_y_version: str = None, team_x_version: str = None, team_y_version: str = None):
        self.data = self._init_data(name=name, player_x_version=player_x_version, player_y_version=player_y_version, team_x_version=team_x_version, team_y_version=team_y_version)
        
    def _init_data(self, name: str, player_x_version: str, player_y_version: str, team_x_version: str, team_y_version: str) -> List[Game]:
        processed_dataset_path = f'data/dataset/{name}.pt'
        if os.path.exists(processed_dataset_path):
            return torch.load(processed_dataset_path)
        else:
            if None in [player_x_version, player_y_version, team_x_version, team_y_version]:
                raise TypeError(f'no dataset named {name} found and at least 1 data version not specified.')
            data = self._create_dataset(player_x_version=player_x_version, player_y_version=player_y_version, team_x_version=team_x_version, team_y_version=team_y_version, name=name)
            torch.save(data, processed_dataset_path)
            return data
    
    def _create_dataset(self, player_x_version: str, player_y_version: str, team_x_version: str, team_y_version: str, name: str) -> List[Game]:
        dataset_versions = {
            'player_x_version': player_x_version,
            'player_y_version': player_y_version,
            'team_x_version': team_x_version,
            'team_y_version': team_y_version,
        }
        data = []
        # customize df here to load games you want
        game_df = pd.read_csv('data/game/bbref_game.csv')
        team_player_dict = pickle.load(open('data/team_player_dict/team_player_dict 2.pkl', 'rb'))
        for season in game_df['SEASON'].unique():
            season_df = game_df[game_df['SEASON'] == season]
            season_data = season_df.apply(self._row_to_game, axis=1, args=(dataset_versions,team_player_dict,)).to_list()
            season_file = f'data/dataset/{name}_{season}.pt'
            torch.save(season_data, season_file)
            print(f'saved {season} season data to {season_file}')
            data.extend(season_data)
        
        return data
        
    def _row_to_game(self, row: pd.DataFrame, dataset_versions: Dict, team_player_dict: Dict) -> Game:
            x, y, metadata = {}, {}, {}
            home = row['HOME_TEAM']
            away = row['AWAY_TEAM']
            date = row['DATE']
            season = row['SEASON']
            
            teams = [home, away]
            
            # clear previous print output
            clear_output()
            print(f'beginning processing of {home} vs. {away} on {date}')
            
            home_players = team_player_dict[home][date]
            away_players = team_player_dict[away][date]
            players = home_players + away_players
            
            tensor_lists = {
                'player_x_tensor_list': [], 
                'player_y_tensor_list': [], 
                'team_x_tensor_list': [], 
                'team_y_tensor_list': []
            }
            home_score, away_score = None, None

            for z in ['x', 'y']:
                player_version = dataset_versions[f'player_{z}_version']
                for player in players:
                    if player == '.DS_S':
                        with open('flag.txt', 'a') as flag_file:
                            flag_file.write(f'bug found for {home} v. {away} on {date}\n')
                    else:
                        df = pd.read_csv(f'data/player/{player_version}/{player}.csv')
                        df = df[df['DATE'] == date]
                        if z == 'x':
                            # distinguish home and away players with last-column flags
                            if player in home_players:
                                df['HOME'] = 1
                            else:
                                df['HOME'] = -1
                        df = df.drop(columns=['TEAM', 'SEASON', 'DATE', 'OPPONENT', 'HOME_TEAM'])
                        tensor = torch.tensor(df.values)
                        tensor_lists[f'player_{z}_tensor_list'].append(tensor)

                team_version = dataset_versions[f'team_{z}_version']
                for team in teams:
                    df = pd.read_csv(f'data/team/{team_version}/{team}.csv')
                    df = df[df['DATE'] == date]
                    if z == 'x':
                        if team == home:
                            df['HOME'] = 1
                        else:
                            df['HOME'] = -1
                    else:
                        if team == home:
                            home_score = df.iloc[0]['PTS'].item()
                        else:
                            away_score = df.iloc[0]['PTS'].item()
                    df = df.drop(columns=['SEASON', 'DATE', 'OPPONENT', 'HOME_TEAM'])
                    tensor = torch.tensor(df.values)
                    tensor_lists[f'team_{z}_tensor_list'].append(tensor)
                    
            player_x_tensor = torch.stack(tensor_lists['player_x_tensor_list']).squeeze(dim=1)
            player_y_tensor = torch.stack(tensor_lists['player_y_tensor_list']).squeeze(dim=1)
            team_x_tensor = torch.stack(tensor_lists['team_x_tensor_list']).squeeze(dim=1)
            team_y_tensor = torch.stack(tensor_lists['team_y_tensor_list']).squeeze(dim=1)
                
            x = {
                'players': player_x_tensor,
                'teams': team_x_tensor
            }
            
            y = {
                'players': player_y_tensor,
                'teams': team_y_tensor,
                'score': torch.tensor((home_score, away_score)),
                'home_win': torch.tensor(int(home_score > away_score))
            }
            
            metadata = {
                'season': season,
                'home': home,
                'away': away,
                'date': date,
                'home_players': home_players,
                'away_players': away_players
            }
            
            game = Game(x=x, y=y, metadata=metadata)
            
            return game
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Game:
        return self.data[index]
    
def custom_collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    custom collate function to handle padding of inputs and targets given that number of players in a game may vary.

    parameters
    ----------
    batch: List[Tuple[torch.Tensor, torch.Tensor]]
        batch of games to be collated.

    returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        inputs and targets, properly padded with zeros so that each game in a given batch has the same player dimension (if applicable).
    """
    # separate inputs and targets
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    # find the maximum number of players
    max_inputs = max(inp.shape[0] for inp in inputs)
    max_target = max(target.shape[0] for target in targets)

    # padding along player dimension
    # if inputs or outputs are not player-wise, max_ == n_ for all items in batch, so this does nothing but does not break anything
    padded_targets, padded_inputs = [], []
    for inp, target in zip(inputs, targets):
        n_targets = target.shape[0]
        n_inputs = inp.shape[0]
        target_padding = torch.zeros(max_target - n_targets, target.shape[1])
        input_padding = torch.zeros(max_inputs - n_inputs, inp.shape[1])
        padded_target = torch.cat([target, target_padding], dim=0)
        padded_input = torch.cat([inp, input_padding], dim=0)
        padded_targets.append(padded_target)
        padded_inputs.append(padded_input)

    # turn batch list into tensor
    padded_inputs = torch.stack(padded_inputs)
    padded_targets = torch.stack(padded_targets)
    
    return padded_inputs, padded_targets

def shape_dataset(dataset: List[Tuple[torch.Tensor, torch.Tensor]]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    if len(dataset[0][1].shape) == 1:
        dataset = [(x, y.unsqueeze(dim=1)) for x, y in dataset]
    return dataset

def normalize_dataset(dataset: List[Tuple[torch.Tensor, torch.Tensor]]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    # calculate max values across the entire dataset
    all_inputs = torch.cat([item[0] for item in dataset], dim=0)
    all_targets = torch.cat([item[1] for item in dataset], dim=0)

    max_input, _ = torch.max(torch.abs(all_inputs), dim=0)
    max_target, _ = torch.max(torch.abs(all_targets), dim=0)
    
    # normalize all first items by dividing by the max
    normalized_inputs = all_inputs / (max_input + 1e-8)  # add small epsilon to avoid division by zero
    normalized_targets = all_targets / (max_target + 1e-8)  # add small epsilon to avoid division by zero
    
    # now normalized_inputs and normalized_targets are concatenated along the 0th dimension
    # this split them back up into their original sizes
    normalized_dataset = []
    input_index, target_index = 0, 0
    for i in range(len(dataset)):
        n_inputs = dataset[i][0].shape[0]
        n_targets = dataset[i][1].shape[0]
        end_input_index = input_index + n_inputs
        end_target_index = target_index + n_targets
        normalized_input = normalized_inputs[input_index:end_input_index]
        normalized_target = normalized_targets[target_index:end_target_index]
        normalized_dataset.append((normalized_input, normalized_target))
        input_index = end_input_index
        target_index = end_target_index

    return normalized_dataset

def engineer_dataset(dataset):
    index_to_feature_dim_map = {0: 6, 1: 4, 2: 4, 3: 1, 4: 2, 5: 2, 7: 2, 8: 2, 10: 2, 11: 2, 13: 2, 14: 1, 15: 3, 16: 4, 17: 2, 18: 2, 19: 2, 21: 6, 22: 4, 23: 2, 24: 1, 25: 1, 36: 2, 37: 3, 38: 2, 39: 1}
    engineered_list = []
    for x, y in dataset:
        features_list = []
        for idx, feature_dim in index_to_feature_dim_map.items():
            expanded_feature = model.gaussian_expansion(x[:,idx].unsqueeze(dim=1), min=0, max=1, out_features=feature_dim)
            features_list.append(expanded_feature)
        new_features = torch.cat(features_list, dim=1)
        engineered_list.append((new_features, y))
    
    return engineered_list

def simplify_dataset(dataset: List[Tuple[torch.Tensor, torch.Tensor]]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    # 0: MP, 1: FGM, 2: FGA, 3: FG%, 4: 2PT_FGM, 
    # 5: 2PT_FGA, 6: 2PT_FG%, 7: 3PT_FGM, 8: 3PT_FGA, 
    # 9: 3PT_FG%, 10: FTM, 11: FTA, 12: FT%, 
    # 13: ORB, 14: DRB, 15: TRB, 16: AST, 
    # 17: STL, 18: BLK, 19: TOV, 20: PF, 
    # 21: PTS, 22: GMSC, 23: +/-, 24: TS%, 
    # 25: EFG%, 26: 3PAR, 27: FTR, 28: ORB%, 
    # 29: DRB%, 30: TRB%, 31: AST%, 32: STL%, 
    # 33: BLK%, 34: TOV%, 35: USG%, 36: ORTG, 
    # 37: DRTG, 38: BPM
    # index_list = [0, 13, 14, 16, 17, 18, 19, 21, 37, 38]
    index_list = [0,1,2,3,4,5,7,8,10,11,13,14,15,16,17,18,19,21,22,23,24,25,36,37,38,39]
    simplified_list = []
    for x, y in dataset:
        features_list = []
        for index in index_list:
            selected_feature = x[:,index].unsqueeze(dim=1)
            features_list.append(selected_feature)
        new_features = torch.cat(features_list, dim=1)
        simplified_list.append((new_features, y))
    
    return simplified_list

def dataloaders(name: str, x_version: str, y_version: str, train_split: float=0.8,  val_split: float=0.1, test_split: float=0.1, batch_size: int=32, simple: bool=False, engineered: bool=False) -> Tuple[torch_data.DataLoader, torch_data.DataLoader, torch_data.DataLoader]:
    """

    TARGETS FOR TEAM: batch x 2 x n_targets
    TARGETS FOR PLAYER: batch x n_players x n_targets
    TARGETS FOR SCORE: batch x 1 x 2

    """
    if simple and engineered:
        raise ValueError('cannot specify both simple and engineered.')
    dataset_spec = f'{name}_{x_version}_{y_version}_{train_split}_{val_split}_{test_split}_{batch_size}'
    if simple:
        dataset_spec += '_simple'
    elif engineered:
        dataset_spec += '_engineered'
    dataloaders_path = f'data/dataloader/{dataset_spec}.pt'
    if os.path.exists(dataloaders_path):
        return torch.load(dataloaders_path)
    else:
        try:
            dataset = GameDataset(name=name)
        except:
            raise ValueError(f'dataset {name} not found. please build dataset before loading datalaoders on it.')
        
    dataset = [(data.x[x_version].float(), data.y[y_version].float()) for data in dataset]
    dataset = shape_dataset(dataset)
    dataset = normalize_dataset(dataset)
    if simple:
        dataset = simplify_dataset(dataset)
    elif engineered:
            dataset = engineer_dataset(dataset)

    total_size = len(dataset)
    indices = list(range(total_size))
    random.shuffle(indices)

    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = int(total_size * test_split)

    train_indices, val_indices, test_indices = indices[:train_size], indices[train_size:train_size+val_size], indices[-test_size:]

    train_sampler = torch_data.SubsetRandomSampler(train_indices)
    valid_sampler = torch_data.SubsetRandomSampler(val_indices)
    test_sampler = torch_data.SubsetRandomSampler(test_indices)

    train_dataloader = torch_data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=custom_collate)
    val_dataloader = torch_data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, collate_fn=custom_collate)
    test_dataloader = torch_data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, collate_fn=custom_collate)

    torch.save((train_dataloader, val_dataloader, test_dataloader), dataloaders_path)
    return train_dataloader, val_dataloader, test_dataloader