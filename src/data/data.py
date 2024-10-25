import os, pickle, torch, random
from typing import Tuple, Dict, Any, List
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import pandas as pd
from torch import Tensor
from src.ml.model import gaussian_expansion
from IPython.display import clear_output

class Game():
    def __init__(self, x: Dict[str, Tensor], y: Dict[str, Tensor], metadata: Dict[str, Any]):
        self.x = x
        self.y = y
        self.metadata = metadata
        
    def __str__(self):
        return str(self.metadata)

class GameDataset(Dataset):
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
        year_strings = [f'20{year-1:02d}-{year:02d}' for year in range(2,22)]
        print(year_strings)
        for year_string in year_strings:
            path_to_load = f'data/dataset/elephant_{year_string}.pt'
            data += torch.load(path_to_load)
        game_df = pd.read_csv('data/game/bbref_game.csv')
        game_df = game_df[~game_df['SEASON'].isin(['2000-01'] + year_strings)]
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
            
            # clear any previous print output
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
    
def custom_collate(batch):
    # Separate inputs and targets
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    # Find the maximum number of players
    max_players = max(inp.shape[0] for inp in inputs)
    
    # Pad inputs
    padded_inputs = []
    for inp in inputs:
        n_players, emb_dim = inp.shape
        padding = torch.zeros(max_players - n_players, emb_dim)
        padded_inp = torch.cat([inp, padding], dim=0)
        padded_inputs.append(padded_inp)
    
    # Stack padded inputs and targets
    padded_inputs = torch.stack(padded_inputs)
    targets = torch.stack(targets)
    
    return padded_inputs, targets

def custom_collate(batch):
    # Separate inputs and targets
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    # Find the maximum number of players
    max_players = max(inp.shape[0] for inp in inputs)
    
    # Pad inputs
    padded_inputs = []
    padded_targets = []
    for inp, target in zip(inputs, targets):
        n_players, emb_dim = inp.shape
        inp_padding = torch.zeros(max_players - n_players, emb_dim)
        target_padding = torch.zeros(max_players - n_players)
        padded_inp = torch.cat([inp, inp_padding], dim=0)
        print(target.shape)
        print(target_padding.shape)
        padded_target = torch.cat([target, target_padding], dim=0)
        padded_inputs.append(padded_inp)
        padded_targets.append(padded_target)
    
    # Stack padded inputs and targets
    padded_inputs = torch.stack(padded_inputs)
    padded_targets = torch.stack(padded_targets)
    
    return padded_inputs, padded_targets

def reshape_x(tensor):
    return tensor.view(tensor.shape[0], -1)

def normalize_dataset(dataset):
    all_first_items = torch.cat([item[0] for item in dataset], dim=0)
    all_second_items = torch.stack([item[1] for item in dataset])
    
    # calculate max values across the entire dataset
    max_first_values, _ = torch.max(torch.abs(all_first_items), dim=0)
    max_second_values, _ = torch.max(torch.abs(all_second_items), dim=0)
    
    # normalize all first items by dividing by the max
    normalized_first_items = all_first_items / (max_first_values + 1e-8)  # add small epsilon to avoid division by zero
    normalized_second_items = all_second_items / (max_second_values + 1e-8)  # add small epsilon to avoid division by zero
    
    normalized_list = []
    start_idx = 0
    for idx, item in enumerate(dataset):
        n_players = item[0].shape[0]
        end_idx = start_idx + n_players
        normalized_game = normalized_first_items[start_idx:end_idx]
        normalized_result = normalized_second_items[idx].view(-1)
        normalized_list.append((normalized_game, normalized_result))
        start_idx = end_idx
    
    return normalized_list

def festively_normalize_dataset(dataset):
    all_first_items = torch.cat([item[0] for item in dataset], dim=0)
    all_second_items = torch.cat([item[1] for item in dataset], dim=0)
    
    # calculate max values across the entire dataset
    max_first_values, _ = torch.max(torch.abs(all_first_items), dim=0)
    max_second_values, _ = torch.max(torch.abs(all_second_items), dim=0)
    
    # normalize all first items by dividing by the max
    normalized_first_items = all_first_items / (max_first_values + 1e-8)  # add small epsilon to avoid division by zero
    normalized_second_items = all_second_items / (max_second_values + 1e-8)  # add small epsilon to avoid division by zero
    
    normalized_list = []
    start_idx = 0
    for idx, item in enumerate(dataset):
        n_players = item[0].shape[0]
        end_idx = start_idx + n_players
        normalized_game = normalized_first_items[start_idx:end_idx]
        normalized_result = normalized_second_items[start_idx:end_idx]
        normalized_list.append((normalized_game, normalized_result))
        start_idx = end_idx
    
    return normalized_list

def engineer_dataset(dataset):
    index_to_feature_dim_map = {0: 6, 1: 4, 2: 4, 3: 1, 4: 2, 5: 2, 7: 2, 8: 2, 10: 2, 11: 2, 13: 2, 14: 1, 15: 3, 16: 4, 17: 2, 18: 2, 19: 2, 21: 6, 22: 4, 23: 2, 24: 1, 25: 1, 36: 2, 37: 3, 38: 2, 39: 1}
    engineered_list = []
    for x, y in dataset:
        features_list = []
        for idx, feature_dim in index_to_feature_dim_map.items():
            expanded_feature = gaussian_expansion(x[:,idx].unsqueeze(dim=1), min=0, max=1, out_features=feature_dim)
            features_list.append(expanded_feature)
        new_features = torch.cat(features_list, dim=1)
        engineered_list.append((new_features, y))
    
    return engineered_list

def get_dataloaders(name: str,
                    x_version: str,
                    y_version: str,
                    train_split: float=0.8, 
                    val_split: float=0.1, 
                    test_split: float=0.1, 
                    batch_size: int=32) -> Tuple[DataLoader, DataLoader, DataLoader]:
    try:
        dataset = GameDataset(name=name)
        dataset = [(data.x[x_version].float(), data.y[y_version].float()) for data in dataset]
        dataset = normalize_dataset(dataset)
    except:
        raise ValueError(f'dataset {name} not found. please build dataset before loading datalaoders on it.')
    
    total_size = len(dataset)
    indices = list(range(total_size))
    random.shuffle(indices)

    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size

    train_indices, val_indices, test_indices = indices[:train_size], indices[train_size:train_size+val_size], indices[train_size+val_size:]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=custom_collate)
    val_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, collate_fn=custom_collate)
    test_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, collate_fn=custom_collate)

    return train_dataloader, val_dataloader, test_dataloader

def get_engineered_dataloaders(name: str,
                    x_version: str,
                    y_version: str,
                    train_split: float=0.8, 
                    val_split: float=0.1, 
                    test_split: float=0.1, 
                    batch_size: int=32) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataloaders_path = f'data/dataloader/{name}_{x_version}_{y_version}_{train_split}_{val_split}_{test_split}_{batch_size}.pt'
    if os.path.exists(dataloaders_path):
        return torch.load(dataloaders_path)
    else:
        try:
            dataset = GameDataset(name=name)
            dataset = [(data.x[x_version].float(), data.y[y_version].float()) for data in dataset]
            dataset = normalize_dataset(dataset)
            dataset = engineer_dataset(dataset)
        except:
            raise ValueError(f'dataset {name} not found. please build dataset before loading datalaoders on it.')
        
        total_size = len(dataset)
        indices = list(range(total_size))
        random.shuffle(indices)

        train_size = int(total_size * train_split)
        val_size = int(total_size * val_split)
        test_size = total_size - train_size - val_size

        train_indices, val_indices, test_indices = indices[:train_size], indices[train_size:train_size+val_size], indices[train_size+val_size:]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=custom_collate)
        val_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, collate_fn=custom_collate)
        test_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, collate_fn=custom_collate)

        torch.save((train_dataloader, val_dataloader, test_dataloader), dataloaders_path)
        return train_dataloader, val_dataloader, test_dataloader
    

def get_simplified_dataloaders(name: str,
                    x_version: str,
                    y_version: str,
                    train_split: float=0.8, 
                    val_split: float=0.1, 
                    test_split: float=0.1, 
                    batch_size: int=32) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataloaders_path = f'data/dataloader/simplified_{name}_{x_version}_{y_version}_{train_split}_{val_split}_{test_split}_{batch_size}.pt'
    if os.path.exists(dataloaders_path):
        return torch.load(dataloaders_path)
    else:
        try:
            dataset = GameDataset(name=name)
            dataset = [(data.x[x_version].float(), data.y[y_version].float()) for data in dataset]
            dataset = normalize_dataset(dataset)
            dataset = simplify_dataset(dataset)
        except:
            raise ValueError(f'dataset {name} not found. please build dataset before loading datalaoders on it.')
        
        total_size = len(dataset)
        indices = list(range(total_size))
        random.shuffle(indices)

        train_size = int(total_size * train_split)
        val_size = int(total_size * val_split)
        test_size = total_size - train_size - val_size

        train_indices, val_indices, test_indices = indices[:train_size], indices[train_size:train_size+val_size], indices[train_size+val_size:]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=custom_collate)
        val_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, collate_fn=custom_collate)
        test_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, collate_fn=custom_collate)

        torch.save((train_dataloader, val_dataloader, test_dataloader), dataloaders_path)
        return train_dataloader, val_dataloader, test_dataloader

def simplify_dataset(dataset):
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
    engineered_list = []
    for x, y in dataset:
        features_list = []
        for index in index_list:
            selected_feature = x[:,index].unsqueeze(dim=1)
            features_list.append(selected_feature)
        new_features = torch.cat(features_list, dim=1)
        engineered_list.append((new_features, y))
    
    return engineered_list

def get_festive_dataloaders(name: str,
                    train_split: float=0.8, 
                    val_split: float=0.1, 
                    test_split: float=0.1, 
                    batch_size: int=32) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataloaders_path = f'data/dataloader/simplified_{name}_festive_{train_split}_{val_split}_{test_split}_{batch_size}.pt'
    if os.path.exists(dataloaders_path):
        return torch.load(dataloaders_path)
    else:
        try:
            dataset = GameDataset(name=name)
            dataset = [(data.x['players'].float(), data.y['players'][:,21].float()) for data in dataset]
            dataset = festively_normalize_dataset(dataset)
            dataset = engineer_dataset(dataset)
        except:
            raise ValueError(f'dataset {name} not found. please build dataset before loading datalaoders on it.')
        
        total_size = len(dataset)
        indices = list(range(total_size))
        random.shuffle(indices)

        train_size = int(total_size * train_split)
        val_size = int(total_size * val_split)
        test_size = total_size - train_size - val_size

        train_indices, val_indices, test_indices = indices[:train_size], indices[train_size:train_size+val_size], indices[train_size+val_size:]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=festive_collate)
        val_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, collate_fn=festive_collate)
        test_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, collate_fn=festive_collate)

        torch.save((train_dataloader, val_dataloader, test_dataloader), dataloaders_path)
        return train_dataloader, val_dataloader, test_dataloader