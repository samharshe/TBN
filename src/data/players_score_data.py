import os, torch, random
from typing import Tuple
from torch.utils import data as torch_data
import torch
from src.data import data as data_utils

def dataloaders(name: str, train_split: float=0.8,  val_split: float=0.1, test_split: float=0.1, batch_size: int=32) -> Tuple[torch_data.DataLoader, torch_data.DataLoader, torch_data.DataLoader]:
    """
    """
    dataset_spec = f'{name}_playerwisescore_{train_split}_{val_split}_{test_split}_{batch_size}'
    dataloaders_path = f'data/dataloader/{dataset_spec}.pt'
    if os.path.exists(dataloaders_path):
        return torch.load(dataloaders_path)
    else:
        try:
            dataset = data_utils.GameDataset(name=name)
        except:
            raise ValueError(f'dataset {name} not found. please build dataset before loading datalaoders on it.')
        
    dataset = [(data.x['players'].float(), data.y['players'][:,21].float()) for data in dataset]
    dataset = data_utils.shape_dataset(dataset)
    dataset = data_utils.normalize_dataset(dataset)
    dataset = data_utils.engineer_dataset(dataset)

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

    train_dataloader = torch_data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=data_utils.custom_collate)
    val_dataloader = torch_data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, collate_fn=data_utils.custom_collate)
    test_dataloader = torch_data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, collate_fn=data_utils.custom_collate)

    torch.save((train_dataloader, val_dataloader, test_dataloader), dataloaders_path)
    return train_dataloader, val_dataloader, test_dataloader