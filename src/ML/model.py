import torch
from torch import nn
from typing import Tuple

def home_away_tensors(in_tensor: torch.Tensor, original_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    home_zero_tensor, away_zero_tensor = torch.zeros_like(in_tensor), torch.zeros_like(in_tensor)
    home_mask, away_mask = (original_tensor[:, :, -1] == 1), (original_tensor[:, :, -1] == -1)
    home_zero_tensor[home_mask], away_zero_tensor[away_mask] = in_tensor[home_mask], in_tensor[away_mask]
    return home_zero_tensor, away_zero_tensor

def gaussian_expansion(x: torch.Tensor, min: float, max: float, out_features: int) -> torch.Tensor:
    # for simpler syntax elsewhere
    if out_features == 1:
        return x
    
    # keep from weird errors with improper broadcasting
    if x.shape[-1] != 1:
        raise ValueError('x must be 1-dimensional')
    
    # tensors for simple broadcasting
    centers = torch.linspace(min, max, out_features)
    sigma = (max - min) / out_features
    
    # all in one go
    gaussian_expansion_tensor = (torch.exp((-0.5)*((x - centers) / sigma)**2)) / (sigma * torch.sqrt(torch.tensor(2*torch.pi))) 
    
    # return tensor
    return gaussian_expansion_tensor

class InitializedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)
    
    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        out_tensor = self.linear(in_tensor)
        return out_tensor
    
class MultiHeadWrapper(nn.Module):
    def __init__(self, d_mod, n_heads):
        super().__init__()
        self.q = InitializedLinear(in_features=d_mod, out_features=d_mod)
        self.k = InitializedLinear(in_features=d_mod, out_features=d_mod)
        self.v = InitializedLinear(in_features=d_mod, out_features=d_mod)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=d_mod, num_heads=n_heads)
    
    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        q, k, v = self.q(in_tensor), self.k(in_tensor), self.v(in_tensor)
        out_tensor = self.multihead_attention(q, k, v)[0]
        return out_tensor
        
class MeanSecondDim(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        out_tensor = torch.mean(in_tensor, dim=2, keepdim=True)
        return out_tensor

class MeanFirstDim(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, in_tensor):
        out_tensor = torch.mean(in_tensor, dim=1, keepdim=True)
        return out_tensor

class SumFirstDim(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        out_tensor = torch.sum(in_tensor, dim=1, keepdim=False)
        return out_tensor
    
class MckinneyPlayerModel(nn.Module):
    def __init__(self, d_in, d_mod, d_out):
        super().__init__()
        self.d_in, self.d_mod, self.d_out = d_in, d_mod, d_out
        self.act = nn.ReLU()
        self.dropout_p = 0.0

        self.embedding = nn.Sequential(
            MultiHeadWrapper(d_mod=self.d_in, n_heads=5),
            self.act,
            nn.LayerNorm(normalized_shape=self.d_in),
            InitializedLinear(in_features=self.d_in, out_features=self.d_mod),
            nn.Dropout(p=self.dropout_p),
            self.act,
            nn.LayerNorm(normalized_shape=self.d_mod)
        )

        self.body = nn.Sequential(
            MultiHeadWrapper(d_mod=self.d_mod, n_heads=4),
            self.act,
            nn.LayerNorm(normalized_shape=self.d_mod),
            InitializedLinear(in_features=self.d_mod, out_features=self.d_mod),
            nn.Dropout(p=self.dropout_p),
            self.act,
            nn.LayerNorm(normalized_shape=self.d_mod)
        )

        self.pred = nn.Sequential(
            MeanFirstDim(),
            InitializedLinear(in_features=self.d_mod, out_features=self.d_out),
            nn.Dropout(p=self.dropout_p)
        )
    
    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        out_tensor = self.embedding(in_tensor)
        out_tensor = self.body(out_tensor)
        out_tensor = self.pred(out_tensor)
        return out_tensor

class HomeAwayModel(nn.Module):
    def __init__(self, d_in, d_mod, d_out):
        super().__init__()
        self.d_in, self.d_mod, self.d_out = d_in, d_mod, d_out
        self.act = nn.ReLU()
        self.dropout_p = 0.0

        self.embedding = nn.Sequential(
            MultiHeadWrapper(d_mod=self.d_in, n_heads=5),
            self.act,
            nn.LayerNorm(normalized_shape=self.d_in),
            InitializedLinear(in_features=self.d_in, out_features=self.d_mod),
            nn.Dropout(p=self.dropout_p),
            self.act,
            nn.LayerNorm(normalized_shape=self.d_mod),
            InitializedLinear(in_features=self.d_mod, out_features=self.d_mod),
            nn.Dropout(p=self.dropout_p),
            self.act,
            nn.LayerNorm(normalized_shape=self.d_mod)
        )

        self.home_pred = nn.Sequential(
            MeanFirstDim(),
            InitializedLinear(in_features=self.d_mod, out_features=1),
            nn.Dropout(p=self.dropout_p)
        )

        self.away_pred = nn.Sequential(
            MeanFirstDim(),
            InitializedLinear(in_features=self.d_mod, out_features=1),
            nn.Dropout(p=self.dropout_p)
        )
        
    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        out_tensor = self.embedding(in_tensor)
        home_tensor, away_tensor = home_away_tensors(in_tensor=out_tensor, original_tensor=in_tensor)
        home_pred = self.home_pred(home_tensor)
        away_pred = self.away_pred(away_tensor)
        out_tensor = torch.cat((home_pred, away_pred), dim=1).unsqueeze(dim=2)
        return out_tensor
    
class PlayerScoreModel(nn.Module):
    def __init__(self, d_in, d_mod):
        super().__init__()
        self.d_in, self.d_mod = d_in, d_mod
        self.act = nn.ReLU()
        self.dropout_p = 0.0

        self.embedding = nn.Sequential(
            InitializedLinear(in_features=self.d_in, out_features=self.d_mod),
            nn.Dropout(p=self.dropout_p),
            self.act,
            nn.LayerNorm(normalized_shape=self.d_mod),
            InitializedLinear(in_features=self.d_mod, out_features=self.d_mod),
            nn.Dropout(p=self.dropout_p),
            self.act,
            nn.LayerNorm(normalized_shape=self.d_mod),
        )

        self.body = nn.Sequential(
            MultiHeadWrapper(d_mod=self.d_mod, n_heads=4),
            self.act,
            nn.LayerNorm(normalized_shape=self.d_mod),
        )

        self.pred = nn.Sequential(
            InitializedLinear(in_features=self.d_mod, out_features=1),
            nn.Dropout(p=self.dropout_p),
        )
     
    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        out_tensor = self.embedding(in_tensor)
        out_tensor = self.body(out_tensor) + out_tensor
        out_tensor = self.pred(out_tensor)
        return out_tensor

class TeamScoreModel(nn.Module):
    def __init__(self, d_in, d_mod):
        super().__init__()
        self.d_in, self.d_mod = d_in, d_mod
        self.act = nn.ReLU()
        self.dropout_p = 0.1

        self.embedding = nn.Sequential(
            MultiHeadWrapper(d_mod=self.d_in, n_heads=7),
            self.act,
            nn.LayerNorm(normalized_shape=self.d_in),
            InitializedLinear(in_features=self.d_in, out_features=self.d_mod),
            nn.Dropout(p=self.dropout_p),
            self.act,
            nn.LayerNorm(normalized_shape=self.d_mod)
        )

        self.pred = nn.Sequential(
            InitializedLinear(in_features=self.d_mod, out_features=1),
        )
     
    def forward(self, in_list: list[torch.Tensor]) -> torch.Tensor:
        in_tensor = in_list[0]
        out_tensor = self.embedding(in_tensor)
        out_tensor = self.pred(out_tensor)
        return out_tensor

class HybridModel(nn.Module):
    def __init__(self, d_in, d_mod):
        super().__init__()
        self.d_mod = d_mod
        self.player_d_in, self.team_d_in = d_in[0], d_in[1]
        self.act = nn.ReLU()
        self.dropout_p = 0.0

        self.team_embedding = nn.Sequential(
            MultiHeadWrapper(d_mod=self.team_d_in, n_heads=7),
            self.act,
            nn.LayerNorm(normalized_shape=self.team_d_in),
            InitializedLinear(in_features=self.team_d_in, out_features=self.d_mod),
            nn.Dropout(p=self.dropout_p),
            self.act,
            nn.LayerNorm(normalized_shape=self.d_mod),
        )

        self.player_embedding = nn.Sequential(
            MultiHeadWrapper(d_mod=self.player_d_in, n_heads=5),
            self.act,
            nn.LayerNorm(normalized_shape=self.player_d_in),
            InitializedLinear(in_features=self.player_d_in, out_features=self.d_mod),
            nn.Dropout(p=self.dropout_p),
            self.act,
            nn.LayerNorm(normalized_shape=self.d_mod),
        )

        self.player_home = nn.Sequential(
            MeanFirstDim(),
            InitializedLinear(in_features=self.d_mod, out_features=self.d_mod),
            nn.Dropout(p=self.dropout_p),
            self.act,
            nn.LayerNorm(normalized_shape=self.d_mod)
        )

        self.player_away = nn.Sequential(
            MeanFirstDim(),
            InitializedLinear(in_features=self.d_mod, out_features=self.d_mod),
            nn.Dropout(p=self.dropout_p),
            self.act,
            nn.LayerNorm(normalized_shape=self.d_mod)
        )

        self.hybrid_body = nn.Sequential(
            InitializedLinear(in_features=self.d_mod*2, out_features=self.d_mod),
            nn.Dropout(p=self.dropout_p),
            self.act,
            nn.LayerNorm(normalized_shape=self.d_mod)
        )

        self.pred = nn.Sequential(
            InitializedLinear(in_features=self.d_mod, out_features=1),
        )
     
    def forward(self, in_list: list[torch.Tensor]) -> torch.Tensor:
        player_tensor, team_tensor = in_list[0], in_list[1]
        team_tensor = self.team_embedding(team_tensor)
        player_tensor = self.player_embedding(player_tensor)
        home_players_tensor, away_players_tensor = home_away_tensors(in_tensor=player_tensor, original_tensor=player_tensor)
        home_players_tensor, away_players_tensor = self.player_home(home_players_tensor), self.player_away(away_players_tensor)
        players_tensor = torch.cat((home_players_tensor, away_players_tensor), dim=1)
        out_tensor = torch.cat((team_tensor, players_tensor), dim=2)
        out_tensor = self.hybrid_body(out_tensor)
        out_tensor = self.pred(out_tensor)
        return out_tensor