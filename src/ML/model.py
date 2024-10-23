import torch
from torch import nn

def get_home_away_tensors(in_tensor, original_tensor):
    home_mask = original_tensor[:,:,-1]==1
    batch_num_home_players = home_mask.sum(dim=1)
    batch_max_home_players = torch.max(batch_num_home_players)
    home_tensor = torch.zeros(in_tensor.size(0), batch_max_home_players, in_tensor.size(2))
    flattened_in_tensor = in_tensor.view(-1, in_tensor.size(2))
    flattened_home_mask = home_mask.view(-1)
    filtered_rows = flattened_in_tensor[flattened_home_mask]
    batch_offsets = torch.arange(in_tensor.size(0)).repeat_interleave(batch_num_home_players)
    row_offsets = torch.cat([torch.arange(nv) for nv in batch_num_home_players])
    home_tensor[batch_offsets, row_offsets] = filtered_rows
    
    away_mask = original_tensor[:,:,-1]==1
    batch_num_away_players = away_mask.sum(dim=1)
    batch_max_away_players = torch.max(batch_num_away_players)
    away_tensor = torch.zeros(in_tensor.size(0), batch_max_away_players, in_tensor.size(2))
    flattened_in_tensor = in_tensor.view(-1, in_tensor.size(2))
    flattened_away_mask = away_mask.view(-1)
    filtered_rows = flattened_in_tensor[flattened_away_mask]
    batch_offsets = torch.arange(in_tensor.size(0)).repeat_interleave(batch_num_away_players)
    row_offsets = torch.cat([torch.arange(nv) for nv in batch_num_away_players])
    away_tensor[batch_offsets, row_offsets] = filtered_rows
    
    return home_tensor, away_tensor

def gaussian_expansion(x, min, max, out_features):
    # for simpler syntax elsewhere
    if out_features == 1:
        return x
    
    centers = torch.linspace(min, max, out_features)
    sigma = (max - min) / out_features
    
    # keep from weird errors with improper broadcasting
    assert x.shape[-1] == 1, 'x must be 1-dimensional'
    
    # all in one go...no reason to break up steps
    gaussian_expansion_tensor = (torch.exp((-0.5)*((x - centers) / sigma)**2)) / (sigma * torch.sqrt(torch.tensor(2*torch.pi))) 
    
    # return tensor
    return gaussian_expansion_tensor

class LinearWithDropout(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super().__init__()
        self.linear = InitializedLinear(in_features=in_features, out_features=out_features)
        self.dropout = nn.Dropout(p=dropout_p)
    
    def forward(self, in_tensor):
        out_tensor = self.linear(in_tensor)
        out_tensor = self.dropout(out_tensor)
        return out_tensor
    
class ResidualLinearWithLayerNormAndDropout(nn.Module):
    def __init__(self, d_mod, dropout_p=0):
        # 'features', not 'in_features' and 'out_features' because residual layers can't work with in_features \neq out_features
        super().__init__()
        self.linear = LinearWithDropout(in_features=d_mod, out_features=d_mod, dropout_p=dropout_p)
        
        self.layer_norm = nn.LayerNorm(normalized_shape=d_mod)
    
    def forward(self, in_tensor):
        residual = in_tensor
        out = self.linear(in_tensor)
        out = self.layer_norm(out)
        out = out + residual
        
        return out
    
class ResidualLinearWithDropout(nn.Module):
    def __init__(self, d_mod, dropout_p=0):
        # 'features', not 'in_features' and 'out_features' because residual layers can't work with in_features \neq out_features
        super().__init__()
        self.linear = LinearWithDropout(in_features=d_mod, out_features=d_mod, dropout_p=dropout_p)
    
    def forward(self, in_tensor):
        residual = in_tensor
        out = self.linear(in_tensor)
        out = out + residual
        
        return out

class LinearWithLayerNormAndDropout(nn.Module):
    def __init__(self, in_features, out_features, dropout_p=0):
        super().__init__()
        self.linear = LinearWithDropout(in_features=in_features, out_features=out_features, dropout_p=dropout_p)
        
        self.layer_norm = nn.LayerNorm(normalized_shape=out_features)
    
    def forward(self, in_tensor):
        out_tensor = self.linear(in_tensor)
        out_tensor = self.layer_norm(out_tensor)
        
        return out_tensor

class CompleteAttentionWithLayerNorm(nn.Module):
    # complete in sense that it calculates q, k, v, then attention (then layer norm)
    def __init__(self, d_mod, n_heads):
        super().__init__()
        self.q = nn.Linear(in_features=d_mod, out_features=d_mod)
        self.k = nn.Linear(in_features=d_mod, out_features=d_mod)
        self.v = nn.Linear(in_features=d_mod, out_features=d_mod)

        self.multi_head = nn.MultiheadAttention(embed_dim=d_mod, num_heads=n_heads)
        self.layer_norm = nn.LayerNorm(normalized_shape=d_mod)
        
    def forward(self, in_tensor):
        residual = in_tensor
        q, k, v = self.q(in_tensor), self.k(in_tensor), self.v(in_tensor)
        out_tensor = self.multi_head(q, k, v)[0]
        out_tensor = self.layer_norm(out_tensor)
        out_tensor = out_tensor + residual
        
        return out_tensor

class SimpleTeamModel(nn.Module):
    def __init__(self, d_mod):
        super().__init__()
        self.act = nn.LeakyReLU()
        
        self.embedding = nn.Linear(35, d_mod)
        self.linear_1 = nn.Linear(d_mod, d_mod)
        self.linear_2 = nn.Linear(d_mod*2, d_mod*2)
        self.linear_3 = nn.Linear(d_mod*2, d_mod)
        self.prediction = nn.Linear(d_mod, 1)
        
    def forward(self, in_tensor):
        # embedding block
        x = self.embedding(in_tensor)
        x = self.act(x)
        x = self.linear_1(x) + x
        x = self.act(x)
        x = x.view(in_tensor.size(0), -1)
        x = self.act(x)
        x = self.linear_2(x) + x
        x = self.act(x)
        x = self.linear_3(x)
        x = self.act(x)
        x = self.prediction(x).squeeze(dim=1)
        return x

class SimplePlayerModel(nn.Module):
    def __init__(self, d_in, d_mod):
        super().__init__()
        self.act = nn.LeakyReLU()
        
        self.embedding = LinearWithLayerNorm(in_features=d_in, out_features=d_mod)
        
        self.post_embedding = ResidualLinearWithLayerNorm(d_mod=d_mod)
        
        self.attention_blocks = nn.ModuleList([
            CompleteAttentionWithLayerNorm(d_mod=d_mod, n_heads=4),
            CompleteAttentionWithLayerNorm(d_mod=d_mod, n_heads=4),
        ])
        
        self.player_blocks = nn.ModuleList([
            ResidualLinearWithLayerNorm(d_mod=d_mod),
            ResidualLinearWithLayerNorm(d_mod=d_mod),
        ])
        
        self.team_blocks = nn.ModuleList([
            ResidualLinearWithLayerNorm(d_mod=d_mod*2),
            ResidualLinearWithLayerNorm(d_mod=d_mod*2),
        ])
        
        self.pre_prediction = LinearWithLayerNorm(in_features=d_mod*2, out_features=d_mod)
        self.layer_norm = nn.LayerNorm(normalized_shape=d_mod)
        self.prediction = nn.Linear(in_features=d_mod, out_features=1)
    
    def forward(self, in_tensor):
        # embedding block
        players_tensor = self.embedding(in_tensor) # d_batch x n_players x 40 -> d_batch x n_players x d_mod
        
        # players attentblock
        players_tensor = self.post_embedding(players_tensor) # d_batch x n_players x d_mod
        for attention_block in self.attention_blocks:
            players_tensor = attention_block(players_tensor) # d_batch x n_players x d_mod
        for player_block in self.player_blocks:
            players_tensor = player_block(players_tensor) # d_batch x n_players x d_mod
         
          
        # teams block  
        home_tensor, away_tensor = get_home_away_tensors(in_tensor=players_tensor, original_tensor=in_tensor) # each d_batch x n_players(h/a) x d_mod
        home_tensor = home_tensor.sum(dim=1, keepdims=False) # d_batch x d_mod
        away_tensor = away_tensor.sum(dim=1, keepdims=False) # d_batch x d_mod
        teams_tensor = torch.cat([home_tensor, away_tensor], dim=1) # d_batch x (d_mod * 2)
        for team_block in self.team_blocks:
            teams_tensor = team_block(teams_tensor) # d_batch x (d_mod*2)
        
        # prediction block
        out_tensor = self.pre_prediction(teams_tensor)
        out_tensor = self.layer_norm(out_tensor)
        out_tensor = self.prediction(out_tensor).squeeze(dim=1) # d_batch x (d_mod*2) -> d_batch
        
        return out_tensor

class ScorePlayerModel(nn.Module):
    def __init__(self, d_in, d_mod):
        super().__init__()
        self.act = nn.LeakyReLU()
        
        self.embedding = LinearWithLayerNorm(in_features=d_in, out_features=d_mod)
        
        self.post_embedding = ResidualLinearWithLayerNorm(d_mod=d_mod)
        
        self.attention_blocks = nn.ModuleList([
            CompleteAttentionWithLayerNorm(d_mod=d_mod, n_heads=4),
            CompleteAttentionWithLayerNorm(d_mod=d_mod, n_heads=4),
        ])
        
        self.player_blocks = nn.ModuleList([
            ResidualLinearWithLayerNorm(d_mod=d_mod),
            ResidualLinearWithLayerNorm(d_mod=d_mod),
        ])
        
        self.team_blocks = nn.ModuleList([
            ResidualLinearWithLayerNorm(d_mod=d_mod*2),
            ResidualLinearWithLayerNorm(d_mod=d_mod*2),
        ])
        
        self.pre_prediction = LinearWithLayerNorm(in_features=d_mod*2, out_features=d_mod)
        self.layer_norm = nn.LayerNorm(normalized_shape=d_mod)
        self.prediction = nn.Linear(in_features=d_mod, out_features=2)
    
    def forward(self, in_tensor):
        # embedding block
        players_tensor = self.embedding(in_tensor) # d_batch x n_players x 40 -> d_batch x n_players x d_mod
        
        # players attentblock
        players_tensor = self.post_embedding(players_tensor) # d_batch x n_players x d_mod
        for attention_block in self.attention_blocks:
            players_tensor = attention_block(players_tensor) # d_batch x n_players x d_mod
        for player_block in self.player_blocks:
            players_tensor = player_block(players_tensor) # d_batch x n_players x d_mod
          
        # teams block  
        home_tensor, away_tensor = get_home_away_tensors(in_tensor=players_tensor, original_tensor=in_tensor) # each d_batch x n_players(h/a) x d_mod
        home_tensor = home_tensor.sum(dim=1, keepdims=False) # d_batch x d_mod
        away_tensor = away_tensor.sum(dim=1, keepdims=False) # d_batch x d_mod
        teams_tensor = torch.cat([home_tensor, away_tensor], dim=1) # d_batch x (d_mod * 2)
        for team_block in self.team_blocks:
            teams_tensor = team_block(teams_tensor) # d_batch x (d_mod*2)
        
        # prediction block
        out_tensor = self.pre_prediction(teams_tensor)
        out_tensor = self.layer_norm(out_tensor)
        out_tensor = self.prediction(out_tensor) # d_batch x (d_mod*2) -> d_batch x 2
        
        return out_tensor

class InitializedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)
    
    def forward(self, in_tensor):
        return self.linear(in_tensor)
    
class MultiHeadWrapper(nn.Module):
    def __init__(self, d_mod, n_heads):
        super().__init__()
        self.q = InitializedLinear(in_features=d_mod, out_features=d_mod)
        self.k = InitializedLinear(in_features=d_mod, out_features=d_mod)
        self.v = InitializedLinear(in_features=d_mod, out_features=d_mod)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=d_mod, num_heads=n_heads)
    
    def forward(self, in_tensor):
        q, k, v = self.q(in_tensor), self.k(in_tensor), self.v(in_tensor)
        out_tensor = self.multihead_attention(q, k, v)[0]
        return out_tensor

class GradualScorePlayerModel(nn.Module):
    def __init__(self, d_in, d_mod):
        super().__init__()
        self.d_mod, self.d_in = d_mod, d_in
        self.act = nn.LeakyReLU()
        self.dropout_p = 0.0
        
        self.embedding = nn.Sequential(
            nn.LayerNorm(normalized_shape=self.d_in),
            LinearWithDropout(in_features=self.d_in, out_features=self.d_mod*2, dropout_p=self.dropout_p),
            self.act,
            nn.LayerNorm(normalized_shape=self.d_mod*2),
            LinearWithDropout(in_features=self.d_mod*2, out_features=self.d_mod, dropout_p=self.dropout_p),
            self.act,
            nn.LayerNorm(normalized_shape=self.d_mod)
        )
        
        self.multihead_attention = nn.Sequential(
            MultiHeadWrapper(d_mod=self.d_mod, n_heads=4),
            self.act,
            nn.LayerNorm(normalized_shape=self.d_mod),
        )
        
        self.prediction_linear = nn.Sequential(
            LinearWithDropout(in_features=self.d_mod, out_features=2, dropout_p=self.dropout_p),
        )
        
    def forward(self, in_tensor):
        out_tensor = self.embedding(in_tensor)
        
        out_tensor = self.multihead_attention(out_tensor)
        
        out_tensor = self.prediction_linear(out_tensor)
        
        out_tensor = torch.mean(out_tensor, dim=1)
        return out_tensor
    
class PredictionLinear(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data = torch.tensor([0.6481, 0.6330])
        self.dropout = nn.Dropout(p=dropout_p)
        
    def forward(self, in_tensor):
        out_tensor = self.linear(in_tensor)
        out_tensor = self.dropout(out_tensor)
        return out_tensor
    
class EvenMoreGradualPlayerModel(nn.Module):
    def __init__(self, d_in, d_mod):
        super().__init__()
        self.d_in, self.d_mod = d_in, d_mod
        self.dropout_p = 0.0
        self.act = nn.ReLU()
        
        self.embedding = nn.Sequential(
            nn.LayerNorm(normalized_shape=self.d_in),
            LinearWithDropout(in_features=self.d_in, out_features=self.d_mod, dropout_p=self.dropout_p),
            self.act,
            nn.LayerNorm(normalized_shape=self.d_mod)
        )
        
        self.body = nn.Sequential(
            MultiHeadWrapper(d_mod=self.d_mod, n_heads=4),
            self.act,
            nn.LayerNorm(normalized_shape=self.d_mod)
        )
        
        self.prediction = PredictionLinear(in_features=self.d_mod, out_features=2, dropout_p=self.dropout_p)
        
    def forward(self, in_tensor):
        out_tensor = self.embedding(in_tensor)
        
        out_tensor = self.body(out_tensor)
        
        out_tensor = torch.mean(out_tensor, dim=1, keepdim=False)
        out_tensor = self.prediction(out_tensor)
        return out_tensor
        