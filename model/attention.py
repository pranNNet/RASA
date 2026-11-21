'''
attention.py

Cross-attention block for use as local operator
'''
import torch
import torch.nn as nn
from torch.nn.attention import sdpa_kernel, SDPBackend


class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, n_hidden):
        super().__init__()

        self.norm_Eq = nn.LayerNorm(d_model) 
        self.norm_Et = nn.LayerNorm(d_model)


        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, d_model)
        )

    def forward(self, Eq, Et, need_weights=False):
        '''
        h [Q, d], t [Q, 17, d] -> h [Q, d], w [Q, heads, 17] (None if not need_weights)
        '''

        EqN = self.norm_Eq(Eq)
        EtN = self.norm_Et(Et)

        # Eq -> Query
        # Et -> Key, Value

        with sdpa_kernel(SDPBackend.MATH):
            out, W = self.mha(EqN[:, None], EtN, EtN,
                                need_weights=need_weights,
                                average_attn_weights=False)
            # out [Q, 1, d], W [Q, heads, 1, 17]

        Eq = Eq + out[:, 0]
        Eq = Eq + self.mlp(self.norm_mlp(Eq))    # residual

        if W is not None:
            W = W[:, :, 0] # [Q, heads, 17]
        return Eq, W



class CrossAttention(nn.Module):
    def __init__(self, d_model, N_layers, N_heads, N_hidden):
        super().__init__()

        self.layers = nn.ModuleList(
            [AttentionLayer(d_model, N_heads, N_hidden) for _ in range(N_layers)]
        )


    def forward(self, Eq, Et, return_attn=False):
        '''
        h [Q, d] query tokens, t [Q, 17, d] tap tokens
        -> h [Q, d]
           (and, if return_attn, a list of n_layers tensors [Q, heads, 17])
        '''

        weights = []

        for layer in self.layers:
            Eq, W = layer(Eq, Et, need_weights=return_attn)
            weights.append(W)

        if return_attn:
            return Eq, weights
        else:
            return Eq


