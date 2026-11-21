'''
encoder.py

decodes cross-attention output:
    -> maps modified embedding vector to change in state
'''
import torch
import torch.nn as nn



class Decoder(nn.Module):
    def __init__(self, d_model, N_hidden, N_out = 4):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(d_model, N_hidden),
            nn.GELU(),
            nn.Linear(N_hidden, N_hidden),
            nn.GELU(),
            nn.Linear(N_hidden, N_out) #change in state 
        )

        # zero-init last layers so they start at delta = 0
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)


    def forward(self, Eq):
        '''
        Eq [Q, d] -> delta [Q, 4]
        '''
        return self.mlp(Eq)