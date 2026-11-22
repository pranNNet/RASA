'''
tokens.py

turns stencil 'taps" into tokens
'''
import torch
import torch.nn as nn

from model.stencil import OFFSETS, RING, CENTER



class Encoder(nn.Module):
    def __init__(self, N_tap, N_query, d_model,
                 dt, dx, a_inf):
        super().__init__()

        self.a_inf, self.dt, self.dx = a_inf, dt, dx

        self.proj_tap = nn.Linear(N_tap, d_model)
        self.proj_query = nn.Linear(N_query, d_model)

        #store offsets for pos encoding
        self.register_buffer("offsets", OFFSETS.float())   # [17, 2]


    def forward(self, taps, l):
        '''
        taps [Q, 17, 4] (log-primatives) + l (level)
        -> 
        tap embedding, query embedding

        - tap features : [diff in state from center, rel cell off (x, y)]
        - query features: [center state, l, local cfl]
        '''

        Q = taps.shape[0]

        x_q = taps[:, CENTER]

        # TAP EMBEDDING
        tap_diff = torch.asinh((taps - x_q[:, None]) / (2**l * self.dx))

        # pos encoding
        pos_off = self.offsets.expand(Q, 17, 2)

        tap_feat = torch.cat([tap_diff, pos_off], dim=-1)       # [Q, 17, 6]
        tap_embed = self.proj_tap(tap_feat)


        # QUERY EMBEDDING
    
        #local CLF number for conditioning
        c = torch.sqrt(torch.exp(x_q[:, 3] - x_q[:, 0]))
        s = torch.sqrt(x_q[:, 1] **2 + x_q[:, 2] ** 2) + c
        cfl = s * self.a_inf * self.dt / (2**l * self.dx) #[Q]

        #level index
        lvl = torch.full_like(cfl, float(l)) # [Q]

        query_feat = torch.cat([x_q, lvl[:, None], cfl[:, None]], dim=-1)
        query_embed = self.proj_query(query_feat)
        return query_embed, tap_embed

        