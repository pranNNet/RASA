'''
embed.py (RPA)

router tokens. Same as v1's model/embed.py, except:
- tap diffs are dR from the payload (u, v divided by the local c_i, not c_inf)
- query uses the Mach vector instead of the absolute centre state (keeps Euler scaling exact)
- also returns the tap features, which the wave heads use as keys
'''
import torch
import torch.nn as nn

from model.stencil import OFFSETS, CENTER



class Encoder(nn.Module):
    def __init__(self, N_tap, N_query, d_model,
                 dt, dx, a_inf):
        super().__init__()

        # c_inf: reference sound speed = velocity scale of the model variables
        self.a_inf, self.dt, self.dx = a_inf, dt, dx

        self.proj_tap = nn.Linear(N_tap, d_model)
        self.proj_query = nn.Linear(N_query, d_model)

        #store offsets for pos encoding
        self.register_buffer("offsets", OFFSETS.float())   # [17, 2]

    def forward(self, taps, phi, l):
        '''
        taps [Q, 17, 4] (log-primatives), phi [Q, 17, 12] payload, l (level)
        ->
        query embedding [Q, d], tap embedding [Q, 17, d], tap features [Q, 17, 6]

        - tap features : [asinh(dR / (2^l dx)), rel cell off (x, y)]
        - query features: [Mach (u/c, v/c), l, local cfl]
        '''

        Q = taps.shape[0]

        x_q = taps[:, CENTER]

        # TAP EMBEDDING
        tap_diff = torch.asinh(phi[..., :4] / (2**l * self.dx))

        # pos encoding
        pos_off = self.offsets.expand(Q, 17, 2)

        tap_feat = torch.cat([tap_diff, pos_off], dim=-1)       # [Q, 17, 6]
        tap_embed = self.proj_tap(tap_feat)


        # QUERY EMBEDDING

        # local CFL
        c = torch.sqrt(torch.exp(x_q[:, 3] - x_q[:, 0]))
        s = torch.sqrt(x_q[:, 1] **2 + x_q[:, 2] ** 2) + c
        cfl = s * self.a_inf * self.dt / (2**l * self.dx) #[Q]

        # Mach vector (u/c, v/c): replaces v1's absolute centre state
        mach = x_q[:, 1:3] / c[:, None]  # [Q, 2]

        #level index
        lvl = torch.full_like(cfl, float(l)) # [Q]

        query_feat = torch.cat([mach, lvl[:, None], cfl[:, None]], dim=-1)   # [Q, 4]
        query_embed = self.proj_query(query_feat)
        return query_embed, tap_embed, tap_feat
    