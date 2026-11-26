'''
embed.py (RFA)

Face tokens for the router.
tap features (10): dR_k vs the face means [log rho, (u_n - u_n_bar)/c_bar, (u_t - u_t_bar)/c_bar, log P],
                   |dR_k|, offsets / 2.5
query features (8): face Mach (u_n_bar, u_t_bar)/c_bar, Courant lam dt / s, level, jump R_b - R_a (normalized as dR)
dR parts are divided by the reference-shock scales (embed.scales in the config).
Also returns the tap features, which the flux heads use as keys.
'''
import torch
import torch.nn as nn

from model.flux.rusanov import speed
from model.rfa.face_stencil import FACE_OFFSETS, A, B

class Encoder(nn.Module):
    def __init__(self, N_tap, N_query, d_model, dt, dx, c_inf, scales):
        super().__init__()

        self.c_inf, self.dt, self.dx = c_inf, dt, dx

        self.proj_tap = nn.Linear(N_tap, d_model)
        self.proj_query = nn.Linear(N_query, d_model)

        self.register_buffer("offsets", FACE_OFFSETS.float())         # [18, 2]
        self.register_buffer("scales", torch.tensor(scales).float())   # [4]

    def forward(self, taps, l):
        '''
        taps [F, 18, 4] model variables in the face frame, l level
        ->
        query embedding [F, d], tap embedding [F, 18, d], tap features [F, 18, 10]
        '''
        Fn = taps.shape[0]
        Ra, Rb = taps[:, A], taps[:, B]

        # face means (assumed: arithmetic means of rho', P', c' and the velocities of a and b)
        rho_bar = 0.5 * (Ra[:, 0].exp() + Rb[:, 0].exp())
        p_bar = 0.5 * (Ra[:, 3].exp() + Rb[:, 3].exp())
        c_bar = 0.5 * (torch.exp(0.5 * (Ra[:, 3] - Ra[:, 0])) + torch.exp(0.5 * (Rb[:, 3] - Rb[:, 0])))
        u_bar = 0.5 * (Ra[:, 1:3] + Rb[:, 1:3])                          # [F, 2] (u_n, u_t)
        ref = torch.cat([rho_bar.log()[:, None], u_bar, p_bar.log()[:, None]], dim=-1)   # [F, 4]

        one = torch.ones_like(c_bar)
        norm = torch.stack([one, c_bar, c_bar, one], dim=-1) * self.scales.to(taps)      # [F, 4]

        # TAP FEATURES
        dR = (taps - ref[:, None]) / norm[:, None]                       # [F, 18, 4]
        tap_feat = torch.cat([dR, dR.abs(), self.offsets.to(taps).expand(Fn, 18, 2) / 2.5], dim=-1)

        # QUERY FEATURES
        lam = torch.maximum(speed(Ra, 0), speed(Rb, 0))
        cfl = lam * self.c_inf * self.dt / (2**l * self.dx)
        lvl = torch.full_like(cfl, float(l))
        jump = (Rb - Ra) / norm                                          # [F, 4]
        query_feat = torch.cat([u_bar / c_bar[:, None], cfl[:, None], lvl[:, None], jump], dim=-1)   # [F, 8]

        return self.proj_query(query_feat), self.proj_tap(tap_feat), tap_feat