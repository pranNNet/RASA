'''
rasa.py

Full RASAv1 model

'''
import torch
import torch.nn as nn

from model.embed import Encoder
from model.attention import CrossAttention
from model.decode import Decoder

from model.consistency import build_levels, level_consistency
from model.refinement import refinement_masks
from model.stencil import gather_stencil, CENTER



class RASA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.cfg = cfg
        self.gamma = cfg['physics']['gamma']
        self.rho_inf = cfg['physics']['scale']['rho_inf']
        self.P_inf = cfg['physics']['scale']['P_inf']
        self.a_inf = (self.gamma * self.P_inf / self.rho_inf) ** 0.5

        self.Encoder = Encoder(
            N_tap = cfg['embed']['N_tap'],
            N_query = cfg['embed']['N_query'],
            d_model = cfg['attn']['d_model'],
            dt = cfg['physics']['domain']['dt'],
            dx = cfg['physics']['domain']['Lx']/cfg['physics']['domain']['Nx'],
            a_inf = self.a_inf
        )

        self.Attention = CrossAttention(
            d_model = cfg['attn']['d_model'],
            N_layers = cfg['attn']['N_attn_layers'],
            N_heads = cfg['attn']['N_heads'],
            N_hidden = cfg['attn']['N_hidden']
        )

        self.Decoder = Decoder(
            d_model = cfg['attn']['d_model'],
            N_hidden = cfg['decoder']['N_hidden'],
            N_out = cfg['decoder']['N_out']
        )

    def to_model_vars(self, X):
        return torch.stack([(X[:, 0] / self.rho_inf).log(),
                            X[:, 1] / self.a_inf,
                            X[:, 2] / self.a_inf,
                            (X[:, 3] / self.P_inf).log()], dim=1)

    def from_model_vars(self, Y):
        return torch.stack([Y[:, 0].exp() * self.rho_inf,
                            Y[:, 1] * self.a_inf,
                            Y[:, 2] * self.a_inf,
                            Y[:, 3].exp() * self.P_inf], dim=1)

    def forward(self, X0, return_levels=False, return_attn=False):
        '''
        X0 [B, 4, Nx, Ny] primatives U (rho, u, v, P) on finest
        ->
        dX0 [B, 4, Nx, Ny] delta U

        or
        (dX0, out) (if return_levels or return_attn = True)
        out["levels"][l] = (x_new [Q_l, 4], (b, i, j))  new values of the
                            query cells of level l and where they sit
                            (for per-level training targets later)
        out["masks"]     = list of M masks Ms[l], finest first
        out["attn"][l]   = list of n_layers tensors [Q_l, heads, 17]
                            (only if return_attn)

        '''
        #convert primatives to log: U (log(rho), u, v, log(P))
        X0_raw = X0
        X0 = self.to_model_vars(X0_raw)
        
        # ---- Build the coarse levels ----
        Xs = build_levels(X0, self.cfg['refine']['n_levels'], self.gamma) # Xs[l] [B, 4, n/2^l, n/2^l]

        # ---- Build refinement masks ----
        Ms = refinement_masks(
            levels = Xs,
            tol = self.cfg['refine']['tol'],
            w = self.cfg['refine']['dilation']
        )


        # ---- one network pass per level ----
        X_new, levels, attn = [], [], []
        for l, (X, M) in enumerate(zip(Xs, Ms)):
            #17 taps of every query cell at curr level, read from curr state
            taps, (b, i, j) = gather_stencil(X, M) # [Q, 17, 4], [Q] x3

            if len(b) == 0:
                #edge case where no query cells at curr level
                X_new.append(X)
                levels.append((taps[:, CENTER], (b, i, j)))
                attn.append([])
                continue

            # Encoder
            Eq, Et = self.Encoder(taps, l)      # [Q, d], [Q, 17, d]

            # Attention Pass
            if return_attn:
                Eq, W = self.Attention(Eq, Et, return_attn = True)
            else:
                Eq, W = self.Attention(Eq, Et, return_attn = False), []

            # Decoder predicts change
            delta = self.Decoder(Eq)     # [Q, 4]

            # Residual update: Eq = Eq + delta
            x_new = taps[:, CENTER] + delta     # [Q, 4]
            Xl = X.clone()
            Xl[b, :, i, j] = x_new

            X_new.append(Xl)
            levels.append((x_new,(b, i, j)))
            attn.append(W)


        # ---- Make the levels agree ----
        Y0 = level_consistency(X_new, Ms, self.gamma)[0]

        # ---- Return change of finest grid ----
        Y0_raw = self.from_model_vars(Y0)
        dX0 = Y0_raw - X0_raw
        if not (return_levels or return_attn):
            return dX0
        out = {'levels': levels, 'masks': Ms}
        if return_attn:
            out['attn'] = attn
        return dX0, out


