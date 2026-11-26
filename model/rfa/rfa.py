'''
rfa.py

RFA (routed flux attention): faces are the queries, the output is the face flux.
  face taps [F, 18, 4] -> payload, Encoder -> CrossAttention (router) -> StencilHeads (flux heads)
  flux = Rusanov + face flux units * correction, in the face frame
x- and y-faces share the network (y-faces via swap). Levels, masks, refluxing, the update
and the floor are shared with RPA-C (flux/update.py). Faces between a level's cells and cells
under a finer level are not queried: they come from the finer level (refluxing).
'''
import torch

from model.rasa import RASA
from model.attention import CrossAttention
from model.rpa.stencil_heads import StencilHeads
from model.flux.update import conservative_forward

from model.rfa.face_stencil import FACE_OFFSETS, gather_faces, swap
from model.rfa.payload import payload
from model.rfa.embed import Encoder

class RFA(RASA):
    def build_network(self, cfg):
        self.Encoder = Encoder(
            N_tap = cfg['embed']['N_tap'],
            N_query = cfg['embed']['N_query'],
            d_model = cfg['attn']['d_model'],
            dt = cfg['physics']['domain']['dt'],
            dx = cfg['physics']['domain']['Lx']/cfg['physics']['domain']['Nx'],
            a_inf = self.a_inf,
            scales = cfg['embed']['scales']
        )

        self.Router = CrossAttention(
            d_model = cfg['attn']['d_model'],
            N_layers = cfg['attn']['N_attn_layers'],
            N_heads = cfg['attn']['N_heads'],
            N_hidden = cfg['attn']['N_hidden']
        )

        self.Heads = StencilHeads(
            d_model = cfg['attn']['d_model'],
            N_heads = cfg['stencil_heads']['N_heads'],
            d_k = cfg['stencil_heads']['d_k'],
            N_tap = cfg['embed']['N_tap'],
            r_max = cfg['stencil_heads']['r_max'],
            N_out = 4,
            N_val = 24,
            offsets = FACE_OFFSETS
        )

    def local_update(self, taps, l, return_attn=False):
        '''
        taps [F, 18, 4] face taps (face frame), l level
        ->
        flux [F, 4] in the face frame (mass, normal momentum, tangential momentum, energy),
        W list: router maps [F, 4, 18] per layer, then flux-head maps [F, 16, 18]
        '''
        phi, base, scale = payload(taps, self.gamma)
        Eq, Et, feat = self.Encoder(taps, l)

        if return_attn:
            h, W = self.Router(Eq, Et, return_attn=True)
        else:
            h, W = self.Router(Eq, Et), []

        corr, alpha = self.Heads.mix(h, feat, phi)     # [F, 4]
        if return_attn:
            W = W + [alpha]
        return base + scale * corr, W

    def axis_faces(self, X, M, cov, l, return_attn):
        '''
        faces between rows f and f+1 of X (face frame) -> F_int [B, 4, n-1, n], attention maps
        query faces: touch an M cell, and neither side is under a finer level
        '''
        B, _, n, _ = X.shape
        Mf = (M[:, :-1] | M[:, 1:]) & ~cov[:, :-1] & ~cov[:, 1:]
        taps, (b, f, j) = gather_faces(X, Mf)

        F_int = X.new_zeros(B, 4, n - 1, n)
        if len(b) == 0:
            return F_int, []
        fl, W = self.chunked_update(taps, l, return_attn)
        F_int[b, :, f, j] = fl
        return F_int, W

    def level_faces(self, X, M, cov, l, return_attn=False):
        '''
        X [B, 4, n, n] level, M mask, cov cells under finer levels
        -> interior faces Fx_int [B, 4, n-1, n], Fy_int [B, 4, n, n-1], attention maps (x-faces then y-faces)
        '''
        Fx, Wx = self.axis_faces(X, M, cov, l, return_attn)
        T = lambda m: m.transpose(-1, -2)
        FyT, Wy = self.axis_faces(swap(X), T(M), T(cov), l, return_attn)
        Fy = T(FyT[:, [0, 2, 1, 3]])                   # (mass, mom_y, mom_x, E) -> (mass, mom_x, mom_y, E)
        W = [torch.cat(w) for w in zip(Wx, Wy)]
        return Fx, Fy, W

    def forward(self, X0, return_levels=False, return_attn=False):
        '''Same as RASA.forward, plus out['floor'] (floored cells per level). See flux/update.py'''
        return conservative_forward(self, X0, self.level_faces, return_levels, return_attn)