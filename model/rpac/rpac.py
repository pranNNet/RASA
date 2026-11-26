'''
rpac.py

RPA-C: RPA with the cell update swapped for a flux update (proposal "conservative RASA").
Each query cell sends mass, momentum and energy across its 4 faces:
  send = Rusanov split send + flux units * learned correction (RPA's heads, 16 outputs)
A face's flux = what one side sends minus what the other sends back, used by both cells,
so totals are exact. Payload, encoder and router are RPA's; the heads have 16 outputs.
'''
import torch
import torch.nn.functional as F

from model.rpa.rpa import RPA
from model.rpa.payload import payload
from model.rpa.stencil_heads import StencilHeads
from model.stencil import gather_stencil, CENTER
from model.flux.rusanov import speed, send, flux_scale
from model.flux.update import conservative_forward

# (tap index in OFFSETS, axis, sign) for the 4 sends: east (+x), west (-x), north (+y), south (-y)
DIRS = [(7, 0, 1), (1, 0, -1), (5, 1, 1), (3, 1, -1)]

def base_sends(taps, gamma):
    '''
    taps [Q, 17, 4] model variables -> Rusanov split sends [Q, 4 dirs, 4 vars] (scaled conserved)
    face speed = max of |u_n| + c of the cell and that neighbour, so both sides use the same one
    '''
    Rc = taps[:, CENTER]
    out = []
    for k, d, s in DIRS:
        lam = torch.maximum(speed(Rc, d), speed(taps[:, k], d))
        out.append(send(Rc, lam, d, s, gamma))
    return torch.stack(out, dim=1)

def ring(M, cov):
    '''
    M, cov [B, n, n] bool -> cells sharing a face with M, not in M, not under a finer level
    (their sends give M's faces toward the coarser side; their own updates are thrown away)
    '''
    P = F.pad(M.float(), (1, 1, 1, 1))
    nb = P[:, :-2, 1:-1] + P[:, 2:, 1:-1] + P[:, 1:-1, :-2] + P[:, 1:-1, 2:]
    return (nb > 0) & ~M & ~cov

class RPAC(RPA):
    def build_network(self, cfg):
        '''RPA's network, with heads that output 16 sends instead of a 4-value delta'''
        super().build_network(cfg)
        self.Heads = StencilHeads(
            d_model = cfg['attn']['d_model'],
            N_heads = cfg['stencil_heads']['N_heads'],
            d_k = cfg['stencil_heads']['d_k'],
            N_tap = cfg['embed']['N_tap'],
            r_max = cfg['stencil_heads']['r_max'],
            N_out = cfg['stencil_heads']['N_out']
        )

    def local_update(self, taps, l, return_attn=False):
        '''
        taps [Q, 17, 4] model variables, l level
        ->
        sends [Q, 4 dirs (E, W, N, S), 4 vars], W list (router maps, then head maps [Q, 16, 17])
        '''
        phi = payload(taps)                         # [Q, 17, 12]
        Eq, Et, feat = self.Encoder(taps, phi, l)

        if return_attn:
            h, W = self.Router(Eq, Et, return_attn=True)
        else:
            h, W = self.Router(Eq, Et), []

        corr, alpha = self.Heads.mix(h, feat, phi)  # [Q, 16]
        if return_attn:
            W = W + [alpha]

        Rc = taps[:, CENTER]
        scale = flux_scale(Rc[:, 0].exp(), torch.exp(0.5 * (Rc[:, 3] - Rc[:, 0])))   # [Q, 4]
        return base_sends(taps, self.gamma) + scale[:, None] * corr.view(-1, 4, 4), W

    def level_faces(self, X, M, cov, l, return_attn=False):
        '''
        X [B, 4, n, n] level, M mask, cov cells under finer levels
        -> interior faces Fx_int [B, 4, n-1, n], Fy_int [B, 4, n, n-1], attention maps of the M cells
        '''
        B, _, n, _ = X.shape
        Mq = M | ring(M, cov)
        taps, (b, i, j) = gather_stencil(X, Mq)

        S = X.new_zeros(B, 4, 4, n, n)                 # [B, dir, var, n, n]
        W = []
        if len(b) > 0:
            sends, W = self.chunked_update(taps, l, return_attn)
            S[b, :, :, i, j] = sends
            keep = M[b, i, j]                          # drop ring cells from the maps
            W = [w[keep] for w in W]

        Fx_int = S[:, 0, :, :-1, :] - S[:, 1, :, 1:, :]   # east send of row k-1 minus west send of row k
        Fy_int = S[:, 2, :, :, :-1] - S[:, 3, :, :, 1:]
        return Fx_int, Fy_int, W

    def forward(self, X0, return_levels=False, return_attn=False):
        '''Same as RASA.forward, plus out['floor'] (floored cells per level). See flux/update.py'''
        return conservative_forward(self, X0, self.level_faces, return_levels, return_attn)