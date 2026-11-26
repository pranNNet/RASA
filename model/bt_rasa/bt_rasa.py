'''
bt_rasa.py

BT-RASA (v2): RPA with narrow tap tokens and a folded router.
RPA:     payload -> Encoder -> CrossAttention (64-wide taps) -> StencilHeads (keys: 6 raw features)
BT-RASA: payload -> Encoder (query + features) -> TapLift (psi, F wide)
                 -> FoldedRouter (psi) -> StencilHeads (keys: psi)
Masks are rounded to blocks (refine.block_size); each active block is gathered as one tile with ghost cells.
Levels and level consistency are RASA's.
BTRASABase (bottom): the same on base RASA's path, MLP decoder instead of payload + stencil heads.
'''
import torch.nn as nn

from model.rpa.rpa import RPA
from model.rpa.payload import payload
from model.rpa.embed import Encoder
from model.bt_rasa.folded_router import TapLift, FoldedRouter
from model.rpa.stencil_heads import StencilHeads
from model.embed import Encoder as BaseEncoder
from model.decode import Decoder
from model.stencil import CENTER, gather_tiles

class BTRASA(RPA):
    def build_network(self, cfg):
        F = cfg['tap_lift']['F']

        self.Encoder = Encoder(
            N_tap = cfg['embed']['N_tap'],
            N_query = cfg['embed']['N_query'],
            d_model = cfg['attn']['d_model'],
            dt = cfg['physics']['domain']['dt'],
            dx = cfg['physics']['domain']['Lx']/cfg['physics']['domain']['Nx'],
            a_inf = self.a_inf
        )
        # the 6 -> 64 tap embedding is replaced by TapLift; Identity keeps it free and parameter-less
        self.Encoder.proj_tap = nn.Identity()

        self.Lift = TapLift(
            N_tap = cfg['embed']['N_tap'],
            N_lift = cfg['tap_lift']['N_lift'],
            F = F
        )

        self.Router = FoldedRouter(
            d_model = cfg['attn']['d_model'],
            N_layers = cfg['attn']['N_attn_layers'],
            N_heads = cfg['attn']['N_heads'],
            N_hidden = cfg['attn']['N_hidden'],
            F = F
        )

        self.Heads = StencilHeads(
            d_model = cfg['attn']['d_model'],
            N_heads = cfg['stencil_heads']['N_heads'],
            d_k = cfg['stencil_heads']['d_k'],
            N_tap = F,                      # keys are the tap tokens psi
            r_max = cfg['stencil_heads']['r_max']
        )

    def local_update(self, taps, l, return_attn=False):
        '''
        taps [Q, 17, 4] model variables, l level
        ->
        x_new [Q, 4], W list: router maps [Q, 4, 17] per layer, then stencil-head maps [Q, H, 17]
        '''
        phi = payload(taps)                         # [Q, 17, 12]
        Eq, _, feat = self.Encoder(taps, phi, l)    # [Q, d], -, [Q, 17, 6]
        psi = self.Lift(feat)                       # [Q, 17, F]

        if return_attn:
            h, W = self.Router(Eq, psi, return_attn=True)
            delta, alpha = self.Heads(h, psi, phi, taps[:, CENTER], return_attn=True)
            W = W + [alpha]
        else:
            h, W = self.Router(Eq, psi), []
            delta = self.Heads(h, psi, phi, taps[:, CENTER])

        return taps[:, CENTER] + delta, W

    def gather(self, X, M):
        '''
        tiles: every cell of every active block is computed, keep marks the ones written back
        '''
        return gather_tiles(X, M, self.cfg['refine']['block_size'],
                            pad=self.cfg.get('execution', {}).get('block_pad', 1))

class BTRASABase(BTRASA):
    '''
    BT-RASA on base RASA's path (MLP decoder instead of payload + stencil heads).
    Encoder (model/embed.py) -> TapLift (psi) -> FoldedRouter -> Decoder -> x_new = centre + out_scale * delta
    Block masks and tiles as BTRASA (inherits gather).
    '''
    def build_network(self, cfg):
        F = cfg['tap_lift']['F']

        self.Encoder = BaseEncoder(
            N_tap = cfg['embed']['N_tap'],
            N_query = cfg['embed']['N_query'],
            d_model = cfg['attn']['d_model'],
            dt = cfg['physics']['domain']['dt'],
            dx = cfg['physics']['domain']['Lx']/cfg['physics']['domain']['Nx'],
            a_inf = self.a_inf
        )
        # tap embedding replaced by TapLift: Encoder now returns the raw tap features [Q, 17, 6]
        self.Encoder.proj_tap = nn.Identity()

        self.Lift = TapLift(N_tap = cfg['embed']['N_tap'], N_lift = cfg['tap_lift']['N_lift'], F = F)

        self.Router = FoldedRouter(
            d_model = cfg['attn']['d_model'],
            N_layers = cfg['attn']['N_attn_layers'],
            N_heads = cfg['attn']['N_heads'],
            N_hidden = cfg['attn']['N_hidden'],
            F = F
        )

        self.Decoder = Decoder(
            d_model = cfg['attn']['d_model'],
            N_hidden = cfg['decoder']['N_hidden'],
            N_out = cfg['decoder']['N_out']
        )

    def local_update(self, taps, l, return_attn=False):
        '''
        taps [Q, 17, 4] model variables, l level
        ->
        x_new [Q, 4], W list: router maps [Q, 4, 17] per layer (empty if not return_attn)
        '''
        Eq, feat = self.Encoder(taps, l)        # [Q, d], [Q, 17, 6]
        psi = self.Lift(feat)                   # [Q, 17, F]

        if return_attn:
            h, W = self.Router(Eq, psi, return_attn=True)
        else:
            h, W = self.Router(Eq, psi), []

        return taps[:, CENTER] + self.out_scale * self.Decoder(h), W