'''
rpa.py

Full RPA model: RASA v1 with the per-level network swapped.
v1:  Encoder -> CrossAttention -> Decoder -> x_new = centre + out_scale * delta
RPA: payload -> rpa Encoder -> CrossAttention (router) -> StencilHeads -> x_new = centre + delta
Everything else (model variables, levels, masks, stencil gather, level consistency) is RASA's.
'''
from model.rasa import RASA
from model.attention import CrossAttention
from model.stencil import CENTER

from model.rpa.payload import payload
from model.rpa.embed import Encoder
from model.rpa.stencil_heads import StencilHeads

class RPA(RASA):
    def build_network(self, cfg):
        self.Encoder = Encoder(
            N_tap = cfg['embed']['N_tap'],
            N_query = cfg['embed']['N_query'],
            d_model = cfg['attn']['d_model'],
            dt = cfg['physics']['domain']['dt'],
            dx = cfg['physics']['domain']['Lx']/cfg['physics']['domain']['Nx'],
            a_inf = self.a_inf
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
            r_max = cfg['stencil_heads']['r_max']
        )

    def local_update(self, taps, l, return_attn=False):
        '''
        taps [Q, 17, 4] model variables, l level
        ->
        x_new [Q, 4], W list: router maps [Q, 4, 17] per layer, then stencil-head maps [Q, H, 17]
        '''
        phi = payload(taps)     # [Q, 17, 12]
        Eq, Et, feat = self.Encoder(taps, phi, l)       # [Q, d], [Q, 17, d], [Q, 17, 6]

        if return_attn:
            h, W = self.Router(Eq, Et, return_attn=True)
            delta, alpha = self.Heads(h, feat, phi, taps[:, CENTER], return_attn=True)
            W = W + [alpha]
        else:
            h, W = self.Router(Eq, Et), []
            delta = self.Heads(h, feat, phi, taps[:, CENTER])

        # no need for output scaling
        return taps[:, CENTER] + delta, W