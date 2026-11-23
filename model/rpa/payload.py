'''
rpa.py

RPA (Routed Payload Attention), v1's primary ablation. Replaces Encoder + CrossAttention + Decoder.
- payload: the values the wave heads mix. Fixed stencil differences, no learned weights.
- router: v1's CrossAttention (imported from attention.py) on RPA's own tokens   [to come]
- wave heads + update                                                            [to come]
'''
import torch

from model.stencil import OFFSETS, CENTER



def payload(taps):
    '''
    taps [Q, 17, 4] (log rho', u', v', log P') from gather_stencil
    ->
    phi [Q, 17, 12] = [dR, dx*dR, dy*dR]

    - dR = (dlog rho, du/c_i, dv/c_i, dlog P), difference from the centre tap
    - c_i in units of c_inf: c' = sqrt(P'/rho') = exp((log P' - log rho')/2)
    - dx = di (rows = x), dy = dj (columns = y), in units of the level's spacing
    '''

    x_c = taps[:, CENTER]       # [Q, 4]
    c = torch.exp(0.5 * (x_c[:, 3] - x_c[:, 0]))        # [Q]
    one = torch.ones_like(c)
    scale = torch.stack([one, 1 / c, 1 / c, one], -1)   # [Q, 4]

    dR = (taps - x_c[:, None]) * scale[:, None]     # [Q, 17, 4]

    off = OFFSETS.to(taps)
    return torch.cat([dR, off[:, 0, None] * dR, off[:, 1, None] * dR], dim=-1)    


