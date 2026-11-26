'''
payload.py (RFA)

RFA payload: split-flux differences, no learned weights.
Face references: rho_bar, c_bar = means of cells a and b; lam = max over a, b of |u_n| + c.
f+-(Q_k) = 1/2 (f_n(Q_k) +- lam Q_k), divided by the face flux units (rho c, rho c^2, rho c^2, rho c^3).
'''
import torch

from model.flux.rusanov import speed, send, flux_scale
from model.rfa.face_stencil import FACE_OFFSETS, A, B

def payload(taps, gamma):
    '''
    taps [F, 18, 4] model variables in the face frame
    ->
    phi [F, 18, 24] = [psi, dn * psi, dt * psi], psi = [f+(Q_k) - f+(Q_a), f-(Q_k) - f-(Q_b)] / scale
    base [F, 4] Rusanov flux f+(Q_a) + f-(Q_b), scale [F, 4] face flux units
    '''
    Ra, Rb = taps[:, A], taps[:, B]
    lam = torch.maximum(speed(Ra, 0), speed(Rb, 0))                      # [F]

    c = torch.exp(0.5 * (taps[..., 3] - taps[..., 0]))                   # [F, 18]
    rho_bar = 0.5 * (Ra[:, 0].exp() + Rb[:, 0].exp())
    c_bar = 0.5 * (c[:, A] + c[:, B])
    scale = flux_scale(rho_bar, c_bar)                                   # [F, 4]

    fp = send(taps, lam[:, None], 0, 1, gamma)                           # 1/2 (f + lam Q)  [F, 18, 4]
    fm = -send(taps, lam[:, None], 0, -1, gamma)                         # 1/2 (f - lam Q)
    psi = torch.cat([fp - fp[:, A, None], fm - fm[:, B, None]], dim=-1) / scale.repeat(1, 2)[:, None]

    off = FACE_OFFSETS.to(taps)                                          # [18, 2]
    phi = torch.cat([psi, off[:, 0, None] * psi, off[:, 1, None] * psi], dim=-1)
    return phi, fp[:, A] + fm[:, B], scale