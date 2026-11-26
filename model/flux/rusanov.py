'''
rusanov.py

Shared flux physics for the conservative variants (RPA-C, RFA).
Scaled variables as in consistency.py: Q = (rho', rho'u', rho'v', E'), speeds in units of c_inf.
Inputs R are model variables (log rho', u', v', log P'), channels last [..., 4].
In these units the x-flux is (rho'u', rho'u'^2 + P'/gamma, rho'u'v', u'(E' + P')),
and one step is Q -= (c_inf dt / s) * (face flux differences), s = level spacing.
'''
import torch

def conserved(R, gamma):
    '''R [..., 4] -> Q [..., 4]'''
    rho, u, v, P = R[..., 0].exp(), R[..., 1], R[..., 2], R[..., 3].exp()
    E = P / (gamma - 1) + 0.5 * gamma * rho * (u**2 + v**2)
    return torch.stack([rho, rho * u, rho * v, E], dim=-1)

def flux(R, d, gamma):
    '''R [..., 4] -> physical flux along x (d = 0, rows) or y (d = 1, columns) [..., 4]'''
    rho, u, v, P = R[..., 0].exp(), R[..., 1], R[..., 2], R[..., 3].exp()
    E = P / (gamma - 1) + 0.5 * gamma * rho * (u**2 + v**2)
    un, p = R[..., 1 + d], P / gamma
    return torch.stack([rho * un,
                        rho * u * un + (p if d == 0 else 0),
                        rho * v * un + (p if d == 1 else 0),
                        un * (E + P)], dim=-1)

def speed(R, d):
    '''R [..., 4] -> |u_n| + c [...], c' = sqrt(P'/rho')'''
    return R[..., 1 + d].abs() + torch.exp(0.5 * (R[..., 3] - R[..., 0]))

def flux_scale(rho, c):
    '''rho, c [...] -> flux units (rho c, rho c^2, rho c^2, rho c^3) [..., 4]'''
    return torch.stack([rho * c, rho * c**2, rho * c**2, rho * c**3], dim=-1)

def send(R, lam, d, sign, gamma):
    '''
    What a cell sends across a face along +n (sign = +1) or back along -n (sign = -1):
    1/2 (lam Q + sign f_n(Q)).  R [..., 4], lam [...] face speed -> [..., 4]
    '''
    return 0.5 * (lam[..., None] * conserved(R, gamma) + sign * flux(R, d, gamma))

def rusanov(Ra, Rb, d, gamma):
    '''
    Ra, Rb [..., 4] cells on the minus / plus side of a face -> Rusanov flux [..., 4]
    = (a sends forward) - (b sends back), lam = max of the two |u_n| + c
    '''
    lam = torch.maximum(speed(Ra, d), speed(Rb, d))
    return send(Ra, lam, d, 1, gamma) - send(Rb, lam, d, -1, gamma)