'''
positivity.py

Floor on rho' and P' after the conservative update, with a count of floored cells.
This is the only step that breaks exact conservation. The theta limiter (proposal part 8,
steps 1-2) is not built yet: the proposal starts with the floor and its log only.
'''
import torch

def floor_to_model(U, gamma, rho_min, p_min):
    '''
    U [Q, 4] scaled conserved (rho', rho'u', rho'v', E')
    ->
    R [Q, 4] model variables (log rho', u', v', log P') after the floor, n number of floored cells
    '''
    rho = U[:, 0]
    bad = rho < rho_min
    rho = rho.clamp(min=rho_min)
    u, v = U[:, 1] / rho, U[:, 2] / rho
    P = (gamma - 1) * (U[:, 3] - 0.5 * gamma * rho * (u**2 + v**2))
    bad = bad | (P < p_min)
    P = P.clamp(min=p_min)
    return torch.stack([rho.log(), u, v, P.log()], dim=-1), int(bad.sum())