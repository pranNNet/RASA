'''
loss.py

per-level L1 loss: one weighted mean over all query cells of all levels
'''
import torch

from model.consistency import build_levels
from model.refinement import dilate


def boundary_cells(M):
    '''
    M [B, nx, ny] -> [B, nx, ny] bool
    True for cells in M with any of their 8 neighbours outside M.
    '''
    return M & dilate(~M, 3)

def grad_l1(Y, T):
    '''
    Y, T [B, 4, n, n] predicted and true model variables
    -> mean L1 error of the cell-to-cell differences, both directions (scalar)
    '''
    dx = lambda A: A[..., 1:, :] - A[..., :-1, :]
    dy = lambda A: A[..., :, 1:] - A[..., :, :-1]
    return ((dx(Y) - dx(T)).abs().mean() + (dy(Y) - dy(T)).abs().mean()) / 2

def per_level_loss(model, out, X_next, boundary_weight,
                   level_weight=None, grad_weight=0.0, X_pred=None):
    '''
    out: from model(X0, return_levels=True)
    X_next [B, 4, N, N] raw true next frame (rho, u, v, P)
    -> loss (scalar), parts (dict of floats for logging)
    '''

    with torch.no_grad():
        T = build_levels(model.to_model_vars(X_next),
                         model.cfg['refine']['n_levels'], model.gamma)

    errs, ws, parts = [], [], {}
    for l, ((x_new, (b, i, j)), M) in enumerate(zip(out['levels'], out['masks'])):
        if len(b) == 0:
            continue
        t = T[l][b, :, i, j]    # [Q_l, 4]
        e = (x_new - t).abs()   # [Q_l, 4]
        err = e.mean(dim=1)     # [Q_l]
        w = 1 + (boundary_weight - 1) * boundary_cells(M)[b, i, j].float()
        if level_weight is not None:
            w = w * level_weight[l]
        errs.append(err)
        ws.append(w)
        parts[f'level{l}'] = err.mean().item()
        parts.setdefault('_var', []).append(e.detach())

    err, w = torch.cat(errs), torch.cat(ws)
    loss = (w * err).sum() / w.sum()

    if X_pred is not None:
        g = grad_l1(model.to_model_vars(X_pred), model.to_model_vars(X_next))
        loss = loss + grad_weight * g
        parts['grad'] = g.item()

    e_all = torch.cat(parts.pop('_var')).mean(dim=0)
    for k, name in enumerate(['log_rho', 'u', 'v', 'log_P']):
        parts[name] = e_all[k].item()
    return loss, parts
