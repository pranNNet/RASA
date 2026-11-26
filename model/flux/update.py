'''
update.py

Shared step after the network for RPA-C and RFA: domain-edge faces, refluxing,
the conservative update and the floor. conservative_forward is the forward pass both call.

Face arrays on a level with n x n cells:
  Fx [B, 4, n+1, n]: Fx[:, :, k, j] is the face between rows k-1 and k (k = 0 and n are domain edges)
  Fy [B, 4, n, n+1]: same along columns
'''
import torch
import torch.nn.functional as F

from model.consistency import build_levels, level_consistency, pool_mask, to_conservative
from model.refinement import refinement_masks
from model.flux.rusanov import flux
from model.flux.positivity import floor_to_model

def with_edges(Fx_int, Fy_int, X, gamma):
    '''
    interior faces Fx_int [B, 4, n-1, n], Fy_int [B, 4, n, n-1], level X [B, 4, n, n] model variables
    -> Fx, Fy with domain-edge faces added: outflow, so the edge cell's physical flux, no correction
    '''
    R = X.permute(0, 2, 3, 1)
    fx = flux(R, 0, gamma).permute(0, 3, 1, 2)
    fy = flux(R, 1, gamma).permute(0, 3, 1, 2)
    Fx = torch.cat([fx[:, :, :1], Fx_int, fx[:, :, -1:]], dim=2)
    Fy = torch.cat([fy[..., :1], Fy_int, fy[..., -1:]], dim=3)
    return Fx, Fy

def reflux(Fx, Fy, Fx_fine, Fy_fine, cov):
    '''
    Coarse faces next to a cell covered by the finer level take the mean of the two fine faces under them.
    Fx, Fy coarse faces, Fx_fine [B, 4, 2n+1, 2n], Fy_fine [B, 4, 2n, 2n+1], cov [B, n, n] bool
    -> Fx, Fy
    '''
    Rx = (Fx_fine[:, :, ::2, 0::2] + Fx_fine[:, :, ::2, 1::2]) / 2     # [B, 4, n+1, n]
    Ry = (Fy_fine[:, :, 0::2, ::2] + Fy_fine[:, :, 1::2, ::2]) / 2     # [B, 4, n, n+1]

    c = cov.float()
    cx = F.pad(c, (0, 0, 1, 1))                                       # [B, n+2, n]
    cy = F.pad(c, (1, 1))                                             # [B, n, n+2]
    mx = (cx[:, :-1] + cx[:, 1:]) > 0                                 # face touches a covered cell
    my = (cy[..., :-1] + cy[..., 1:]) > 0
    return torch.where(mx[:, None], Rx, Fx), torch.where(my[:, None], Ry, Fy)

def update_cells(X, Fx, Fy, M, k, gamma, rho_min, p_min):
    '''
    X [B, 4, n, n] model variables, faces Fx, Fy, M [B, n, n] bool, k = c_inf dt / s
    ->
    x_new [Q, 4] model variables of the M cells, (b, i, j), number of floored cells
    Q_new = Q - k (F_E - F_W + F_N - F_S), only at the M cells.
    '''
    b, i, j = M.nonzero(as_tuple=True)
    div = (Fx[:, :, 1:] - Fx[:, :, :-1]) + (Fy[..., 1:] - Fy[..., :-1])  # [B, 4, n, n]
    U = to_conservative(X, gamma)[b, :, i, j] - k * div[b, :, i, j]     # [Q, 4]
    x_new, n = floor_to_model(U, gamma, rho_min, p_min)
    return x_new, (b, i, j), n

def conservative_forward(model, X0_raw, level_faces, return_levels=False, return_attn=False):
    '''
    Same inputs and outputs as RASA.forward, plus out['floor'] = floored cells per level.
    level_faces(X, M, cov, l, return_attn) -> interior Fx_int, Fy_int, attention list
      (cov [B, n, n] = cells of level l covered by finer levels)
    '''
    cfg, gamma = model.cfg, model.gamma
    dom, fcfg = cfg['physics']['domain'], cfg['flux']
    dx = dom['Lx'] / dom['Nx']

    X0 = model.to_model_vars(X0_raw)
    Xs = build_levels(X0, cfg['refine']['n_levels'], gamma)
    Ms = refinement_masks(levels=Xs, tol=cfg['refine']['tol'], w=cfg['refine']['dilation'])

    X_new, levels, attn, floor = [], [], [], []
    cov = torch.zeros_like(Ms[0])
    for l, (X, M) in enumerate(zip(Xs, Ms)):
        if l > 0:
            cov = pool_mask(Ms[l - 1] | cov)       # level-l cells under any finer level

        Fx_int, Fy_int, W = level_faces(X, M, cov, l, return_attn)
        Fx, Fy = with_edges(Fx_int, Fy_int, X, gamma)
        if l > 0:
            Fx, Fy = reflux(Fx, Fy, Fx_f, Fy_f, cov)
        Fx_f, Fy_f = Fx, Fy

        k = model.c_inf * dom['dt'] / (2**l * dx)
        x_new, (b, i, j), n = update_cells(X, Fx, Fy, M, k, gamma, fcfg['rho_min'], fcfg['p_min'])
        Xl = X.clone()
        Xl[b, :, i, j] = x_new

        X_new.append(Xl)
        levels.append((x_new, (b, i, j)))
        attn.append(W)
        floor.append(n)

    Y0 = level_consistency(X_new, Ms, gamma, fill=fcfg.get('fill', 'bilinear'))[0]
    dX0 = model.from_model_vars(Y0) - X0_raw
    if not (return_levels or return_attn):
        return dX0
    out = {'levels': levels, 'masks': Ms, 'floor': floor}
    if return_attn:
        out['attn'] = attn
    return dX0, out