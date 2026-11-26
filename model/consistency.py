'''
consistency.py

combine all M levels and make them agree
'''
import torch
import torch.nn.functional as F

# -------------------------------------- HELPERS --------------------------------------
# All conversions here work on MODEL variables (see RASA.to_model_vars):
#   rho' = rho/rho_inf, u' = u/c_inf, P' = P/P_inf, with c_inf = sqrt(gamma P_inf/rho_inf).
# Writing the energy E = P/(gamma-1) + 1/2 rho |u|^2 in those variables and dividing
# by P_inf gives
#   E' = P'/(gamma-1) + 1/2 * gamma * rho' |u'|^2      (the extra gamma = rho_inf c_inf^2 / P_inf)
# rho', rho'u', rho'v', E' are the raw conserved values divided by fixed numbers,
# so their 2x2 averages are still exact averages of mass, momentum and energy.
def to_conservative(W, gamma = 1.4):
    """
    Model primitives (log rho', u', v', log P') [B, 4, h, w] 
    -> 
    scaled conserved (rho', rho'u', rho'v', E').
    """

    rho, u, v, P = W[:, 0].exp(), W[:, 1], W[:, 2], W[:, 3].exp()
    E = P/ (gamma - 1) + 0.5 * gamma * rho * (u**2 + v**2)   # gamma: see note above
    return torch.stack([rho, rho*u, rho*v, E], dim=1)

def to_primitive(U, gamma = 1.4):
    """
    scaled conserved (rho', rho'u', rho'v', E').
    -> 
    Model primitives (log rho', u', v', log P') [B, 4, h, w] 
    """
    rho, mom_x, mom_y, E = U[:, 0], U[:, 1], U[:, 2], U[:, 3]
    u, v = mom_x/rho, mom_y/rho
    P = (gamma - 1.0) * (E - 0.5 * gamma * rho * (u**2 + v**2))   # inverse of the line above
    return torch.stack([rho.log(), u, v, P.log()], dim=1)

def build_levels(X0, n_levels, gamma = 1.4):
    '''
    Fine primatives X0 [B, 4, N, N] -> list of n_levels primative arrays, finest first

    - levels[l] is [B, 4, N/2^l, N/2^l]

    '''
    U = to_conservative(X0, gamma)
    levels = [X0]
    for _ in range(n_levels - 1):
        # avg_pool2d(U, 2)
        U = F.avg_pool2d(U, 2)
        levels.append(to_primitive(U, gamma))
    return levels

def pool_mask(M):
    '''
    [B, h, w] bool -> [B, h/2, w/2] bool
    - True if any of 2x2 children is True
    '''
    return F.max_pool2d(M.float()[:, None], 2)[:, 0].bool()

def upsample(U):
    '''
    Bilinear 2x upsample [B, 4, h, w] -> [B, 4, 2h, 2w], then each 2x2 block is shifted
    so it averages exactly back to its coarse cell (v1.1, M1). Without the shift,
    upsample then avg_pool2d blurs by [1/8, 3/4, 1/8] per direction.
    '''
    up = F.interpolate(U, scale_factor=2, mode='bilinear', align_corners = False)
    err = U - F.avg_pool2d(up, 2)                                   # [B, 4, h, w]
    return up + err.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)

def minmod_fill(U):
    '''
    Limited linear 2x fill [B, 4, h, w] -> [B, 4, 2h, 2w] (conservative variants, flux.fill: minmod).
    Each child = Q +- sx/4 +- sy/4, with minmod slopes sx, sy (zero at the domain edge).
    Each 2x2 block averages exactly back to its coarse cell, and no child goes past its
    coarse neighbours (per conserved variable).
    '''
    P = F.pad(U, (1, 1, 1, 1), mode='replicate')
    mm = lambda a, b: torch.where(a * b > 0, torch.sign(a) * torch.minimum(a.abs(), b.abs()), torch.zeros_like(a))
    sx = mm(P[..., 2:, 1:-1] - U, U - P[..., :-2, 1:-1])     # along rows (x)
    sy = mm(P[..., 1:-1, 2:] - U, U - P[..., 1:-1, :-2])     # along columns (y)

    h, w = U.shape[-2:]
    rep = lambda A: A.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
    side_x = U.new_tensor([-0.25, 0.25]).repeat(h)[:, None]  # [2h, 1]: lower child -, upper child +
    side_y = U.new_tensor([-0.25, 0.25]).repeat(w)[None, :]  # [1, 2w]
    return rep(U) + rep(sx) * side_x + rep(sy) * side_y

def level_consistency(Xs, Ms, gamma = 1.4, fill = 'bilinear'):
    '''
    Xs, Ms: lists of M primative and masks arrays, finest first
    fill: coarse-to-fine fill, 'bilinear' (v1.1 shifted upsample) or 'minmod' (minmod_fill)
    Returns a list of M complete primative arrays, same shape Xs
    '''
    M = len(Xs)
    up = minmod_fill if fill == 'minmod' else upsample

    Us = [to_conservative(x, gamma) for x in Xs]

    # update coarse map based on fine values
    fresh = [Ms[0]]
    for l in range(M-1):
        p = pool_mask(fresh[l])     # level l+1 cells under fresh level-l cells
        Us[l+1] = torch.where(p[:, None], F.avg_pool2d(Us[l],2), Us[l+1])
        fresh.append(Ms[l+1] | p)

    # update fine values based on coarse values
    for l in range(M - 2, -1, -1):
        Us[l] = torch.where(~fresh[l][:, None], up(Us[l+1]), Us[l])

    return [to_primitive(U, gamma) for U in Us]