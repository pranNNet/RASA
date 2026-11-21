'''
consistency.py

combine all M levels and make them agree
'''
import torch
import torch.nn.functional as F



# -------------------------------------- HELPERS --------------------------------------
def to_conservative(W, gamma = 1.4):
    """
    Primitives (log rho, u, v, log p) [B, 4, h, w] 
    -> 
    conserved (rho, rho u, rho v, E).

    - based on choice of reference scales (P_inf, a_inf, rho_inf)
    """

    rho, u, v, P = W[:, 0].exp(), W[:, 1], W[:, 2], W[:, 3].exp()
    E = P/ (gamma - 1) + 0.5 * gamma *rho * (u**2 + v**2)
    return torch.stack([rho, rho*u, rho*v, E], dim=1)

def to_primitive(U, gamma = 1.4):
    """
    conserved (rho, rho u, rho v, E).
    -> 
    Primitives (log rho, u, v, log p) [B, 4, h, w] 
    """
    rho, mom_x, mom_y, E = U[:, 0], U[:, 1], U[:, 2], U[:, 3]
    u, v = mom_x/rho, mom_y/rho
    P = (gamma - 1.0) * (E - 0.5 * gamma * rho * (u**2 + v**2))
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
    Bilinear 2x upsample [B, h, 4, w] -> [B, 4, 2h, 2w]
    '''
    return F.interpolate(U, scale_factor=2, mode='bilinear', align_corners = False)

def level_consistency(Xs, Ms, gamma = 1.4):
    '''
    Xs, Ms: lists of M primative and masks arrays, finest first
    Returns a list of M complete primative arrays, same shape Xs
    '''
    M = len(Xs)

    Us = [to_conservative(x, gamma) for x in Xs]

    # update coarse map based on fine values
    fresh = [Ms[0]]
    for l in range(M-1):
        p = pool_mask(fresh[l])     # level l+1 cells under fresh level-l cells
        Us[l+1] = torch.where(p[:, None], F.avg_pool2d(Us[l],2), Us[l+1])
        fresh.append(Ms[l+1] | p)

    # update fine values based on coarse values
    for l in range(M - 2, -1, -1):
        Us[l] = torch.where(~fresh[l][:, None], upsample(Us[l+1]), Us[l])

    return [to_primitive(U, gamma) for U in Us]
        



