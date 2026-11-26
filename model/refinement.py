# '''
# refinement.py

# refinement indicator calculations and masking for levels

# assignments refinements level to each part of domain
# '''
# import torch
# import torch.nn.functional as F


# def grad_indicator(W):
#     '''
#     W [B, 4, n, n] primatives (log(rho), u, v, log(P)) 
#     ->
#     g [B, n, n] |grad log

#     Central Diff. w/ outflow BCs (replicate padding)
#     - grid spacing independent
#     '''

#     f = F.pad(W[:, [0, 3]], (1, 1, 1, 1), mode='replicate') # [B, 2, n+2, n+2]

#     dx = (f[..., 2:, 1:-1] - f[..., :-2, 1:-1]) / 2 # [B, 2, n, n]
#     dy = (f[..., 1:-1, 2:] - f[..., 1:-1, :-2]) / 2  # y along columns (dim 3, j)

#     return torch.sqrt(dx**2 + dy**2).sum(dim=1)

# # -------------------------------------- HELPERS --------------------------------------
# # Masks are bool [B, n, n]

# def dilate(r, k):
#    '''
#    grow True regions by k//2 cells on every side
#    - cell is true if any cell in k x k window is True
#    '''  
#    return F.max_pool2d(r.float()[:, None], k, stride=1, padding=k//2)[:, 0].bool()

# def up(r, s=2):
#     '''
#     Raise resolution by s: copy cells into s x s block
#     '''
#     return r.repeat_interleave(s, dim=1).repeat_interleave(s, dim=2) 

# def down(r):
#     '''
#     Halve resolution: coarse cell is True if ANY of its 2 x 2 fine cells is
#     '''
#     return F.max_pool2d(r.float()[:, None], 2)[:, 0].bool()

# def refinement_masks(levels, tol, w, indicator = 'lvf'):
#     '''
#     levels: list of M primative arrays (fine -> coarse)
#         - levels[l] : [B, 4, N/2^l, N/2^l]
#     tol: refine a cell if g > tol
#     w: dilation, in cells
#     ->
#     masks: list of M bool arrays (fine -> coarse)
#         - masks[l] : [B, N/2^l, N/2^l]
#         - True where l is used
#     '''
#     M = len(levels)

#     # region[l] = cells on level l or finer (on l's grid)
#     region = [None] * M
#     region[M - 1] = torch.ones_like(grad_indicator(levels[M - 1]), dtype=torch.bool)


#     # Pass 1: refine cells 
#     for l in range(M - 1, 0, -1):
#         g = grad_indicator(levels[l])

#         # only cells in this level's region can be refined
#         mark = (g > tol) & region[l]
#         # dilate as needed
#         mark = dilate(mark, 2 * w + 1)
#         # one level-l cell = 2x2 cells of level l-1
#         region[l - 1] = up(mark)

#     # Pass 2: coursen as needed
#     for l in range(M - 1):
#         child = down(region[l]) # child region on parent's grid
#         region[l + 1] = region[l + 1] | dilate(child, 3) # cover child + 1-cell ring

#         if l + 1 < M - 1:
#             # each cell of level must be whole cell of level above it
#             region[l + 1] = up(down(region[l + 1]))


#     # Final masks: level l = region minus finer parts
#     masks = [region[0]]
#     for l in range(1, M):
#         masks.append(region[l] & ~down(region[l - 1]))
#     return masks



'''
refinement.py

refinement indicator calculations and masking for levels

assignments refinements level to each part of domain
'''
import torch
import torch.nn.functional as F

def grad_indicator(W):
    '''
    W [B, 4, n, n] primatives (log(rho), u, v, log(P)) 
    ->
    g [B, n, n] |grad log

    Central Diff. w/ outflow BCs (replicate padding)
    - grid spacing independent
    '''

    f = F.pad(W[:, [0, 3]], (1, 1, 1, 1), mode='replicate') # [B, 2, n+2, n+2]

    dx = (f[..., 2:, 1:-1] - f[..., :-2, 1:-1]) / 2 # [B, 2, n, n]
    dy = (f[..., 1:-1, 2:] - f[..., 1:-1, :-2]) / 2  # y along columns (dim 3, j)

    return torch.sqrt(dx**2 + dy**2).sum(dim=1)

# -------------------------------------- HELPERS --------------------------------------
# Masks are bool [B, n, n]

def dilate(r, k):
   '''
   grow True regions by k//2 cells on every side
   - cell is true if any cell in k x k window is True
   '''  
   return F.max_pool2d(r.float()[:, None], k, stride=1, padding=k//2)[:, 0].bool()

def up(r, s=2):
    '''
    Raise resolution by s: copy cells into s x s block
    '''
    return r.repeat_interleave(s, dim=1).repeat_interleave(s, dim=2) 

def down(r):
    '''
    Halve resolution: coarse cell is True if ANY of its 2 x 2 fine cells is
    '''
    return F.max_pool2d(r.float()[:, None], 2)[:, 0].bool()

def round_up(r, B):
    '''
    Round a region up to whole B x B blocks: a block is True if ANY of its cells is
    '''
    return up(F.max_pool2d(r.float()[:, None], B)[:, 0].bool(), B)

def block_waste(cell_masks, block_masks, B):
    '''
    w = cells computed with blocks / cells computed with per-cell masks
    - with blocks, every block holding a cell of M_l is computed whole (B x B cells of level l)
    '''
    blocks = sum(int(F.max_pool2d(M.float()[:, None], B).sum()) for M in block_masks)
    cells = sum(int(M.sum()) for M in cell_masks)
    return blocks * B * B / cells

def refinement_masks(levels, tol, w, indicator = 'lvf', B = None):
    '''
    levels: list of M primative arrays (fine -> coarse)
        - levels[l] : [B, 4, N/2^l, N/2^l]
    tol: refine a cell if g > tol
    w: dilation, in cells
    B: block size (BT-RASA). None = per-cell masks (v1.1). If set, each level's region is
       rounded up to whole B x B blocks of that level before the coarser level is built from it
    ->
    masks: list of M bool arrays (fine -> coarse)
        - masks[l] : [B, N/2^l, N/2^l]
        - True where l is used
    '''
    M = len(levels)

    # region[l] = cells on level l or finer (on l's grid)
    region = [None] * M
    region[M - 1] = torch.ones_like(grad_indicator(levels[M - 1]), dtype=torch.bool)

    # Pass 1: refine cells 
    for l in range(M - 1, 0, -1):
        g = grad_indicator(levels[l])

        # only cells in this level's region can be refined
        mark = (g > tol) & region[l]
        # dilate as needed
        mark = dilate(mark, 2 * w + 1)
        # one level-l cell = 2x2 cells of level l-1
        region[l - 1] = up(mark)

    # Pass 2: coursen as needed
    for l in range(M - 1):
        if B:
            region[l] = round_up(region[l], B)
        child = down(region[l]) # child region on parent's grid
        region[l + 1] = region[l + 1] | dilate(child, 3) # cover child + 1-cell ring

        if l + 1 < M - 1:
            # each cell of level must be whole cell of level above it
            region[l + 1] = up(down(region[l + 1]))

    # Final masks: level l = region minus finer parts
    masks = [region[0]]
    for l in range(1, M):
        masks.append(region[l] & ~down(region[l - 1]))
    return masks