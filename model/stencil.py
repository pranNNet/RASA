'''
stencil.py

gathers the 17 taps per query cell, one level at a time 
'''
import torch

# (di, dj) in index units of level being gathered 
OFFSETS = torch.tensor(
    [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)] 
    + [(-2, 0), (2, 0), (0, -2), (0, 2), (-2, -2), (-2, 2), (2, -2), (2, 2)] #(what is this code?)
)
RING = torch.tensor([0] * 9 + [1] * 8) 
CENTER = 4 # row 4 of query is (0, 0)


def gather_stencil(X, M):
    '''
    X [B, C, n, n] level values, M [B, n, n] bool mask 
    ->
    taps [Q, 17, C], (b, i, j) each [Q]

    - Q = number of 'True' cells in M. indicices outside grid are clamped to edge (no ghost/padding cells?)
    '''
    n = X.shape[-1]

    b, i, j = M.nonzero(as_tuple=True)      # [Q] each
    di, dj = OFFSETS[:, 0].to(X.device), OFFSETS[:, 1].to(X.device)

    # row and columns of every tap
    ii = (i[:, None] + di).clamp(0, n-1)        # [Q, 17]
    jj = (j[:, None] + dj).clamp(0, n-1)        # clamp used to ensure edge cells stays at nearest edge
    taps = X.permute(0, 2, 3, 1)[b[:, None], ii, jj] # [Q, 17, C]
    return taps, (b, i, j)
