'''
stencil.py

gathers the 17 taps per query cell, one level at a time 
'''
import torch
import torch.nn.functional as F

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

    - Q = number of 'True' cells in M, in row-major order (same as M.nonzero)
    - dense build: pad once by 2 with edge values, take 17 shifted slices, then keep the masked cells.
      Edge padding = the old index clamp (cells past the edge read the nearest edge cell).
    '''
    b, i, j = M.nonzero(as_tuple=True)      # [Q] each
    nx, ny = X.shape[-2:]

    Xp = F.pad(X, (2, 2, 2, 2), mode='replicate').permute(0, 2, 3, 1)    # [B, nx+4, ny+4, C]
    dense = torch.stack([Xp[:, 2 + di: 2 + di + nx, 2 + dj: 2 + dj + ny]
                         for di, dj in OFFSETS.tolist()], dim=-2)       # [B, nx, ny, 17, C]
    return dense[M], (b, i, j)


