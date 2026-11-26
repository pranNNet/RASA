# '''
# stencil.py

# gathers the 17 taps per query cell, one level at a time 
# '''
# import torch
# import torch.nn.functional as F

# # (di, dj) in index units of level being gathered 
# OFFSETS = torch.tensor(
#     [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)] 
#     + [(-2, 0), (2, 0), (0, -2), (0, 2), (-2, -2), (-2, 2), (2, -2), (2, 2)] #(what is this code?)
# )
# RING = torch.tensor([0] * 9 + [1] * 8) 
# CENTER = 4 # row 4 of query is (0, 0)


# def gather_stencil(X, M):
#     '''
#     X [B, C, n, n] level values, M [B, n, n] bool mask 
#     ->
#     taps [Q, 17, C], (b, i, j) each [Q]

#     - Q = number of 'True' cells in M, in row-major order (same as M.nonzero)
#     - dense build: pad once by 2 with edge values, take 17 shifted slices, then keep the masked cells.
#       Edge padding = the old index clamp (cells past the edge read the nearest edge cell).
#     '''
#     b, i, j = M.nonzero(as_tuple=True)      # [Q] each
#     nx, ny = X.shape[-2:]

#     Xp = F.pad(X, (2, 2, 2, 2), mode='replicate').permute(0, 2, 3, 1)    # [B, nx+4, ny+4, C]
#     dense = torch.stack([Xp[:, 2 + di: 2 + di + nx, 2 + dj: 2 + dj + ny]
#                          for di, dj in OFFSETS.tolist()], dim=-2)       # [B, nx, ny, 17, C]
#     return dense[M], (b, i, j)


'''
stencil.py

gathers the 17 taps per query cell, one level at a time 
'''
import torch
import torch.nn.functional as F

# (di, dj) in index units of level being gathered 
OFFSETS = torch.tensor(
    [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)] 
    + [(-2, 0), (2, 0), (0, -2), (0, 2), (-2, -2), (-2, 2), (2, -2), (2, 2)] #second ring: distance-2 axis and diagonal taps
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

def gather_tiles(X, M, B, pad=1, g=2):
    '''
    BT-RASA: gather every active B x B block of a level as one tile with g ghost cells.
    X [Bt, C, n, n] level values, M [Bt, n, n] bool mask, B block size,
    pad: block count rounded up to a multiple of pad (fixed shapes), g ghost cells (2 for 17 taps)
    ->
    taps [nb * B * B, 17, C], (b, i, j) each [nb * B * B], keep [nb * B * B] bool

    - active block = any cell of M in it. Every cell of an active block is computed.
    - keep: cell is in M and its block is not padding (only these are written back / in the loss)
    - padded blocks repeat block 0 and are never kept
    - edge padding as gather_stencil, so kept cells get exactly gather_stencil's taps
    - order: block, then row, then column inside the block (whole tiles stay together in chunks)
    '''
    C, T = X.shape[1], B + 2 * g

    # active blocks
    act = F.max_pool2d(M.float()[:, None], B)[:, 0].bool()     # [Bt, n/B, n/B]
    s, bi, bj = act.nonzero(as_tuple=True)                     # [nb] each
    nb = len(s)
    n_pad = -(-nb // pad) * pad - nb
    real = torch.arange(nb + n_pad, device=X.device) < nb
    s, bi, bj = (torch.cat([t, t[:1].expand(n_pad)]) for t in (s, bi, bj))

    # pad once, then one strided view: tile (p, q) covers cells pB - g ... pB + B + g - 1
    Xp = F.pad(X, (g, g, g, g), mode='replicate')
    tiles = Xp.unfold(2, T, B).unfold(3, T, B)[s, :, bi, bj]   # [nb, C, T, T]

    # 17 shifted B x B views of each tile
    taps = torch.stack([tiles[:, :, g + di: g + di + B, g + dj: g + dj + B]
                        for di, dj in OFFSETS.tolist()], dim=1)    # [nb, 17, C, B, B]
    taps = taps.permute(0, 3, 4, 1, 2).reshape(-1, 17, C)         # [nb*B*B, 17, C]

    # cell coordinates
    r = torch.arange(B, device=X.device)
    i = (bi[:, None, None] * B + r[None, :, None]).expand(-1, B, B).reshape(-1)
    j = (bj[:, None, None] * B + r[None, None, :]).expand(-1, B, B).reshape(-1)
    b = s[:, None, None].expand(-1, B, B).reshape(-1)

    keep = M[b, i, j] & real.repeat_interleave(B * B)
    return taps, (b, i, j), keep
