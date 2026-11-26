'''
face_stencil.py

RFA face stencil: 18 taps per face, in the face's frame (normal n, tangent t).
Faces are gathered along rows (x-faces). y-faces use the same code on swap(X):
transposed, with u and v swapped, so u_n is always channel 1 and u_t channel 2.
'''
import torch
import torch.nn.functional as F

# (row, column) of each tap relative to cell a (the minus side): rows a-2 .. a+3, columns -1 .. 1
CELLS = [(di, dj) for di in range(-2, 4) for dj in (-1, 0, 1)]
# offsets from the face centre, in units of the level spacing: delta_n in +-0.5, +-1.5, +-2.5; delta_t in -1, 0, 1
FACE_OFFSETS = torch.tensor([(di - 0.5, dj) for di, dj in CELLS])   # [18, 2]
A, B = 7, 10    # taps of cell a (delta_n = -0.5) and cell b (+0.5), on the face line

def swap(X):
    '''[B, 4, n, n] model variables -> transposed, with u and v swapped (y-faces become x-faces)'''
    return X.transpose(-1, -2)[:, [0, 2, 1, 3]]

def gather_faces(X, Mf):
    '''
    X [B, C, n, n] level values (in the face frame), Mf [B, n-1, n] bool: face between rows f and f+1
    ->
    taps [F, 18, C], (b, f, j) each [F]
    Edge padding as gather_stencil.
    '''
    b, f, j = Mf.nonzero(as_tuple=True)
    nx, ny = X.shape[-2:]

    Xp = F.pad(X, (1, 1, 2, 2), mode='replicate').permute(0, 2, 3, 1)      # [B, nx+4, ny+2, C]
    dense = torch.stack([Xp[:, 2 + di: 2 + di + nx - 1, 1 + dj: 1 + dj + ny]
                         for di, dj in CELLS], dim=-2)                     # [B, nx-1, ny, 18, C]
    return dense[Mf], (b, f, j)