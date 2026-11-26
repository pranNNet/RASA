# '''
# stencil_heads.py
 
# RPA stencil heads: replace v1's decoder (model/decode.py), same job: router output -> delta [Q, 4].
# Each head is one attention step (as in model/attention.py) whose values are the fixed payload,
# then a learned rank-1 coefficient r * (l . d). Sum over heads = the update.
# '''
# import torch
# import torch.nn as nn
 
# from model.stencil import OFFSETS
 
 

# class StencilHeads(nn.Module):
#     def __init__(self, d_model, N_heads=16, d_k=8, N_tap=6, r_max=2.0):
#         super().__init__()

#         self.H, self.d_k, self.r_max = N_heads, d_k, r_max

#         # query from router input, keys from tap features (no bias)
#         self.W_Q = nn.Linear(d_model, N_heads * d_k, bias=False)
#         self.W_K = nn.Parameter(torch.randn(N_heads, d_k, N_tap) / N_tap ** 0.5)

#         # per-head coefficient map [h, dbar] (d_model + 2) -> a (12) | b (4)
#         self.W_C = nn.Parameter(torch.randn(N_heads, d_model + 2, 16) / (d_model + 2) ** 0.5)
#         self.b_C = nn.Parameter(torch.zeros(N_heads, 16))

#         # zero-init the b part so r = 0, and starts at delta = 0
#         with torch.no_grad():
#             self.W_C[..., 12:] = 0

#         self.register_buffer('offsets', OFFSETS.float()) # [17, 2]


#     def forward(self, h, feat, phi, x_c, return_attn=False):
#         '''
#         h [Q, d] router output, feat [Q, 17, 6] tap features (keys)
#         phi [Q, 17, 2] payload (values), x_c [Q, 4] center tap (model variables)
#         ->
#         delta [Q, 4] log rho', u', v', log P')
#         (and, if return_attn, alpha [Q, H, 17])
#         '''
#         Q, H = h.shape[0], self.H

#         # stencil weights
#         q = self.W_Q(h).view(Q, H, self.d_k)        # [Q, H, d_k]
#         kq = torch.einsum('qhd,hdf->qhf', q, self.W_K)      # [Q, H, 6]
#         scores = torch.einsum('qhf,qkf->qhk', kq, feat) / self.d_k ** 0.5
#         alpha = scores.softmax(dim=-1)      # [Q, H, 17]

#         # readout and head direction
#         d = alpha @ phi     # [Q, H, 12]
#         dbar = alpha @ self.offsets.to(phi)     # [Q, H, 2]

#         # coefficients
#         inp = torch.cat([h[:, None].expand(Q, H, -1), dbar], dim=-1)        # [Q, H, d+1]
#         coef = torch.einsum('qhi,hio->qho', inp, self.W_C) + self.b_C       # [Q, H, 16]
#         a, b = coef[..., :12], coef[..., 12:]
#         l = a / torch.sqrt((a **2).sum(-1, keepdim=True) + 1e-6)        # [Q, H, 12], |l| < 1
#         r = self.r_max * torch.tanh(b)      # [Q, 4]

#         # update: sum over heads
#         dR = (r * (l * d).sum(-1, keepdim = True)).sum(dim=1)

#         # u/c_i -> u' = u/c_inf: multiply velocity parts by c' = sqrt(P'/rho')
#         c = torch.exp(0.5 * (x_c[:, 3] - x_c[:, 0]))
#         one = torch.ones_like(c)
#         delta = dR * torch.stack([one, c, c, one], dim=-1)              # [Q, 4]
 
#         if return_attn:
#             return delta, alpha
#         return delta

'''
stencil_heads.py
 
RPA stencil heads: replace v1's decoder (model/decode.py), same job: router output -> delta [Q, 4].
Each head is one attention step (as in model/attention.py) whose values are the fixed payload,
then a learned rank-1 coefficient r * (l . d). Sum over heads = the update.
Also used by RPA-C (N_out = 16 sends) and RFA (N_val = 24, face offsets) through mix().
'''
import torch
import torch.nn as nn
 
from model.stencil import OFFSETS
 
 

class StencilHeads(nn.Module):
    def __init__(self, d_model, N_heads=16, d_k=8, N_tap=6, r_max=2.0,
                 N_out=4, N_val=12, offsets=OFFSETS):
        super().__init__()

        self.H, self.d_k, self.r_max, self.N_val = N_heads, d_k, r_max, N_val

        # query from router input, keys from tap features (no bias)
        self.W_Q = nn.Linear(d_model, N_heads * d_k, bias=False)
        self.W_K = nn.Parameter(torch.randn(N_heads, d_k, N_tap) / N_tap ** 0.5)

        # per-head coefficient map [h, dbar] (d_model + 2) -> a (N_val) | b (N_out)
        self.W_C = nn.Parameter(torch.randn(N_heads, d_model + 2, N_val + N_out) / (d_model + 2) ** 0.5)
        self.b_C = nn.Parameter(torch.zeros(N_heads, N_val + N_out))

        # zero-init the b part so r = 0, and starts at delta = 0
        with torch.no_grad():
            self.W_C[..., N_val:] = 0

        self.register_buffer('offsets', offsets.float()) # [17, 2] (RPA)

    def mix(self, h, feat, phi):
        '''
        h [Q, d] router output, feat [Q, K, N_tap] tap features (keys), phi [Q, K, N_val] payload (values)
        ->
        sum over heads of r * (l . d) [Q, N_out], alpha [Q, H, K]
        '''
        Q, H = h.shape[0], self.H

        # stencil weights
        q = self.W_Q(h).view(Q, H, self.d_k)        # [Q, H, d_k]
        kq = torch.einsum('qhd,hdf->qhf', q, self.W_K)      # [Q, H, N_tap]
        scores = torch.einsum('qhf,qkf->qhk', kq, feat) / self.d_k ** 0.5
        alpha = scores.softmax(dim=-1)      # [Q, H, K]

        # readout and head direction
        d = alpha @ phi     # [Q, H, N_val]
        dbar = alpha @ self.offsets.to(phi)     # [Q, H, 2]

        # coefficients
        inp = torch.cat([h[:, None].expand(Q, H, -1), dbar], dim=-1)        # [Q, H, d+2]
        coef = torch.einsum('qhi,hio->qho', inp, self.W_C) + self.b_C       # [Q, H, N_val + N_out]
        a, b = coef[..., :self.N_val], coef[..., self.N_val:]
        l = a / torch.sqrt((a **2).sum(-1, keepdim=True) + 1e-6)        # [Q, H, N_val], |l| < 1
        r = self.r_max * torch.tanh(b)      # [Q, H, N_out]

        return (r * (l * d).sum(-1, keepdim = True)).sum(dim=1), alpha     # [Q, N_out], [Q, H, K]

    def forward(self, h, feat, phi, x_c, return_attn=False):
        '''
        RPA: h [Q, d] router output, feat [Q, 17, 6] tap features (keys)
        phi [Q, 17, 12] payload (values), x_c [Q, 4] center tap (model variables)
        ->
        delta [Q, 4] (log rho', u', v', log P')
        (and, if return_attn, alpha [Q, H, 17])
        '''
        # update: sum over heads
        dR, alpha = self.mix(h, feat, phi)      # [Q, 4]

        # u/c_i -> u' = u/a_inf: multiply velocity parts by c' = sqrt(P'/rho')
        c = torch.exp(0.5 * (x_c[:, 3] - x_c[:, 0]))
        one = torch.ones_like(c)
        delta = dR * torch.stack([one, c, c, one], dim=-1)              # [Q, 4]
 
        if return_attn:
            return delta, alpha
        return delta