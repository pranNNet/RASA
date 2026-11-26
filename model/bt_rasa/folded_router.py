'''
folded_router.py (BT-RASA)

Narrow tap tokens + folded router. Same job as model/attention.py's CrossAttention
(2 layers, 4 heads, d = 64, query pre-norm, query feed-forward), but:
- each tap is an F-number token psi_k (default 16), not a 64-number one
- keys/values are linear in psi_k, so attention is done in psi-space and lifted up once:
    score_k = (W_K^T q) . psi_k + q . P_k        (key bias dropped: same for every tap)
    sum_k a_k (W_V psi_k + b_V) = W_V (sum_k a_k psi_k) + b_V
- P: learned position embedding per layer, head and tap index, starts at zero
'''
import torch
import torch.nn as nn

class TapLift(nn.Module):
    def __init__(self, N_tap=6, N_lift=32, F=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_tap, N_lift),
            nn.GELU(),
            nn.Linear(N_lift, F),
            nn.LayerNorm(F)
        )

    def forward(self, feat):
        '''
        feat [Q, 17, 6] tap features -> psi [Q, 17, F] tap tokens
        '''
        return self.net(feat)

class FoldedLayer(nn.Module):
    def __init__(self, d_model, n_heads, n_hidden, F, n_taps=17):
        super().__init__()
        self.H, self.d_h = n_heads, d_model // n_heads

        self.norm_Eq = nn.LayerNorm(d_model)
        self.W_Q = nn.Linear(d_model, d_model)

        # per-head key/value maps act on psi (F numbers)
        self.W_K = nn.Parameter(torch.randn(n_heads, self.d_h, F) / F ** 0.5)
        self.W_V = nn.Parameter(torch.randn(n_heads, self.d_h, F) / F ** 0.5)
        self.b_V = nn.Parameter(torch.zeros(n_heads, self.d_h))
        self.P = nn.Parameter(torch.zeros(n_heads, n_taps, self.d_h))

        self.W_O = nn.Linear(d_model, d_model)

        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, d_model)
        )

    def forward(self, Eq, psi):
        '''
        Eq [Q, d], psi [Q, 17, F] -> Eq [Q, d], alpha [Q, H, 17]
        '''
        Q = Eq.shape[0]
        q = self.W_Q(self.norm_Eq(Eq)).view(Q, self.H, self.d_h)      # [Q, H, d_h]

        # fold the key into psi-space, then score
        qt = torch.einsum('qhd,hdf->qhf', q, self.W_K)                  # [Q, H, F]
        scores = (torch.einsum('qhf,qkf->qhk', qt, psi)
                  + torch.einsum('qhd,hkd->qhk', q, self.P)) / self.d_h ** 0.5
        alpha = scores.softmax(dim=-1)                                  # [Q, H, 17]

        # mix in psi-space, lift up once
        m = alpha @ psi                                                 # [Q, H, F]
        v = torch.einsum('qhf,hdf->qhd', m, self.W_V) + self.b_V        # [Q, H, d_h]

        Eq = Eq + self.W_O(v.reshape(Q, -1))
        Eq = Eq + self.mlp(self.norm_mlp(Eq))
        return Eq, alpha

class FoldedRouter(nn.Module):
    def __init__(self, d_model, N_layers, N_heads, N_hidden, F):
        super().__init__()
        self.layers = nn.ModuleList(
            [FoldedLayer(d_model, N_heads, N_hidden, F) for _ in range(N_layers)]
        )

    def forward(self, Eq, psi, return_attn=False):
        '''
        Eq [Q, d] query tokens, psi [Q, 17, F] tap tokens
        -> h [Q, d]
           (and, if return_attn, a list of N_layers tensors [Q, H, 17])
        '''
        weights = []
        for layer in self.layers:
            Eq, W = layer(Eq, psi)
            weights.append(W)

        if return_attn:
            return Eq, weights
        return Eq