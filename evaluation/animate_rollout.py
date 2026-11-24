'''
attn_diagnostics.py

What do the attention heads learn? Runs a trained RASA or RPA model on test frames, one step
each, through its own local_update. No model code is changed.
Heads are listed flat: RASA = attention layer 0, layer 1; RPA = router layer 0, 1, then the stencil heads.

  upwind.png          1  weight on upwind taps vs local Mach (row = head group, column = level)
  shock_cross.png     1b in shock cells: weight across the jump / share of taps across it (1 = no preference)
  level_swap.png      A  only the level feature changed: attention L1 change per head, and prediction change
  targeted_<type>.png B  same spot at every level: top-level pattern, then (finer level - top level)
  targeted_ratio.png  B  level-to-level distance / head-to-head distance (< 1: levels alike)
  knockout_bars.png   3  one head off: relative error increase in shock / contact / smooth regions
  knockout_maps.png   3  signed error change at knockout.map_frame, for the heads that matter most

Coordinates as stencil.py: tap offset (di, dj) = (x, y) in cells, u along x (i), v along y (j).
Run from rasa/:  python -m eval.attn_diagnostics --config configs/attn_diag.yaml [--model configs/rpa.yaml]
'''
import os
from contextlib import contextmanager

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import load_config, load_window, sample_path, split_indices
from evaluation.evaluate import box_average, dilate, model_vars
from model.consistency import build_levels
from model.refinement import refinement_masks
from model.stencil import CENTER, OFFSETS, gather_stencil

IDX = {tuple(o): k for k, o in enumerate(OFFSETS.tolist())}
UPWIND = OFFSETS[:, 0] < 0         # after canonical(): taps behind the flow (6 of 17)
TYPES = ['shock', 'contact', 'smooth']

# ------------------------------------------------ helpers ------------------------------------------------
def attention_module(model):
    return model.Router if hasattr(model, 'Router') else model.Attention

def head_groups(model):
    '''-> list of (group name, number of heads), in the order local_update returns the maps'''
    att, name = attention_module(model), 'router' if hasattr(model, 'Router') else 'layer'
    g = [(f'{name} {a}', layer.mha.num_heads) for a, layer in enumerate(att.layers)]
    if hasattr(model, 'Heads'):
        g.append(('stencil', model.Heads.H))
    return g

def canonical(vel):
    '''vel [Q, 2] (u, v) -> perm [Q, 17]: tap k sits at slot perm[q, k] once the stencil is flipped
    so the local flow points along +x (flip x if u < 0, flip y if v < 0, then swap axes if |v| > |u|).'''
    off = OFFSETS.to(vel.device).expand(len(vel), 17, 2).clone()
    off[..., 0] *= 1 - 2 * (vel[:, :1] < 0).long()
    off[..., 1] *= 1 - 2 * (vel[:, 1:] < 0).long()
    swap = (vel[:, 1].abs() > vel[:, 0].abs())[:, None, None]
    off = torch.where(swap, off.flip(-1), off)
    lut = torch.full((5, 5), -1, dtype=torch.long, device=vel.device)
    o = OFFSETS.to(vel.device)
    lut[o[:, 0] + 2, o[:, 1] + 2] = torch.arange(17, device=vel.device)
    return lut[off[..., 0] + 2, off[..., 1] + 2]

def to_canonical(A, perm):
    '''A [Q, ..., 17] -> same weights moved to their canonical slots'''
    p = perm.view(len(perm), *([1] * (A.dim() - 2)), 17).expand_as(A)
    return torch.zeros_like(A).scatter_(-1, p, A)

def tap_grad(taps, c):
    '''taps [Q, 17, 4] -> |grad| of channel c per cell [Q] (central differences, in cells)'''
    t = taps[..., c]
    gx = (t[:, IDX[(1, 0)]] - t[:, IDX[(-1, 0)]]) / 2
    gy = (t[:, IDX[(0, 1)]] - t[:, IDX[(0, -1)]]) / 2
    return torch.hypot(gx, gy)

def levels_and_masks(model, X):
    '''raw X [1, 4, N, N] -> model-variable levels Xs and the model's own query masks Ms'''
    Xs = build_levels(model.to_model_vars(X), model.cfg['refine']['n_levels'], model.gamma)
    Ms = refinement_masks(Xs, model.cfg['refine']['tol'], model.cfg['refine']['dilation'])
    return Xs, Ms

def run_level(model, taps, l, swap_to=None):
    '''taps [Q, 17, 4], level l -> x_new [Q, 4], attention of all heads [Q, K, 17].
    swap_to: change only the query's level feature (second-last query feature in both encoders);
    proj_query is linear, so that adds weight[:, level column] * (swap_to - l) to its output.'''
    hook = None
    if swap_to is not None:
        p = model.Encoder.proj_query
        add = p.weight[:, p.in_features - 2] * (swap_to - l)
        hook = p.register_forward_hook(lambda m, i, o: o + add)
    try:
        x, W = model.local_update(taps, l, return_attn=True)
    finally:
        if hook is not None:
            hook.remove()
    return x, torch.cat(W, 1)

@contextmanager
def head_off(model, k):
    '''Flat head k switched off, restored after. Attention head: its input columns of out_proj set to 0.
    RPA stencil head: its r set to 0 (the b part of its coefficient map).'''
    for g, (name, H) in enumerate(head_groups(model)):
        if k < H:
            break
        k -= H
    if name == 'stencil':
        params = [(model.Heads.W_C.data, (k, slice(None), slice(12, None))),
                  (model.Heads.b_C.data, (k, slice(12, None)))]
    else:
        mha = attention_module(model).layers[g].mha
        d = mha.embed_dim // mha.num_heads
        params = [(mha.out_proj.weight.data, (slice(None), slice(k * d, (k + 1) * d)))]
    saved = [w[i].clone() for w, i in params]
    for w, i in params:
        w[i] = 0
    try:
        yield
    finally:
        for (w, i), s in zip(params, saved):
            w[i] = s

def targeted_queries(Xs, n_per, high, low, rng):
    '''Label top-level cells shock / contact / smooth from the field, sample up to n_per of each,
    and take the nested cell under each at every finer level (2I, 2J at L1; 4I+1, 4J+1 at L0 ...).
    -> {type: (taps per level [list of [n, 17, 4]], top-level velocity [n, 2])}'''
    top = len(Xs) - 1
    taps, (b, i, j) = gather_stencil(Xs[top], torch.ones_like(Xs[top][:, 0], dtype=torch.bool))
    gP, gR = tap_grad(taps, 3), tap_grad(taps, 0)
    label = {'shock': gP > high, 'contact': (gR > high) & (gP < low), 'smooth': (gR < low) & (gP < low)}
    out = {}
    for t in TYPES:
        cand = label[t].nonzero()[:, 0].cpu().numpy()
        if len(cand) == 0:
            continue
        pick = torch.as_tensor(rng.choice(cand, min(n_per, len(cand)), replace=False), device=b.device)
        bt, it, jt = b[pick], i[pick], j[pick]
        per_level = []
        for l in range(top + 1):
            s = 2 ** (top - l)
            ii, jj, n = s * it + max(s // 2 - 1, 0), s * jt + max(s // 2 - 1, 0), Xs[l].shape[-1]
            M = torch.zeros_like(Xs[l][:, 0], dtype=torch.bool)
            M[bt, ii, jj] = True
            tl, (bl, il, jl) = gather_stencil(Xs[l], M)          # comes back in row-major order
            order = torch.searchsorted((bl * n + il) * n + jl, (bt * n + ii) * n + jj)
            per_level.append(tl[order])
        out[t] = (per_level, taps[pick, CENTER, 1:3])
    return out

def region_masks(true, high, low, band):
    '''true raw [4, n, n] -> {shock, contact, smooth} bool [n, n], same labels as targeted_queries'''
    gP, gR = (np.hypot(*np.gradient(np.log(true[c]))) for c in (3, 0))
    shock = dilate(gP > high, band)
    contact = dilate((gR > high) & (gP < low), band) & ~shock
    return {'shock': shock, 'contact': contact, 'smooth': ~(shock | contact)}

def iter_frames(cfg, data_cfg, Nx, device='cpu'):
    '''-> (sample position in split, frame, X raw [1, 4, Nx, Nx] torch, next frame raw [4, N, N] numpy)'''
    g = data_cfg['gamma']
    for n, s in enumerate(split_indices(data_cfg)[cfg['split']][:cfg['samples']]):
        for f in cfg['frames']:
            raw = load_window(sample_path(data_cfg, s), f, 1, data_cfg['step_stride'], g)
            if len(raw) < 2:
                continue
            X = torch.from_numpy(box_average(raw[0], Nx, g).astype(np.float32))[None].to(device)
            yield n, f, X, raw[1].astype(np.float64)

# ------------------------------------------------ analyses ------------------------------------------------
@torch.no_grad()
def run(model, frames, cfg, ev):
    '''frames from iter_frames -> dict of numpy arrays (head axis K = all heads, flat)'''
    L = model.cfg['refine']['n_levels']
    groups = head_groups(model)
    K = sum(H for _, H in groups)
    thr, band, R, low = ev['shock']['threshold'], ev['shock']['band'], ev['score_res'], cfg['targeted']['low']
    sc, g = model.cfg['physics']['scale'], model.gamma
    rng = np.random.default_rng(cfg['seed'])

    mach, upw, far_w, far_n = ([[] for _ in range(L)] for _ in range(4))
    ko, ko_maps, true_map = [], np.zeros(0), np.zeros(0)
    sw_attn, sw_pred, sw_n = np.zeros((L, L, K)), np.zeros((L, L)), np.zeros(L)
    tgt = {t: [[] for _ in range(L)] for t in TYPES}

    for n, f, X, true_next in frames:
        Xs, Ms = levels_and_masks(model, X)

        # ---- 1, 1b, A: the model's own query cells, per level ----
        for l in range(L):
            taps, _ = gather_stencil(Xs[l], Ms[l])
            if len(taps) == 0:
                continue
            x, A = run_level(model, taps, l)                          # A [Q, K, 17]
            xc = taps[:, CENTER]
            mach[l].append((xc[:, 1:3].norm(dim=1) / torch.exp(0.5 * (xc[:, 3] - xc[:, 0]))).cpu())
            is_up = UPWIND.to(X.device)[canonical(xc[:, 1:3])]       # [Q, 17] tap k is upwind
            upw[l].append((A * is_up[:, None]).sum(-1).cpu())

            S = tap_grad(taps, 3) > thr                               # shock cells (eval threshold, own grid)
            if S.any():
                dP = (taps[S, :, 3] - xc[S, None, 3]).abs()          # [Qs, 17]
                far = dP > 0.5 * dP.max(1, keepdim=True).values      # tap is across the jump
                far_w[l].append((A[S] * far[:, None]).sum(-1).cpu())
                far_n[l].append(far.float().mean(1).cpu())

            base = (x - xc).abs().mean()
            for l2 in range(L):
                if l2 != l:
                    x2, A2 = run_level(model, taps, l, swap_to=l2)
                    sw_attn[l, l2] += (A2 - A).abs().sum(-1).mean(0).cpu().numpy() * len(taps)
                    sw_pred[l, l2] += float((x2 - x).abs().mean() / base) * len(taps)
            sw_n[l] += len(taps)

        # ---- 3: knockout, scored like evaluate.py (score_res, model vars) per flow region ----
        true = box_average(true_next, R, g)
        mv_true = model_vars(true, sc, g)
        err = lambda: np.abs(model_vars(box_average((X + model(X))[0].double().cpu().numpy(), R, g), sc, g)
                             - mv_true).mean(0)                       # [R, R]
        e0, reg = err(), region_masks(true, thr, low, band)
        rel, maps = np.full((len(TYPES), K), np.nan), np.zeros((K, R, R))
        for k in range(K):
            with head_off(model, k):
                maps[k] = err() - e0                                  # signed: > 0 means worse without head k
            for q, t in enumerate(TYPES):
                if reg[t].any():
                    rel[q, k] = maps[k][reg[t]].mean() / e0[reg[t]].mean()
        ko.append(rel)
        if n == 0 and f == cfg['knockout']['map_frame']:
            ko_maps, true_map = maps, mv_true[0]

        # ---- B: targeted cells, same spot at every level ----
        for t, (per_level, vel) in targeted_queries(Xs, cfg['targeted']['n_per_type'], thr, low, rng).items():
            perm = canonical(vel)                                     # top cell's flow frame for all levels
            for l, taps in enumerate(per_level):
                tgt[t][l].append(to_canonical(run_level(model, taps, l)[1], perm).cpu())

    # ---- collect ----
    u = cfg['upwind']
    edges = np.linspace(0, u['mach_max'], u['n_bins'] + 1)
    r = {'head_names': np.array([f'{nm} h{h}' for nm, H in groups for h in range(H)]),
         'head_group': np.array([q for q, (_, H) in enumerate(groups) for _ in range(H)]),
         'group_names': np.array([nm for nm, _ in groups]),
         'mach_edges': edges, 'upwind_mean': np.full((L, K, u['n_bins']), np.nan),
         'upwind_count': np.zeros((L, u['n_bins'])),
         'far_w': np.full((L, K), np.nan), 'far_share': np.full(L, np.nan),
         'ko_rel': np.nanmean(ko, 0), 'ko_maps': ko_maps, 'ko_true_logrho': true_map,
         'swap_attn': sw_attn / np.maximum(sw_n, 1)[:, None, None],
         'swap_pred': sw_pred / np.maximum(sw_n, 1)[:, None], 'swap_count': sw_n}
    r['swap_attn'][sw_n == 0], r['swap_pred'][sw_n == 0] = np.nan, np.nan
    for l in range(L):
        if mach[l]:
            m, w = torch.cat(mach[l]).numpy(), torch.cat(upw[l]).numpy()
            b = np.digitize(m, edges) - 1
            for q in range(u['n_bins']):
                r['upwind_count'][l, q] = (b == q).sum()
                if r['upwind_count'][l, q]:
                    r['upwind_mean'][l, :, q] = w[b == q].mean(0)
        if far_w[l]:
            r['far_w'][l], r['far_share'][l] = torch.cat(far_w[l]).numpy().mean(0), torch.cat(far_n[l]).numpy().mean()

    # B: mean pattern per level; distance top vs finest level (paired cells) and between heads of a group
    r['tgt_mean'] = np.full((len(TYPES), L, K, 17), np.nan)
    r['tgt_level_dist'], r['tgt_head_dist'] = np.full((len(TYPES), K), np.nan), np.full((len(TYPES), K), np.nan)
    r['tgt_count'] = np.zeros(len(TYPES))
    for q, t in enumerate(TYPES):
        if not tgt[t][0]:
            continue
        A = torch.stack([torch.cat(tgt[t][l]) for l in range(L)], 1).numpy()   # [n, L, K, 17]
        r['tgt_count'][q], r['tgt_mean'][q] = len(A), A.mean(0)
        r['tgt_level_dist'][q] = np.abs(A[:, -1] - A[:, 0]).sum(-1).mean(0)
        for grp in range(len(groups)):
            k = np.flatnonzero(r['head_group'] == grp)
            a = A[:, :, k]                                                     # [n, L, H, 17]
            hh = np.abs(a[:, :, :, None] - a[:, :, None]).sum(-1)              # [n, L, H, H]
            r['tgt_head_dist'][q, k] = hh.sum(-1).mean((0, 1)) / max(len(k) - 1, 1)
    return r

# ------------------------------------------------ plots ------------------------------------------------
def _save(fig, out, name):
    fig.savefig(os.path.join(out, name), dpi=120, bbox_inches='tight')
    plt.close(fig)

def grid5(w):
    '''17 tap values -> 5x5 image [dy, dx] (NaN where there is no tap), for imshow(origin="lower")'''
    img = np.full((5, 5), np.nan)
    for k, (dx, dy) in enumerate(OFFSETS.tolist()):
        img[dy + 2, dx + 2] = w[k]
    return img

def _heatmap(ax, m, cmap, vmin=None, vmax=None):
    ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if np.isfinite(m[i, j]):
                ax.text(j, i, f'{m[i, j]:.2f}', ha='center', va='center', fontsize=7)

def make_plots(r, out, min_count=20, pattern_group=1, n_maps=4):
    os.makedirs(out, exist_ok=True)
    L, K = r['upwind_mean'].shape[:2]
    names, grp, gnames = r['head_names'], r['head_group'], r['group_names']
    G = len(gnames)
    x = np.arange(K)

    # 1: upwind share vs Mach
    mid = 0.5 * (r['mach_edges'][1:] + r['mach_edges'][:-1])
    fig, axs = plt.subplots(G, L, figsize=(3.8 * L, 2.8 * G), squeeze=False, sharey=True)
    for g in range(G):
        k_g = np.flatnonzero(grp == g)
        for l in range(L):
            ax, ok = axs[g, l], r['upwind_count'][l] >= min_count
            for n, k in enumerate(k_g):
                ax.plot(mid[ok], r['upwind_mean'][l, k, ok], color=plt.cm.tab20(n % 20), label=f'h{n}')
            ax.axhline(UPWIND.sum().item() / 17, color='grey', ls='--', lw=1)   # equal weights
            ax.axvline(1, color='grey', ls=':', lw=1)
            ax.set_title(f'{gnames[g]}, level {l}', fontsize=9)
            if g == G - 1:
                ax.set_xlabel('local Mach')
        axs[g, 0].set_ylabel('upwind weight')
        axs[g, -1].legend(fontsize=6, ncol=2 if len(k_g) > 4 else 1)
    _save(fig, out, 'upwind.png')

    # 1b: one panel, weight across the jump / share of taps across it, bars = levels
    fig, ax = plt.subplots(figsize=(1 + 0.45 * K, 3))
    wid = 0.8 / L
    for l in range(L):
        ax.bar(x + (l - (L - 1) / 2) * wid, r['far_w'][l] / r['far_share'][l], wid, label=f'level {l}')
    ax.axhline(1, color='grey', ls='--', lw=1)
    ax.set_xticks(x, names, rotation=90, fontsize=7)
    ax.set_ylabel('across-jump weight / tap share')
    ax.legend(fontsize=7)
    _save(fig, out, 'shock_cross.png')

    # A: level swap. One small matrix per head (row = true level, column = told level),
    # up to 8 heads per row; last column = relative prediction change
    cols = min(max((grp == g).sum() for g in range(G)), 8)
    rows = [(g, c) for g in range(G) for c in range(0, (grp == g).sum(), cols)]
    fig, axs = plt.subplots(len(rows), cols + 1, figsize=(1.9 * (cols + 1), 1.9 * len(rows)), squeeze=False)
    vmax = np.nanmax(r['swap_attn'])
    for ri, (g, c0) in enumerate(rows):
        k_g = np.flatnonzero(grp == g)[c0:c0 + cols]
        for c in range(cols + 1):
            ax = axs[ri, c]
            if c < len(k_g):
                _heatmap(ax, r['swap_attn'][:, :, k_g[c]], 'Blues', 0, vmax)
                ax.set_title(names[k_g[c]], fontsize=8)
            elif c == cols and ri == 0:
                _heatmap(ax, r['swap_pred'], 'Oranges', 0)
                ax.set_title('prediction\n(rel. change)', fontsize=8)
            else:
                ax.axis('off')
            ax.set_xticks(range(L))
            ax.set_yticks(range(L))
            ax.tick_params(labelsize=6)
    fig.supxlabel('told level')
    fig.supylabel('true level')
    _save(fig, out, 'level_swap.png')

    # B: per type, rows = heads of pattern_group; columns = top-level pattern, then (level l - top level)
    k_p = np.flatnonzero(grp == pattern_group)
    for q, t in enumerate(TYPES):
        if not r['tgt_count'][q]:
            continue
        P = r['tgt_mean'][q][:, k_p]                                    # [L, H, 17]
        D = P[:-1] - P[-1]                                              # finer minus top
        fig, axs = plt.subplots(len(k_p), L, figsize=(1.8 * L, 1.8 * len(k_p)), squeeze=False)
        vp, vd = np.nanmax(P[-1]), np.nanmax(np.abs(D))
        for h in range(len(k_p)):
            axs[h, 0].imshow(grid5(P[-1, h]), origin='lower', cmap='magma', vmin=0, vmax=vp)
            axs[h, 0].set_ylabel(names[k_p[h]], fontsize=7)
            for l in range(L - 1):
                axs[h, l + 1].imshow(grid5(D[l, h]), origin='lower', cmap='RdBu_r', vmin=-vd, vmax=vd)
        axs[0, 0].set_title(f'L{L - 1} weights', fontsize=8)
        for l in range(L - 1):
            axs[0, l + 1].set_title(f'L{l} - L{L - 1}', fontsize=8)
        for ax in axs.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f'{t}, {int(r["tgt_count"][q])} cells, flow to the right\n'
                     f'weights 0 to {vp:.2f}, differences -{vd:.3f} to +{vd:.3f}', fontsize=8)
        _save(fig, out, f'targeted_{t}.png')

    fig, ax = plt.subplots(figsize=(1 + 0.45 * K, 2))
    _heatmap(ax, r['tgt_level_dist'] / r['tgt_head_dist'], 'RdBu_r', 0, 2)
    ax.set_xticks(x, names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(TYPES)), TYPES)
    ax.set_title(f'L{L - 1} vs L0 distance / distance to other heads (< 1: levels alike)', fontsize=9)
    _save(fig, out, 'targeted_ratio.png')

    # 3: knockout bars (relative error increase per region) and maps for the heads that matter most
    fig, ax = plt.subplots(figsize=(1 + 0.45 * K, 3))
    for q, t in enumerate(TYPES):
        ax.bar(x + (q - 1) * 0.27, 100 * r['ko_rel'][q], 0.27, label=t)
    ax.axhline(0, color='grey', lw=1)
    ax.set_xticks(x, names, rotation=90, fontsize=7)
    ax.set_ylabel('error increase (%)')
    ax.legend(fontsize=7)
    _save(fig, out, 'knockout_bars.png')

    if r['ko_maps'].ndim == 3:
        top = np.argsort(-np.nanmax(np.abs(r['ko_rel']), 0))[:n_maps]
        fig, axs = plt.subplots(1, len(top) + 1, figsize=(2.8 * (len(top) + 1), 2.8), squeeze=False)
        axs[0, 0].imshow(r['ko_true_logrho'].T, origin='lower', cmap='viridis')
        axs[0, 0].set_title('true log rho', fontsize=8)
        v = np.nanmax(np.abs(r['ko_maps'][top]))
        for n, k in enumerate(top):
            ax = axs[0, n + 1]
            im = ax.imshow(r['ko_maps'][k].T, origin='lower', cmap='RdBu_r', vmin=-v, vmax=v)
            ax.contour(r['ko_true_logrho'].T, levels=6, colors='k', linewidths=0.3)
            ax.set_title(f'{names[k]} off', fontsize=8)
        for ax in axs.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(im, ax=axs[0, 1:].tolist(), shrink=0.8, label='error change (red = worse)')
        _save(fig, out, 'knockout_maps.png')

if __name__ == '__main__':
    import argparse
    from evaluation.evaluate import load_model
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='evaluation/configs/attn_diag.yaml')
    p.add_argument('--model', default=None, help='model config, overrides the one in --config (e.g. configs/rpa.yaml)')
    args = p.parse_args()

    cfg = load_config(args.config)
    model_cfg = load_config(args.model or cfg['model'])
    data_cfg, ev = load_config(cfg['data']), load_config(cfg['eval'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(model_cfg, device)
    frames = iter_frames(cfg, data_cfg, model_cfg['physics']['domain']['Nx'], device)
    r = run(model, frames, cfg, ev)
    out = os.path.join(cfg['out_dir'], model_cfg['name'])
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, 'attn_diag.npz'), **r)
    make_plots(r, out, cfg['upwind']['min_count'], cfg['targeted']['pattern_group'], cfg['knockout']['n_maps'])
    print('saved to', out)