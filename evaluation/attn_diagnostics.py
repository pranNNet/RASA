'''
attn_diagnostics.py

Three questions about the attention heads of a trained RASA or RPA model, a few simple figures each.
One model step per test frame. Assumes inviscid Euler data: there is no physical smoothing, so any
smoothing the model does is numerics.

Effective stencil of a cell = how much each of its 17 taps moves its update (gradient of the update
with respect to the taps). Its lopsided part moves things (transport, with a speed); its symmetric part
smooths them (with a smoothing strength). Measured in smooth cells only, where this is reliable.

H1  Is the model learning physics or numerics?
    h1_transport.png       learned transport speed vs true flow speed (physics: on the y = x line)
    h1_pressure_force.png  learned pressure -> velocity push vs the Euler value (physics: on the y = x line)
    h1_smoothing.png       learned smoothing vs cell size (numerics: grows with cell size)
    h1_direct.png          check without gradients: predicted vs true change per level (1 = right amount)
H2  Is each head self-similar across levels?
    h2_level_dependence.png  attention change when a cell is told another level, vs untrained heads
    h2_selectivity.png       how much each head picks specific taps vs how much it changes with level
    h2_scaling.png / .md     a cell's own values run as every level: does its change scale as Euler says? (1 = yes)
    h2_patterns.png          two heads' average attention at every level, same spots, same colours
H3  What role does each head play? (a head can do physics, numerics, both or neither)
    h3_roles.png  per head: physics it provides vs smoothing it provides (dot size = importance), with a zoom
    h3_roles.md   the same as a table with 90% intervals and a role label

Run from rasa/:  python -m eval.attn_diagnostics --config evaluation/configs/attn_diag.yaml [--model configs/rpa.yaml]
'''
import os
import warnings
from contextlib import contextmanager

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch

from preprocess.data import load_config, load_window, sample_path, split_indices
from evaluation.evaluate import box_average, model_vars
from model.consistency import build_levels
from model.refinement import refinement_masks
from model.stencil import CENTER, OFFSETS, gather_stencil

IDX = {tuple(o): k for k, o in enumerate(OFFSETS.tolist())}   # tap offset (x, y) -> tap index
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

# ------------------------------------------------ effective stencil ------------------------------------------------
def effective_stencil(model, taps, l):
    '''taps [Q, 17, 4] model variables -> J [Q, 4, 17, 4] = d x_new[q, a] / d taps[q, k, b].
    Each cell's update only depends on its own taps, so one backward pass per output variable is enough.'''
    with torch.enable_grad():
        t = taps.detach().clone().requires_grad_(True)
        x, _ = model.local_update(t, l)
        J = [torch.autograd.grad(x[:, a].sum(), t, retain_graph=a < 3)[0] for a in range(4)]
    return torch.stack(J, 1)

def stencil_numbers(J, taps, l, model):
    '''J [Q, 4, 17, 4], taps [Q, 17, 4] at level l -> dict of per-cell numpy arrays [Q], physical units:
    u, c                 true flow speed |u| and sound speed
    speed_rho, speed_P   learned transport speed along the flow, from the log rho -> log rho and
                         log P -> log P stencils: first moment m1 (cells), speed = -m1 . flow direction * dx_l / dt
    force, force_true    learned pressure push on velocity and the Euler value. Euler: du'/dt = -(c'^2 c_inf / gamma)
                         grad log P in model variables, so force = -gamma dx_l / (c_inf dt) * m1 of the
                         log P -> u' (x) and log P -> v' (y) stencils (averaged); Euler value = c'^2 = P'/rho'
    nu_rho, nu_P         learned smoothing: spread of the stencil around m1, nu = trace(var) dx_l^2 / (4 dt)
                         (first-order upwind gives |u| dx (1 - CFL) / 4 per direction averaged over x and y)'''
    o = OFFSETS.to(J)                                                  # [17, 2] cells, (x, y)
    dxl, dt, cinf = 2 ** l * model.Encoder.dx, model.Encoder.dt, model.a_inf
    xc = taps[:, CENTER]
    vel = xc[:, 1:3] * cinf
    speed = vel.norm(dim=1)
    flow = vel / speed.clamp_min(1e-12)[:, None]
    out = {'u': speed, 'c': torch.exp(0.5 * (xc[:, 3] - xc[:, 0])) * cinf}
    for name, a in [('rho', 0), ('P', 3)]:
        w = J[:, a, :, a]                                              # [Q, 17]
        m1 = w @ o                                                     # [Q, 2]
        var = torch.einsum('qk,ki,kj->qij', w, o, o) - m1[:, :, None] * m1[:, None, :]
        out[f'speed_{name}'] = -(m1 * flow).sum(1) * dxl / dt
        out[f'nu_{name}'] = (var[:, 0, 0] + var[:, 1, 1]) * dxl ** 2 / (4 * dt)
    m_u, m_v = J[:, 1, :, 3] @ o, J[:, 2, :, 3] @ o
    out['force'] = -model.gamma * dxl / (2 * cinf * dt) * (m_u[:, 0] + m_v[:, 1])
    out['force_true'] = torch.exp(xc[:, 3] - xc[:, 0])
    return {k: v.detach().cpu().numpy() for k, v in out.items()}

def scaling_sums(model, taps, l, L, kinds):
    '''Scaling test. The same 17 tap values run as level l2 describe the same flow stretched by 2^(l2 - l)
    (Euler has no length scale), which should change by 2^(l - l2) times as much in one step.
    taps [Q, 17, 4] from level l, kinds: list of bool masks [Q] (cell types)
    -> num, den [L, len(kinds), 4]: sums of expected * actual and expected^2 per level run as, type and variable.
    Best-fit slope actual vs expected = num / den (1 = self-similar).'''
    c = taps[:, CENTER]
    d = model.local_update(taps, l)[0] - c                            # change at its own level
    num, den = np.zeros((L, len(kinds), 4)), np.zeros((L, len(kinds), 4))
    for l2 in range(L):
        e = 2.0 ** (l - l2) * d                                         # Euler's expectation
        a = model.local_update(taps, l2)[0] - c
        for q, m in enumerate(kinds):
            num[l2, q] += (e[m] * a[m]).sum(0).cpu().numpy()
            den[l2, q] += (e[m] ** 2).sum(0).cpu().numpy()
    return num, den

def physics_error(d):
    '''per cell: mean of the relative transport-speed error and the relative pressure-force error'''
    return 0.5 * (np.abs(d['speed_rho'] - d['u']) / (d['u'] + d['c'])
                  + np.abs(d['force'] - d['force_true']) / d['force_true'])

def smoothing(d):
    return 0.5 * (d['nu_rho'] + d['nu_P'])

def _cat(ds):
    return {k: np.concatenate([d[k] for d in ds]) for k in ds[0]}

# ------------------------------------------------ run ------------------------------------------------
@torch.no_grad()
def run(model, frames, cfg, ev, full=True):
    '''-> dict of numpy arrays. full=False: only the level dependence (used for the untrained reference)'''
    L = model.cfg['refine']['n_levels']
    groups = head_groups(model)
    K = sum(H for _, H in groups)
    R, g, sc = ev['score_res'], model.gamma, model.cfg['physics']['scale']
    rng = np.random.default_rng(cfg['seed'])
    cells, direct = [], []
    pf = {k: [] for k in ['level_dependence', 'selectivity', 'importance', 'physics', 'smoothing']}
    pat = {'shock': [], 'smooth': []}
    sc_num, sc_den = np.zeros((L, L, 2, 4)), np.zeros((L, L, 2, 4))   # [from level, run as, smooth|shock, var]

    for n, f, X, true_next in frames:
        Xs, Ms = levels_and_masks(model, X)
        ld, ld_n, sel, sel_n, on, off = np.zeros(K), 0, np.zeros(K), 0, [], [[] for _ in range(K)]
        for l in range(L):
            taps, _ = gather_stencil(Xs[l], Ms[l])
            if len(taps) == 0:
                continue
            pick = rng.choice(len(taps), min(cfg['n_cells'], len(taps)), replace=False)
            taps = taps[torch.as_tensor(pick, device=taps.device)]

            # H2: attention change when the cell is told another level
            A = run_level(model, taps, l)[1]
            sel += (A - 1 / 17).abs().sum(-1).mean(0).cpu().numpy()     # distance from equal weights (0 = flat)
            sel_n += 1
            for l2 in range(L):
                if l2 != l:
                    ld += (run_level(model, taps, l, swap_to=l2)[1] - A).abs().sum(-1).mean(0).cpu().numpy()
                    ld_n += 1
            if not full:
                continue

            # H2 scaling test: this level's cells, unchanged, run as every level
            sm = (tap_grad(taps, 0) < cfg['smooth_max']) & (tap_grad(taps, 3) < cfg['smooth_max'])
            n_, d_ = scaling_sums(model, taps, l, L, [sm, tap_grad(taps, 3) > ev['shock']['threshold']])
            sc_num[l] += n_
            sc_den[l] += d_

            # H1 / H3: effective stencil of smooth cells, with all heads and with each head off
            sm = (tap_grad(taps, 0) < cfg['smooth_max']) & (tap_grad(taps, 3) < cfg['smooth_max'])
            if not sm.any():
                continue
            ts = taps[sm]
            on.append({**stencil_numbers(effective_stencil(model, ts, l), ts, l, model), 'level': np.full(len(ts), l)})
            for k in range(K):
                with head_off(model, k):
                    off[k].append(stencil_numbers(effective_stencil(model, ts, l), ts, l, model))
        pf['level_dependence'].append(ld / max(ld_n, 1))
        pf['selectivity'].append(sel / max(sel_n, 1))
        if not full:
            continue

        # H1 direct check (no gradients): the model's own new value of each query cell minus its old value,
        # vs the true change (next frame averaged to that level, as the per-level training targets)
        _, out_l = model(X, return_levels=True)
        Xt = torch.from_numpy(box_average(true_next, X.shape[-1], g).astype(np.float32))[None].to(X.device)
        Ts = build_levels(model.to_model_vars(Xt), L, g)
        for l, (x_new, (b, i, j)) in enumerate(out_l['levels']):
            if len(b):
                x0 = Xs[l][b, :, i, j]
                direct.append(np.c_[np.full(len(b), l), (x_new - x0).cpu().numpy(), (Ts[l][b, :, i, j] - x0).cpu().numpy()])

        if on:
            d = _cat(on)
            cells.append(d)
            e_on, s_on = physics_error(d).mean(), smoothing(d)
            # per-cell scale: smoothing of a first-order upwind scheme at the fastest wave speed, (|u| + c) dx_l / 4
            s_ref = (d['u'] + d['c']) * 2.0 ** d['level'] * model.Encoder.dx / 4
            pf['physics'].append(np.array([physics_error(_cat(off[k])).mean() - e_on for k in range(K)]))
            pf['smoothing'].append(np.array([((s_on - smoothing(_cat(off[k]))) / s_ref).mean() for k in range(K)]))
        else:
            pf['physics'].append(np.full(K, np.nan))
            pf['smoothing'].append(np.full(K, np.nan))

        # importance: error increase vs the truth when the head is off, as a multiple of the normal error
        mv_true = model_vars(box_average(true_next, R, g), sc, g)
        err = lambda: np.abs(model_vars(box_average((X + model(X))[0].double().cpu().numpy(), R, g), sc, g)
                             - mv_true).mean()
        e0, imp = err(), np.zeros(K)
        for k in range(K):
            with head_off(model, k):
                imp[k] = err() / e0 - 1
        pf['importance'].append(imp)

        # H2 patterns: same spot at every level (shock and smooth cells picked on the top level)
        for t, (per_level, vel) in targeted_queries(Xs, cfg['n_per_type'], ev['shock']['threshold'],
                                                    cfg['smooth_max'], rng).items():
            if t in pat:
                perm = canonical(vel)
                pat[t].append(torch.stack([to_canonical(run_level(model, tl, l)[1], perm)
                                           for l, tl in enumerate(per_level)], 1).cpu().numpy())   # [n, L, K, 17]

    r = {'head_names': np.array([f'{nm} h{h}' for nm, H in groups for h in range(H)]),
         'head_group': np.array([q for q, (_, H) in enumerate(groups) for _ in range(H)]),
         'group_names': np.array([nm for nm, _ in groups]),
         'cell_size': np.array([2 ** l * model.Encoder.dx for l in range(L)])}
    r.update({f'pf_{k}': np.array(v) for k, v in pf.items() if v})
    if full:
        r.update({f'cell_{k}': v for k, v in (_cat(cells) if cells else {}).items()})
        with np.errstate(invalid='ignore', divide='ignore'):
            r['scaling_slope'] = sc_num / sc_den                       # [from, run as, smooth|shock, var]
        r['scaling_den'] = sc_den
        r['direct'] = np.concatenate(direct) if direct else np.zeros((0, 9))   # [cells, level | pred 4 | true 4]
        for t, v in pat.items():
            r[f'pat_{t}'] = np.concatenate(v) if v else np.zeros((0, L, K, 17))
    return r

# ------------------------------------------------ figures ------------------------------------------------
def boot(v, n_boot, ci, rng):
    '''per-frame values [F, K] -> mean, lo, hi [K]: 90% interval from resampling frames'''
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        b = np.nanmean(v[rng.integers(0, len(v), (n_boot, len(v)))], 1)
        a = (1 - ci) / 2
        return np.nanmean(v, 0), np.nanquantile(b, a, 0), np.nanquantile(b, 1 - a, 0)

def _save(fig, out, name, caption):
    fig.text(0.5, -0.02, caption, ha='center', va='top', fontsize=8, wrap=True)
    fig.savefig(os.path.join(out, name), dpi=130, bbox_inches='tight')
    plt.close(fig)

def _hist(ax, x, y, title, xlabel, ylabel, level=None, ref='physics: learned = true'):
    '''2D histogram of cells with the y = x line, plus the best-fit slope through 0 (1 = matches).
    level [cells]: also draw one best-fit line per level'''
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    level = None if level is None else level[ok]
    lim = [min(0, np.percentile(np.r_[x, y], 1)), np.percentile(np.r_[x, y], 99)]
    ax.hist2d(x, y, bins=60, range=[lim, lim], cmap='Greys', norm=LogNorm())
    ax.plot(lim, lim, color='C3', lw=1.5, label=ref)
    ax.axhline(0, color='grey', lw=0.8, ls=':')
    slope = (x * y).sum() / (x * x).sum()
    ax.text(0.03, 0.95, f'best-fit slope, all cells {slope:.2f}\n(1 = matches, 0 = none)', transform=ax.transAxes,
            va='top', fontsize=8, bbox=dict(fc='white', ec='none', alpha=0.8))
    if level is not None:
        for l in np.unique(level):
            xl, yl = x[level == l], y[level == l]
            sl = (xl * yl).sum() / (xl * xl).sum()
            ax.plot(lim, [sl * lim[0], sl * lim[1]], color=f'C{int(l)}', lw=1.2, ls='--',
                    label=f'L{int(l)}: slope {sl:.2f} ({len(xl)} cells)')
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.legend(fontsize=7, loc='lower right')

def _dots(ax, mean, lo, hi, r, order):
    '''one dot per head with its 90% interval, coloured by head group'''
    grp, names = r['head_group'][order], r['head_names'][order]
    x = np.arange(len(order))
    for g, gname in enumerate(r['group_names']):
        k = grp == g
        ax.errorbar(x[k], mean[order][k], yerr=[mean[order][k] - lo[order][k], hi[order][k] - mean[order][k]],
                    fmt='o', color=f'C{g}', capsize=2, label=gname)
    ax.set_xticks(x, names, rotation=90, fontsize=7)

def make_figures(r, out, cfg, ref=None):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(cfg['seed'])
    names = r['head_names']
    L = len(r['cell_size'])
    c = {k[5:]: r[k] for k in r if k.startswith('cell_')}

    # ---- H1 ----
    if c:
        fig, axs = plt.subplots(1, 2, figsize=(9, 4.2))
        for ax, v, lab in [(axs[0], 'speed_rho', 'density'), (axs[1], 'speed_P', 'pressure')]:
            _hist(ax, c['u'], c[v], f'{lab} is carried at speed ...', 'true flow speed |u|',
                  f'learned speed {lab} is carried at', c['level'])
        fig.suptitle('H1: does the model move things at the real flow speed?', fontsize=11)
        _save(fig, out, 'h1_transport.png',
              'Each grey square counts smooth cells (darker = more). Cells on the red line move density / pressure '
              'exactly with the flow, as the Euler equations do.\nA cloud on the line points to physics; a flat '
              'cloud or wide scatter means the model is not tracking the flow speed. Dashed lines: fit per level.')

        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        _hist(ax, c['force_true'], c['force'], '', "Euler value (c'^2 = P'/rho')", 'learned push', c['level'])
        fig.suptitle('H1: does a pressure difference push the flow\nas hard as in the Euler equations?', fontsize=10)
        _save(fig, out, 'h1_pressure_force.png',
              'How strongly a pressure difference across a cell changes its velocity in one step, '
              'learned vs the Euler equations.\nOn the red line = the model uses the real pressure force. '
              'Dashed lines: fit per level.')

        fig, ax = plt.subplots(figsize=(5, 4))
        nu = smoothing(c)
        med = np.array([np.median(nu[c['level'] == l]) if (c['level'] == l).any() else np.nan for l in range(L)])
        q = np.array([np.percentile(nu[c['level'] == l], [25, 75]) if (c['level'] == l).any() else [np.nan] * 2
                      for l in range(L)])
        s = r['cell_size']
        ax.errorbar(s, med, yerr=[med - q[:, 0], q[:, 1] - med], fmt='o-', color='k', capsize=3, label='model')
        l0 = np.flatnonzero(np.isfinite(med))[0] if np.isfinite(med).any() else 0
        ax.plot(s, med[l0] * s / s[l0], color='C3', ls='--', label='grows with cell size (numerics)')
        ax.plot(s, np.full(L, med[l0]), color='C0', ls='--', label='same at every cell size')
        ax.axhline(0, color='grey', lw=0.8, ls=':')
        ax.set_xscale('log', base=2)
        ax.set_xticks(s, [f'L{l}\n({2 ** l} dx)' for l in range(L)])
        ax.set_xlabel('cell size')
        ax.set_ylabel('smoothing strength (median, 25-75%)')
        ax.legend(fontsize=7)
        fig.suptitle('H1: does the smoothing grow with cell size,\nas a numerical scheme\'s does?', fontsize=10)
        _save(fig, out, 'h1_smoothing.png',
              'Euler data has no physical smoothing, so any smoothing the model adds is numerics. '
              'A scheme\'s smoothing grows with cell size (red line).\nBelow 0 = the model sharpens.')

    if len(r.get('direct', [])):
        D = r['direct']
        fig, axs = plt.subplots(1, 4, figsize=(16, 4.2))
        for a, (ax, lab) in enumerate(zip(axs, ['log rho', "u'", "v'", 'log P'])):
            _hist(ax, D[:, 5 + a], D[:, 1 + a], lab, f'true change of {lab}', f'predicted change of {lab}',
                  D[:, 0], ref='predicted = true')
        fig.suptitle('H1 check without gradients: does each level change its cells by the right amount in one step?',
                     fontsize=11)
        _save(fig, out, 'h1_direct.png',
              "The model's own new value of each query cell minus its old value, against the true change (next frame "
              'averaged to that level). All query cells, not only smooth ones.\nDashed lines: fit per level. A slope '
              'below 1 at a level = that level moves too little per step.')

    # ---- H2 ----
    if 'scaling_slope' in r:
        fig, axs = plt.subplots(1, 2, figsize=(9, 4))
        for q, (ax, t) in enumerate(zip(axs, ['smooth cells', 'shock cells'])):
            with warnings.catch_warnings():                             # levels with no cells of this type: NaN
                warnings.simplefilter('ignore', RuntimeWarning)
                mm = np.nanmean(r['scaling_slope'][:, :, q], -1)        # mean over the 4 variables
            mm[(r['scaling_den'][:, :, q] == 0).all(-1)] = np.nan
            ax.imshow(mm, cmap='RdBu_r', vmin=0, vmax=2)
            for i in range(L):
                for j in range(L):
                    ax.text(j, i, 'no cells' if not np.isfinite(mm[i, j]) else f'{mm[i, j]:.2f}', ha='center',
                            va='center', fontsize=9)
            ax.set_xticks(range(L), [f'run as L{l}' for l in range(L)])
            ax.set_yticks(range(L), [f'cells from L{l}' for l in range(L)])
            ax.set_title(t, fontsize=9)
        fig.suptitle('H2: run a cell\'s values as a coarser or finer cell. Does its change scale the way Euler says?',
                     fontsize=10)
        _save(fig, out, 'h2_scaling.png',
              'The same 17 tap values run as a level k times coarser describe the same flow stretched k times, which '
              'should change 1/k as much in one step (Euler has no length scale).\n1 = exactly self-similar, < 1 = it '
              'moves too little when run coarser (or too much when run finer), > 1 = the reverse. Diagonal = 1 by '
              'definition. Mean over log rho, u, v, log P.')
        with open(os.path.join(out, 'h2_scaling.md'), 'w') as fh:
            lines = ['# H2 scaling test: slope per variable (1 = self-similar)', '',
                     '| cells | from | run as | log rho | u | v | log P |', '|---|---|---|---|---|---|---|']
            for q, t in enumerate(['smooth', 'shock']):
                for i in range(L):
                    for j in range(L):
                        if i != j:
                            lines.append(f'| {t} | L{i} | L{j} | ' + ' | '.join(f'{v:.2f}' for v in
                                                                              r['scaling_slope'][i, j, q]) + ' |')
            fh.write('\n'.join(lines) + '\n')

    m, lo, hi = boot(r['pf_level_dependence'], cfg['n_boot'], cfg['ci'], rng)
    order = np.lexsort((m, r['head_group']))
    fig, ax = plt.subplots(figsize=(1.5 + 0.32 * len(names), 3.6))
    if ref is not None:
        band = np.nanpercentile(np.nanmean(ref['pf_level_dependence'], 0), [5, 95])
        ax.axhspan(*band, color='grey', alpha=0.25, label='untrained heads (5-95%)')
    _dots(ax, m, lo, hi, r, order)
    ax.set_ylim(bottom=0)
    ax.set_ylabel('attention change when told another level\n(0 = none, 2 = completely different)', fontsize=8)
    ax.legend(fontsize=7)
    fig.suptitle('H2: does each head look at the same taps whatever level it is told it is on?', fontsize=10)
    _save(fig, out, 'h2_level_dependence.png',
          'Lower = more self-similar. Dots below the grey band: training made the head less level-dependent than '
          'random weights are.\nBars = 90% interval over test frames.')

    sl, ld = np.nanmean(r['pf_selectivity'], 0), m
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    if ref is not None:
        ax.axhspan(*np.nanpercentile(np.nanmean(ref['pf_level_dependence'], 0), [5, 95]), color='grey', alpha=0.2,
                   label='untrained heads (5-95%)')
        ax.axvspan(*np.nanpercentile(np.nanmean(ref['pf_selectivity'], 0), [5, 95]), color='grey', alpha=0.2)
    for g, gname in enumerate(r['group_names']):
        k = r['head_group'] == g
        ax.scatter(sl[k], ld[k], color=f'C{g}', label=gname, ec='k', lw=0.4, s=40)
    for k in np.argsort(-sl)[:6]:                                       # label only the most selective heads
        ax.annotate(names[k], (sl[k], ld[k]), fontsize=7, xytext=(4, 3), textcoords='offset points')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('selectivity: how far from equal weight on all 17 taps\n(0 = flat, about 1.9 = all weight on one tap)',
                  fontsize=8)
    ax.set_ylabel('attention change when told another level', fontsize=8)
    ax.legend(fontsize=7)
    fig.suptitle('H2: which heads pick specific taps, and do so the same way at every level?', fontsize=10)
    _save(fig, out, 'h2_selectivity.png',
          'Bottom right = picks specific taps and keeps the same choice at every level (real self-similarity). '
          'Bottom left = barely picks, so self-similar by default.\nTop = changes with level. '
          'Grey bands: untrained heads. Only the 6 most selective heads are labelled.')

    if 'pf_importance' in r and len(r['pat_shock']) + len(r['pat_smooth']):
        heads = [list(names).index(h) for h in (cfg.get('example_heads') or []) if h in names]
        if not heads:   # most important head, and the most self-similar of the more important half
            im, ld = np.nanmean(r['pf_importance'], 0), np.nanmean(r['pf_level_dependence'], 0)
            sl = np.nanmean(r['pf_selectivity'], 0)
            top = np.flatnonzero((im >= np.nanmedian(im)) & (sl >= np.nanmedian(sl)))
            top = top if len(top) else np.flatnonzero(im >= np.nanmedian(im))
            heads = list(dict.fromkeys([int(np.nanargmax(im)), int(top[np.nanargmin(ld[top])])]))
        types = [t for t in ['shock', 'smooth'] if len(r[f'pat_{t}'])]
        rows = [(k, t) for k in heads for t in types]
        P = {(k, t): r[f'pat_{t}'][:, :, k].mean(0) for k, t in rows}            # [L, 17]
        vmax = max(p.max() for p in P.values())
        fig, axs = plt.subplots(len(rows), L, figsize=(2.4 * L, 2.5 * len(rows)), squeeze=False)
        o = OFFSETS.numpy()
        for i, (k, t) in enumerate(rows):
            for j, l in enumerate(range(L - 1, -1, -1)):
                ax = axs[i, j]
                sc_ = ax.scatter(o[:, 0], o[:, 1], c=P[k, t][l], s=260, cmap='viridis', vmin=0, vmax=vmax, ec='k', lw=0.4)
                ax.annotate('', xy=(2.6, -2.6), xytext=(1.2, -2.6), arrowprops=dict(arrowstyle='->', lw=1))
                ax.text(1.9, -2.95, 'flow', ha='center', fontsize=6)
                ax.set_xlim(-2.8, 2.8)
                ax.set_ylim(-3.1, 2.8)
                ax.set_aspect('equal')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f'L{l} ({2 ** l} dx)' if i == 0 else '', fontsize=9)
            ld_k = np.nanmean(r['pf_level_dependence'], 0)[k]
            sl_k = np.nanmean(r['pf_selectivity'], 0)[k]
            axs[i, 0].set_ylabel(f'{names[k]}\n{t} cells\n(level dep. {ld_k:.2f}, selectivity {sl_k:.2f})', fontsize=7)
        fig.colorbar(sc_, ax=axs.ravel().tolist(), shrink=0.8, label='attention weight')
        fig.suptitle('H2: the same spot on every level', fontsize=10)
        _save(fig, out, 'h2_patterns.png',
              'Each dot is one of the 17 taps, the centre dot is the cell itself. Same colours along a row = '
              'the head does the same thing at every level.\nLevel dep. and selectivity = the head\'s values in h2_selectivity.png.')

    # ---- H3 ----
    if 'pf_physics' in r:
        S = {k: boot(r[f'pf_{k}'], cfg['n_boot'], cfg['ci'], rng) for k in ['importance', 'physics', 'smoothing']}
        (im, _, _), (px, plo, phi), (sy, slo, shi) = S['importance'], S['physics'], S['smoothing']
        me = cfg['min_effect']
        roles = []
        for k in range(len(names)):
            role = []
            if plo[k] > 0 and px[k] >= me['physics']:
                role.append('physics')
            if slo[k] > 0 and sy[k] >= me['smoothing']:
                role.append('smoothing')
            if shi[k] < 0 and -sy[k] >= me['smoothing']:
                role.append('sharpening')
            roles.append(' + '.join(role) or 'unclear')
        order = np.argsort(-np.nan_to_num(im, nan=-np.inf))

        # scatter, left: all heads; right: zoom on the crowded middle. Short labels (R0.1 = router 0 head 1,
        # S11 = stencil head 11) only on heads with a role, so the middle stays readable
        short = [n.replace('router ', 'R').replace('layer ', 'R').replace(' h', '.') if not n.startswith('stencil')
                 else 'S' + n.split('h')[-1] for n in names]
        size = 15 + 300 * np.clip(im, 0, None) / max(np.nanmax(im), 1e-12)
        far = np.zeros(len(names), bool)
        for v in (px, sy):                                              # the heads that set the full view's range
            a = np.abs(v - np.nanmedian(v))
            far |= a > 4 * max(np.nanpercentile(a, 75), 1e-12)
        fig, axs = plt.subplots(1, 2, figsize=(12, 5.5))
        for j, ax in enumerate(axs):
            ax.axhline(0, color='grey', lw=0.8)
            ax.axvline(0, color='grey', lw=0.8)
            for g, gname in enumerate(r['group_names']):
                k = r['head_group'] == g
                ax.errorbar(px[k], sy[k], xerr=[px[k] - plo[k], phi[k] - px[k]],
                            yerr=[sy[k] - slo[k], shi[k] - sy[k]], fmt='none', ecolor=f'C{g}', alpha=0.35)
                ax.scatter(px[k], sy[k], s=size[k], color=f'C{g}', label=gname, ec='k', lw=0.4, zorder=3)
            show = far if j == 0 else ~far
            for k in np.flatnonzero(show & (np.array(roles) != 'unclear')):
                ax.annotate(short[k], (px[k], sy[k]), fontsize=7, xytext=(5, 4), textcoords='offset points')
            ax.set_xlabel('physics it provides\n(how much worse transport speed + pressure force get without it)',
                          fontsize=8)
        if (~far).any():
            pad = lambda a: 0.15 * (a.max() - a.min()) + 1e-6
            x, y = px[~far], sy[~far]
            axs[1].set_xlim(x.min() - pad(x), x.max() + pad(x))
            axs[1].set_ylim(y.min() - pad(y), y.max() + pad(y))
            (x0, x1), (y0, y1) = axs[1].get_xlim(), axs[1].get_ylim()
            axs[0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec='k', ls='--', lw=0.8))
        axs[0].set_title('all heads', fontsize=9)
        axs[1].set_title('zoom on the middle', fontsize=9)
        axs[0].set_ylabel('smoothing it provides\n(upwind-scheme units; < 0 = it sharpens)', fontsize=8)
        axs[0].legend(fontsize=7)
        fig.suptitle('H3: what does each head do?', fontsize=11)
        _save(fig, out, 'h3_roles.png',
              'Right = physics head, up = smoothing (numerics) head, down = sharpening head, middle = neither in '
              'smooth flow. Dot size = how much the error grows without the head.\nLabels (R0.1 = router 0 head 1, '
              'S11 = stencil head 11) only on heads with a role in h3_roles.md. Lines = 90% interval over test frames.')

        lines = ['# H3: head roles', '',
                 'physics: transport speed + pressure force get worse without the head (interval above 0 and at least '
                 f'{me["physics"]}). smoothing / sharpening: without the head the model loses / gains at least '
                 f'{me["smoothing"]} of a first-order upwind scheme\'s smoothing (interval excludes 0). '
                 'Measured in smooth cells. '
                 'importance: error growth without the head, as a multiple of the normal error.', '',
                 '| head | importance | physics it provides | smoothing it provides | role |', '|---|---|---|---|---|']
        for k in order:
            lines.append(f'| {names[k]} | {im[k]:.2f} | {px[k]:.3f} [{plo[k]:.3f}, {phi[k]:.3f}] | '
                         f'{sy[k]:.2f} [{slo[k]:.2f}, {shi[k]:.2f}] | {roles[k]} |')
        with open(os.path.join(out, 'h3_roles.md'), 'w') as fh:
            fh.write('\n'.join(lines) + '\n')

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
    Nx = model_cfg['physics']['domain']['Nx']
    model = load_model(model_cfg, device)
    r = run(model, iter_frames(cfg, data_cfg, Nx, device), cfg, ev)
    out = os.path.join(cfg['out_dir'], model_cfg['name'])
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, 'hypotheses.npz'), **r)

    ref = None
    if cfg['untrained_reference']:   # same architecture, fresh random weights: "learned nothing"
        torch.manual_seed(cfg['seed'])
        ref = run(type(model)(model_cfg).to(device).eval(), iter_frames(cfg, data_cfg, Nx, device), cfg, ev, full=False)
    make_figures(r, out, cfg, ref)
    print('saved to', out)