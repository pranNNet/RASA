'''
evaluate.py

Roll every model in the eval config out from frame 0 of each sample in the split and
score each step. Writes one <label>.npz per model to out_dir. T = steps, index k = after step k+1.
  l1            [S, T, 4]  mean |pred - true| in model vars (log rho, u, v, log P)
  l1_shock_band [S, T, 4]  same, only over the band around true shock cells
  shock_overlap [S, T]     IoU of predicted and true shock cells
  totals_err    [S, T, 4]  |total pred - total true| / reference (mass, x mom, y mom, energy)
  blowup_step   [S]        first step with NaN/inf or rho, P <= 0 (-1 if none)
  queries       [S, T, L]  query cells per level (sums of out['masks'])
  step_time     [T]        seconds per model step, mean over samples
Scoring stops at blowup (later values stay NaN).

Run from rasa/:  python -m eval.evaluate --config configs/eval_config.yaml
'''
import os
import time

import numpy as np

from preprocess.data import load_config, load_window, sample_path, split_indices, to_primitive

def to_conserved(W, gamma):
    '''raw (rho, u, v, P) [..., 4, n, n] -> (rho, rho u, rho v, E)'''
    rho, u, v, P = (W[..., c, :, :] for c in range(4))
    return np.stack([rho, rho * u, rho * v, P / (gamma - 1) + 0.5 * rho * (u**2 + v**2)], axis=-3)

def box_average(W, n, gamma):
    '''raw primitives [..., 4, N, N] -> [..., 4, n, n]: average the conserved values over c x c boxes, c = N/n'''
    N = W.shape[-1]
    if N == n:
        return W
    assert N % n == 0, f'{N} is not a multiple of {n}'
    c = N // n
    U = to_conserved(W, gamma)
    U = U.reshape(U.shape[:-2] + (n, c, n, c)).mean(axis=(-3, -1))
    return to_primitive(U, gamma)

def model_vars(W, sc, gamma):
    '''raw primitives -> (log rho/rho_inf, u/c_inf, v/c_inf, log P/P_inf), as RASA.to_model_vars'''
    c_inf = (gamma * sc['P_inf'] / sc['rho_inf']) ** 0.5
    return np.stack([np.log(W[..., 0, :, :] / sc['rho_inf']), W[..., 1, :, :] / c_inf,
                     W[..., 2, :, :] / c_inf, np.log(W[..., 3, :, :] / sc['P_inf'])], axis=-3)

def dilate(M, w):
    '''bool [n, n] -> True within w cells (8 neighbours) of any True cell'''
    P = np.pad(M, w)
    n, m = M.shape
    return np.any([P[i:i + n, j:j + m] for i in range(2 * w + 1) for j in range(2 * w + 1)], axis=0)

def shock_cells(W, threshold):
    '''raw primitives [4, n, n] -> bool [n, n]: |grad log P| per cell (central differences) > threshold'''
    gx, gy = np.gradient(np.log(W[3]))
    return np.hypot(gx, gy) > threshold

def reference_totals(W0, gamma, sc):
    '''true frame 0 -> scale per conserved variable: mean mass, mean mass * c_inf (twice), mean energy'''
    c_inf = (gamma * sc['P_inf'] / sc['rho_inf']) ** 0.5
    U = to_conserved(W0, gamma)
    m, E = U[0].mean(), U[3].mean()
    return np.array([m, m * c_inf, m * c_inf, E])

def score_step(pred, true, ref, gamma, sc, threshold, band):
    '''
    pred, true [4, n, n] raw primitives at score_res, ref [4] from reference_totals
    -> l1 [4], l1_band [4], overlap (float), totals_err [4]
    '''
    e = np.abs(model_vars(pred, sc, gamma) - model_vars(true, sc, gamma))   # [4, n, n]
    l1 = e.mean(axis=(1, 2))

    Sp, St = shock_cells(pred, threshold), shock_cells(true, threshold)
    union = (Sp | St).sum()
    overlap = (Sp & St).sum() / union if union else np.nan
    B = dilate(St, band)
    l1_band = e[:, B].mean(axis=1) if B.any() else np.full(4, np.nan)

    tot = lambda W: to_conserved(W, gamma).mean(axis=(1, 2))
    totals_err = np.abs(tot(pred) - tot(true)) / ref
    return l1, l1_band, overlap, totals_err

def evaluate_model(model, model_cfg, data_cfg, ev, device='cpu'):
    '''Rolls out every sample of ev['split'] and returns the dict of arrays described at the top.'''
    import torch
    gamma, sc = data_cfg['gamma'], model_cfg['physics']['scale']
    Nx, n_lev = model_cfg['physics']['domain']['Nx'], model_cfg['refine']['n_levels']
    T, R = ev['steps'], ev['score_res']
    thr, band = ev['shock']['threshold'], ev['shock']['band']
    idx = split_indices(data_cfg)[ev['split']]
    S = len(idx)

    r = {'l1': np.full((S, T, 4), np.nan), 'l1_shock_band': np.full((S, T, 4), np.nan),
         'shock_overlap': np.full((S, T), np.nan), 'totals_err': np.full((S, T, 4), np.nan),
         'blowup_step': np.full(S, -1), 'queries': np.full((S, T, n_lev), np.nan)}
    times = np.full((S, T), np.nan)

    for s, i in enumerate(idx):
        raw = load_window(sample_path(data_cfg, i), 0, T, data_cfg['step_stride'], gamma)   # [<=T+1, 4, N, N]
        true = box_average(raw.astype(np.float64), R, gamma)
        ref = reference_totals(true[0], gamma, sc)
        X = torch.from_numpy(box_average(raw[0], Nx, gamma).astype(np.float32))[None].to(device)

        for k in range(min(T, len(true) - 1)):   # stop where the truth ends
            t0 = time.perf_counter()
            with torch.no_grad():
                dX, out = model(X, return_levels=True)
            if device != 'cpu':
                torch.cuda.synchronize()
            times[s, k] = time.perf_counter() - t0
            X = X + dX
            r['queries'][s, k] = [m.sum().item() for m in out['masks']]

            p = X[0].double().cpu().numpy()
            if not np.isfinite(p).all() or (p[[0, 3]] <= 0).any():
                r['blowup_step'][s] = k + 1
                break
            r['l1'][s, k], r['l1_shock_band'][s, k], r['shock_overlap'][s, k], r['totals_err'][s, k] = \
                score_step(box_average(p, R, gamma), true[k + 1], ref, gamma, sc, thr, band)

    done = np.isfinite(times).sum(axis=0)   # samples that reached each step
    r['step_time'] = np.where(done > 0, np.nansum(times, axis=0) / np.maximum(done, 1), np.nan)
    return r

def load_model(model_cfg, device='cpu'):
    '''Class from model_cfg['name'], weights from <model_save_dir>/model.pt'''
    import torch
    from model.rasa import RASA
    from model.rpa.rpa import RPA
    Model = {'RASA': RASA, 'RPA': RPA}.get(model_cfg['name'], RASA)   # e.g. RASA_uniform128 -> RASA
    model = Model(model_cfg).to(device)
    path = os.path.join(model_cfg['model_save_dir'], 'model.pt')
    model.load_state_dict(torch.load(path, map_location=device))
    return model.eval()

if __name__ == '__main__':
    import argparse
    import torch
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='evaluation/eval_config.yaml')
    args = p.parse_args()

    ev = load_config(args.config)
    data_cfg = load_config(ev['data'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(ev['out_dir'], exist_ok=True)
    for m in ev['models']:
        model_cfg = load_config(m['config'])
        r = evaluate_model(load_model(model_cfg, device), model_cfg, data_cfg, ev, device)
        out = os.path.join(ev['out_dir'], f"{m['label']}.npz")
        np.savez(out, label=m['label'], **r)
        print('saved', out, '| blowups:', int((r['blowup_step'] > 0).sum()))