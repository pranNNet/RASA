'''
animate_rollout.py

Roll a trained RASA or RPA checkpoint out from frame 0 of one test sample and save
an animation: predicted log rho | true log rho | |error| | level map (0 = L0, 1 = L1, 2 = L2).

Run from rasa/:
python -m evaluation.animate_rollout --model ./experiments/rasa_main/config.yaml --ckpt checkpoints/v1_base/stage3_best.pt --sample 0 --steps 121 --out rollout.mp4
--sample is the index inside the test split (0 = first test sample). Use .mp4 for --out if ffmpeg is installed.
'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

def rollout(model, X0, steps):
    '''
    X0 [4, N, N] raw (rho, u, v, P) numpy
    ->
    pred [steps+1, 4, N, N] raw, levels [steps+1, N, N] int (level used at each fine cell)
    Each step: X <- X + model(X). The level map of frame k is the one the model used to step from k.
    '''
    import torch
    X = torch.from_numpy(X0)[None]
    pred, levels = [X0], []
    with torch.no_grad():
        for _ in range(steps):
            dX, out = model(X, return_levels=True)
            levels.append(level_map(out['masks'], X.shape[-1]))
            X = X + dX
            pred.append(X[0].numpy())
        levels.append(level_map(model(X, return_levels=True)[1]['masks'], X.shape[-1]))
    return np.stack(pred), np.stack(levels)

def level_map(masks, N):
    '''masks: list of M bool tensors [1, N/2^l, N/2^l] (finest first) -> [N, N] int level per fine cell'''
    L = np.zeros((N, N), dtype=int)
    for l, M in enumerate(masks):
        s = N // M.shape[-1]
        L += l * np.kron(M[0].cpu().numpy(), np.ones((s, s), dtype=bool))
    return L

def animate(pred, true, levels, out, fps=10):
    '''
    pred [T, 4, N, N], true [T_true <= T, 4, N, N] raw (rho, u, v, P); levels [T, N, N] int or None
    Saves the animation to out (.gif or .mp4). Colour limits are fixed over all frames.
    Frames past the end of true show a blank true panel and no error.
    '''
    T_true = len(true)
    true = np.concatenate([true, np.full((len(pred) - T_true,) + true.shape[1:], np.nan, true.dtype)])
    with np.errstate(invalid='ignore', divide='ignore'):
        lp, lt = np.log(pred[:, 0]), np.log(true[:, 0])
    err = np.abs(lp - lt)
    vmin, vmax = np.nanmin(lt), np.nanmax(lt)
    emax = np.nanmax(err) if np.isfinite(err).any() else 1.0

    panels = [(lp, 'predicted log rho', dict(cmap='viridis', vmin=vmin, vmax=vmax)),
              (lt, 'true log rho', dict(cmap='viridis', vmin=vmin, vmax=vmax)),
              (err, '|error| log rho', dict(cmap='magma', vmin=0, vmax=emax))]
    if levels is not None:
        panels.append((levels, 'level (0 = finest)', dict(cmap='Greys_r', vmin=0, vmax=2)))

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.2))
    ims = []
    for ax, (a, title, kw) in zip(axes, panels):
        ims.append(ax.imshow(a[0].T, origin='lower', **kw))   # x along rows (dim 2) -> horizontal axis
        ax.set_title(title)
        ax.set_xticks([]), ax.set_yticks([])
        fig.colorbar(ims[-1], ax=ax, fraction=0.046)
    label = fig.suptitle('step 0')

    def frame(k):
        for im, (a, _, _) in zip(ims, panels):
            im.set_data(a[k].T)
        e = f'mean |error| = {np.nanmean(err[k]):.3g}' if k < T_true else 'past end of data'
        label.set_text(f'step {k}   {e}')
        return ims

    FuncAnimation(fig, frame, frames=len(pred), blit=False).save(out, fps=fps)
    plt.close(fig)

if __name__ == '__main__':
    import argparse
    import torch
    from preprocess.data import load_config, load_window, sample_path, split_indices
    from model.rasa import RASA
    # from model.rpa.rpa import RPA

    p = argparse.ArgumentParser()
    p.add_argument('--model', default='./experiments/rasa_main/config.yaml')
    p.add_argument('--data', default='./train/dataset_config.yaml')
    p.add_argument('--ckpt', required=True)
    p.add_argument('--sample', type=int, default=0, help='index inside the test split')
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--out', default='rollout.gif')
    p.add_argument('--fps', type=int, default=10)
    args = p.parse_args()

    model_cfg, data_cfg = load_config(args.model), load_config(args.data)
    model = (RPA if model_cfg['name'] == 'RPA' else RASA)(model_cfg)
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
    model.eval()

    i = split_indices(data_cfg)['test'][args.sample]
    true = load_window(sample_path(data_cfg, i), 0, args.steps, data_cfg['step_stride'], data_cfg['gamma'])
    if len(true) - 1 < args.steps:   # the file has fewer frames: keep rolling out with no truth after that
        print(f'sample has only {len(true) - 1} true steps; steps after that have no truth to compare')
    pred, levels = rollout(model, true[0], args.steps)
    animate(pred, true, levels, args.out, args.fps)
    print('saved', args.out)