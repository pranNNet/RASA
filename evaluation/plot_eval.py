'''
plot_eval.py

Plot any number of evaluate.py results files: one curve per model (mean over samples) per metric
vs step, and a summary table. Never loads a model.
Means skip NaN, so samples that blew up drop out after their blowup step (see the blowups column).

Run from rasa/:  python -m eval.plot_eval results/*.npz --out plots/
'''
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

VARS = ['log rho', 'u', 'v', 'log P']
TOTALS = ['mass', 'x momentum', 'y momentum', 'energy']
# metric -> (subplot titles or None for one panel, y label)
PLOTS = {'l1': (VARS, 'L1 error'),
         'l1_shock_band': (VARS, 'L1 error in shock band'),
         'totals_err': (TOTALS, 'relative total error'),
         'shock_overlap': (None, 'shock IoU'),
         'queries': (None, 'query cells'),
         'step_time': (None, 'seconds per step')}

def curve(r, key):
    '''results dict, metric -> mean over samples [T] or [T, C]'''
    a = r[key]
    if key == 'queries':
        a = a.sum(axis=-1)          # [S, T]
    return a if key == 'step_time' else np.nanmean(a, axis=0)

def plot_queries(results, out):
    '''One colour per model, one line style per level (L0 solid, L1 dashed, ...). Any level count.'''
    styles = ['-', '--', ':', '-.']
    fig, ax = plt.subplots(figsize=(5, 3.5))
    n_lev = 0
    for n, (label, r) in enumerate(results.items()):
        q = np.nanmean(r['queries'], axis=0)       # [T, levels]
        steps = np.arange(1, len(q) + 1)
        n_lev = max(n_lev, q.shape[1])
        for l in range(q.shape[1]):
            ax.plot(steps, q[:, l], color=f'C{n}', ls=styles[l % len(styles)])
    ax.set_xlabel('step')
    ax.set_ylabel('query cells')
    # legend: colour = model, line style = level
    models = [Line2D([], [], color=f'C{n}', label=label) for n, label in enumerate(results)]
    levels = [Line2D([], [], color='grey', ls=styles[l % len(styles)], label=f'L{l}') for l in range(n_lev)]
    ax.legend(handles=models + levels)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'queries.png'), dpi=120)
    plt.close(fig)

def plot_metric(results, key, out):
    if key == 'queries':
        return plot_queries(results, out)
    titles, ylabel = PLOTS[key]
    n = len(titles) if titles else 1
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5), squeeze=False)
    for label, r in results.items():
        c = curve(r, key)
        steps = np.arange(1, len(c) + 1)
        for k, ax in enumerate(axes[0]):
            ax.plot(steps, c[:, k] if titles else c, label=label)
    for k, ax in enumerate(axes[0]):
        ax.set_xlabel('step')
        ax.set_ylabel(ylabel)
        if titles:
            ax.set_title(titles[k])
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, f'{key}.png'), dpi=120)
    plt.close(fig)

def summary_table(results, steps):
    '''Markdown table: mean L1 (all vars), shock IoU, energy error at the given steps, plus blowups,
    mean queries and mean step time.'''
    head = ['model'] + [f'{m} @{s}' for m in ('L1', 'IoU', 'E err') for s in steps] + ['blowups', 'queries', 's/step']
    rows = []
    for label, r in results.items():
        l1 = curve(r, 'l1').mean(axis=1)
        iou, e = curve(r, 'shock_overlap'), curve(r, 'totals_err')[:, 3]
        at = lambda a: [f'{a[s - 1]:.3g}' if s <= len(a) else '-' for s in steps]
        b = r['blowup_step']
        rows.append([label] + at(l1) + at(iou) + at(e) +
                    [f'{(b > 0).sum()}/{len(b)}', f'{np.nanmean(curve(r, "queries")):.0f}',
                     f'{np.nanmean(r["step_time"]):.3g}'])
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    return '\n'.join(lines + ['| ' + ' | '.join(row) + ' |' for row in rows])

def main(files, out, steps=(1, 10, 30, 60, 120)):
    os.makedirs(out, exist_ok=True)
    results = {}
    for f in files:
        with np.load(f) as d:
            results[str(d['label'])] = {k: d[k] for k in d.files}
    with np.errstate(invalid='ignore'), __import__('warnings').catch_warnings():
        __import__('warnings').simplefilter('ignore', RuntimeWarning)   # all-NaN steps after blowups
        for key in PLOTS:
            plot_metric(results, key, out)
        table = summary_table(results, steps)
    with open(os.path.join(out, 'summary.md'), 'w') as f:
        f.write(table + '\n')
    print(table)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('files', nargs='+')
    p.add_argument('--out', default='results/plots/')
    p.add_argument('--steps', type=int, nargs='+', default=[1, 10, 30, 60, 120])
    args = p.parse_args()
    main(args.files, args.out, args.steps)