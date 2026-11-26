'''
trainloop

we trainin' yo
Run from rasa/:  python -m train.train configs/training.yaml
'''
import os
import sys
import time

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from preprocess.data import load_config, load_window, sample_path, split_indices
from model.rasa import RASA
from model.rpa.rpa import RPA
# from model.rpac.rpac import RPAC
# from model.rfa.rfa import RFA
from model.bt_rasa.bt_rasa import BTRASA
from train.dataset import RASADataset
from train.loss import per_level_loss

STAGES = ['stage1', 'stage2', 'stage3'] #should be an input parameter to the training
MODELS = {'RASA': RASA, 'RPA': RPA, 'BT-RASA': BTRASA} #'RPAC': RPAC, 'RFA': RFA,    # model_cfg['type'] -> class

# Helper
def full_fp32(seed):
    '''
    No TF32 anywhere: matmuls and convs stay in true float32
    '''
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(seed)

# ======================================================================================
# TRAINING LOOP
# ======================================================================================

def train_RASA(train_cfg_path, model_cfg_path, data_cfg_path, restart = False):

    train_cfg = load_config(train_cfg_path)
    model_cfg = load_config(model_cfg_path)
    data_cfg = load_config(data_cfg_path)

    full_fp32(seed = train_cfg['run']['seed'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt_dir = os.path.join(train_cfg['run']['checkpoint_dir'], model_cfg['name'])
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(model_cfg['model_save_dir'], exist_ok=True)

    model_type = MODELS[model_cfg['type']]
    model = model_type(cfg = model_cfg).to(device)
    writer = SummaryWriter(os.path.join(train_cfg['log']['tensorboard_dir'], model_cfg['name']))

    # START AT BEGINNING OR AT LAST SAVED
    first, step, state = 0, 0, None
    last = os.path.join(ckpt_dir, 'last.pt')
    if os.path.exists(last) and restart:
        state = torch.load(last, map_location=device)
        first, step = STAGES.index(state['stage']), state['step']
        if state['done']:
            first, state = first + 1, None

    # TRAINING LOOP
    for n in range(first, len(STAGES)):

        if state is not None:       # resume at this stage
            model.load_state_dict(state['model'])
        elif n > 0:
            best = os.path.join(ckpt_dir, f'{STAGES[n - 1]}_best.pt') #start from previous stage's best
            model.load_state_dict(torch.load(best, map_location=device))

        step = run_stage(
            model=model,
            stage_name=STAGES[n],
            train_cfg=train_cfg,
            data_cfg=data_cfg,
            device=device,
            writer=writer,
            step=step,
            ckpt_dir=ckpt_dir,
            state=state
        )
        state = None

    writer.flush()
    writer.close()

    # final model = best checkpoint of the last stage, not the last epoch
    best = os.path.join(ckpt_dir, f'{STAGES[-1]}_best.pt')
    model.load_state_dict(torch.load(best, map_location=device))
    torch.save(model.state_dict(), os.path.join(model_cfg['model_save_dir'], f'model.pt'))
    return model

# ======================================================================================
# TRAIN STAGE
# ======================================================================================

def run_stage(model, stage_name,
              train_cfg, data_cfg,
              device, writer, step, 
              ckpt_dir, state=None):
    '''
    one stage with early stopping
    '''

    batch_size = train_cfg['batch']['size']
    num_workers = train_cfg['batch']['num_workers']

    boundary_weight = train_cfg['loss']['boundary_weight']
    level_weight = train_cfg['loss']['level_weight']
    grad_weight = train_cfg['loss']['grad_weight']
    metric = 'val_loss' if stage_name =='stage1' else 'rollout_l1'

    # LOAD DATA
    train_DS = RASADataset(
                data_cfg=data_cfg, 
                split='train',
                K_max=train_cfg[stage_name]['K_max']
            )
    train_DL = DataLoader(
        dataset=train_DS,
        batch_size=batch_size,
        sampler=RandomSampler(data_source=train_DS,
                              num_samples=int(train_cfg['batch']['epoch_fraction'] * len(train_DS))),
        drop_last=True,
        num_workers=num_workers
    )

    val_DL = DataLoader(
        RASADataset(
            data_cfg=data_cfg, 
            split='val',
            K_max=train_cfg[stage_name]['K_max']
        ),
        batch_size=batch_size,
        num_workers=num_workers
    )

    rollouts = load_rollouts(
        data_cfg=data_cfg,
        steps=train_cfg['val']['rollout_steps']
    )

    # OPTIMIZER & SCHEDULER
    optimizer = torch.optim.AdamW(params=model.parameters(), 
                                  lr=train_cfg[stage_name]['lr'], 
                                  weight_decay=train_cfg['optim']['weight_decay'])
    scheduler = SequentialLR(optimizer=optimizer, 
                             schedulers=[LinearLR(optimizer, start_factor=1/train_cfg['optim']['warmup_steps'], 
                                                  total_iters=train_cfg['optim']['warmup_steps']),
                                         CosineAnnealingLR(optimizer,
                                                           T_max=train_cfg[stage_name]['max_epochs']*len(train_DL) - train_cfg['optim']['warmup_steps'],
                                                           eta_min=train_cfg['optim']['lr_min'])],
                                         milestones=[train_cfg['optim']['warmup_steps']])

    start, best, bad = 0, float('inf'), 0
    # load state if restarting
    if state is not None:
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start, best, bad = state['epoch'] + 1, state['best'], state['bad']

    print('---------------------', stage_name.upper(), '---------------------')

    for epoch in range(start, train_cfg[stage_name]['max_epochs']):
        model.train()
        for x in train_DL:
            
            K = int(torch.randint(1, train_cfg[stage_name]['K_max'] + 1, ()))
            cuda = device == 'cuda'
            if cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            loss, parts, grad_norm, out = train_step(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                x=x.to(device),
                K=K,
                boundary_weight=boundary_weight,
                level_weight=level_weight,
                grad_weight=grad_weight,
                clip=train_cfg['optim']['grad_clip']
            )
            if cuda:
                torch.cuda.synchronize()
            step_time = time.perf_counter() - t0
            step += 1

            # logging train
            if step % train_cfg['log']['every_steps'] == 0:
                scalars = {'train/loss': loss,
                           'train/grad_norm': grad_norm,
                           'train/lr': scheduler.get_last_lr()[0],
                           'train/K': K,
                           'perf/step_time_s': step_time,             # one optimiser step (K model steps)
                           'perf/time_per_model_step_s': step_time / K}
                if cuda:
                    scalars['perf/peak_mem_GB'] = torch.cuda.max_memory_allocated() / 1e9
                for k, v in parts.items():      # level0, level1, ..., log_rho, u, v, log_P, grad
                    group = 'level_loss' if k.startswith('level') else 'physics_loss'
                    scalars[f'{group}/{k}'] = v
                scalars.update({f'queries/level{l}': int(M.sum()) for l, M in enumerate(out['masks'])})
                # conservative variants: floored cells per level (the only step that breaks conservation)
                scalars.update({f'floor/level{l}': n for l, n in enumerate(out.get('floor', []))})
                for k, v in scalars.items():
                    writer.add_scalar(k, v, step)

        val, sample0 = validate(
            model=model,
            val_DL=val_DL,
            rollouts=rollouts,
            batch_size=batch_size,
            boundary_weight=boundary_weight,
            level_weight=level_weight,
            grad_weight=grad_weight,
            device=device
        )

        # logging validate
        for k, v in val.items():
            writer.add_scalar(f'val/{k}', v, step)
        if epoch % train_cfg['log']['image_every_epochs'] == 0:
            log_images(
                writer, model, *sample0, step
            )
        print(f'Epoch {epoch}:', val, flush=True)

        if val[metric] < best:
            best, bad = val[metric], 0
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f'{stage_name}_best.pt'))
        else:
            bad += 1

        done = bad >= train_cfg[stage_name]['patience'] or epoch == train_cfg[stage_name]['max_epochs'] - 1
        torch.save({'stage': stage_name, 'epoch': epoch, 'step': step, 'best': best, 'bad': bad, 'done': done,
                    'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()},
                    os.path.join(ckpt_dir, 'last.pt'))
        if done:
            break

    return step

def train_step(model, optimizer, scheduler, x, K, boundary_weight, level_weight, grad_weight, clip):
    '''
    x [B, K_max+1, 4, N, N] raw frames. K-1 steps without grad, then one trained step.
    -> loss, parts, grad norm before clipping, out
    '''
   
    X = x[:, 0]
    with torch.no_grad():
       for _ in range(K-1):
           X = X + model(X)

    dX, out = model(X, return_levels=True)
    loss, parts = per_level_loss(model, out, x[:, K], boundary_weight,
                                 level_weight=level_weight, grad_weight=grad_weight, X_pred=X + dX)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    scheduler.step()
    return loss.item(), parts, grad_norm.item(), out

@torch.no_grad()
def validate(model,val_DL, rollouts, batch_size, boundary_weight, level_weight, grad_weight, device = 'cpu'):
    '''
    -> metrics, and (predicted X, true X, masks) of val sample 0 at the last rollout step
    '''
    model.eval()
    mv = model.to_model_vars

    # single step on every val window
    losses, L1s = [], []
    for x in val_DL:
        X0, X1 = x[:, 0].to(device), x[:, 1].to(device)
        dX, out = model(X0, return_levels=True)

        losses.append(per_level_loss(model, out, X1, boundary_weight,
                                     level_weight=level_weight, grad_weight=grad_weight, X_pred=X0 + dX)[0].item())
        L1s.append((mv(X0 + dX) - mv(X1)).abs().mean().item())

    # rollouty from fram 0 of each val sample, scored at the last step
    errs, blowups, sample0 = [], 0, None
    for r in torch.split(rollouts, batch_size):
        r = r.to(device)
        X = r[:, 0]
        for _ in range(r.shape[1] - 2):
            X = X + model(X)
        dX, out = model(X, return_levels=True) 
        X = X + dX

        e = (mv(X) - mv(r[:, -1])).abs().mean(dim=(1, 2, 3)) # [b], NaN if it blows up
        blowups += int((~e.isfinite()).sum())
        errs += e[e.isfinite()].tolist()

        if sample0 is None:
            sample0 = (X[0], r[0, -1], [M[0] for M in out['masks']])

    mean = lambda v: sum(v) / len(v) if v else float('inf')
    return {'val_loss': mean(losses), 'val_l1_fine': mean(L1s),
            'rollout_l1': mean(errs), 'rollout_blowups': blowups}, sample0

# ======================================================================================
# MINOR TRAIN HELPERS
# ======================================================================================

def log_images(writer, model, X, X_true, masks, step):
    '''Log images of the rollout.'''
    '''Predicted | true log rho, then the level masks scaled up to N, side by side.'''
    a = model.to_model_vars(torch.stack([X, X_true]))[:, 0].nan_to_num()   # [2, N, N]
    a = (a - a.min()) / (a.max() - a.min() + 1e-12)
    writer.add_image('rollout/log_rho_pred_true', torch.cat([a[0], a[1]], dim=1)[None], step)
 
    N = X.shape[-1]
    up = lambda M: M.float().repeat_interleave(N // M.shape[0], 0).repeat_interleave(N // M.shape[1], 1)
    writer.add_image('rollout/masks_L0_L1_L2', torch.cat([up(M) for M in masks], dim=1)[None], step)
 

def load_rollouts(data_cfg, steps):
    '''Frames 0..steps of every val sample -> [S, steps+1, 4, N, N], kept on the CPU.'''
    return torch.stack([torch.from_numpy(load_window(sample_path(data_cfg, i), 0, steps,
                                                     data_cfg['step_stride'], data_cfg['gamma']))
                        for i in split_indices(data_cfg)['val']])

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--train', default='./experiments/bt_rasa/train_config.yaml')
    p.add_argument('--model', default='./experiments/bt_rasa/config.yaml')
    p.add_argument('--data', default='./experiments/bt_rasa/dataset_config.yaml')
    p.add_argument('--restart', action='store_true', help='resume from last.pt if it exists')
    args = p.parse_args()
 
    torch.cuda.set_per_process_memory_fraction(0.99)   # cap PyTorch at 95% of the GPU's memory
    # torch.cuda.memory._record_memory_history(max_entries=100000)
    # try:
    train_RASA(train_cfg_path=args.train,
                model_cfg_path=args.model,
                data_cfg_path=args.data,
                restart=args.restart)
    # except torch.cuda.OutOfMemoryError:
    #     torch.cuda.memory._dump_snapshot('oom.pickle')
    #     raise