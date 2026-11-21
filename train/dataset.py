'''
dataset.py

map-style dataset for rasa training
'''
import h5py
import torch
from torch.utils.data import Dataset

from preprocess.data import frame_starts, load_window, sample_path, split_indices



class RASADataset(Dataset):
    '''
    Item (sample i, start n) -> raw (rho, u, v, P) frames n, n+s, ..., n+K_max [K_max+1, 4, Nx, Ny]
    '''

    def __init__(self, data_cfg, split, K_max):
        self.K_max, self.stride, self.gamma = K_max, data_cfg['step_stride'], data_cfg['gamma']
        self.items = []

        for i in split_indices(data_cfg)[split]:
            path = sample_path(data_cfg, i)
            with h5py.File(path, 'r') as f:     # read T only
                T = f['u'].shape[0]
            self.items += [(path, n) for n in frame_starts(T, K_max, self.stride)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, n = self.items[idx]
        return torch.from_numpy(load_window(path, n, self.K_max, self.stride, self.gamma))

