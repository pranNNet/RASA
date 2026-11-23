"""RASA data loading from per-sample files (NumPy only). The PyTorch Dataset lives in the training code."""
import os

import h5py
import numpy as np
import yaml

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def split_indices(cfg):
    """Sample indices for train / val / test, taken in file order."""
    s = cfg["split"]
    a, b = s["train"], s["train"] + s["val"]
    return {"train": list(range(a)),
            "val": list(range(a, b)),
            "test": list(range(b, b + s["test"]))}

def sample_path(cfg, i):
    return os.path.join(cfg["sample_dir"], f"sample_{i:03d}.h5")

def to_primitive(U, gamma):
    """Conserved (rho, rho u, rho v, E) [..., 4, n, n] -> raw (rho, u, v, P) [..., 4, n, n]."""
    rho, mx, my, E = (U[..., c, :, :] for c in range(4))
    u, v = mx / rho, my / rho
    P = (gamma - 1.0) * (E - 0.5 * rho * (u**2 + v**2))
    return np.stack([rho, u, v, P], axis=-3)

def frame_starts(T, K, stride):
    """Valid start frames n for a window n, n+s, ..., n+K*s inside one sample of T frames."""
    return range(T - K * stride)

def load_window(path, start, K, stride, gamma, coarsen=1):
    """Reads frames start, start+s, ..., start+K*s -> raw primitives [K+1, 4, N, N] float32.
    coarsen c > 1: c x c average of the conserved U before converting (uniform-grid ablation)."""
    with h5py.File(path, "r") as f:
        U = f["u"][start : start + K * stride + 1 : stride]
    if coarsen > 1:
        T, C, n, m = U.shape
        U = U.reshape(T, C, n // coarsen, coarsen, m // coarsen, coarsen).mean(axis=(3, 5))
    return to_primitive(U, gamma).astype(np.float32)
