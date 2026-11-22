'''
split_dataset.py
- seperates a parc-gym dataset.h5 file into seperate sample.h5 files
    - works with multiple dataset.h5 files to make a unified dataset of sample files
- Run: python split_dataset.py data_config.yaml"""
'''
import os
import sys

import h5py
import yaml



def split_dataset(raw_files, sample_dir):
    os.makedirs(sample_dir, exist_ok=True)
    k = 0
    for raw in raw_files:
        with h5py.File(raw, 'r') as f:
            for i in sorted(f['samples'], key=int):
                g = f['samples'][i]
                with h5py.File(os.path.join(sample_dir, f'sample_{k:03d}.h5'), 'w') as out:
                    for name in ('u', 't', 'x', 'y'):
                        out[name] = g[name][...]
                    out.attrs.update(g.attrs)
                    out.attrs['source_file'] = os.path.basename(raw)
                    out.attrs['source_index'] = int(i)
                k += 1
    return k


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        cfg = yaml.safe_load(f)
    print("wrote", split_dataset(cfg["raw_files"], cfg["sample_dir"]), "samples")
