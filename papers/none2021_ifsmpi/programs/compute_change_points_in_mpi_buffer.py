#!/usr/bin/env python3
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
import json
import argparse
import concurrent.futures
import pickle
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from misc import upload_paths_from_config
upload_paths_from_config()
from pyESN import optimal_esn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot MPI buffers dumped from TRANS package.')
    parser.add_argument('--cores', metavar='CORES', type=int, default=4, help='allowed number of cores')
    parser.add_argument('filename', metavar='FILENAME', nargs='?', help='input parameters filename')
    args = parser.parse_args()
    with open(args.filename, 'r') as f:
        inputs = json.load(f)
    os.environ["OMP_NUM_THREADS"] = str(args.cores)
    os.environ["MKL_NUM_THREADS"] = str(args.cores)
    os.environ["NUMEXPR_NUM_THREADS"] = str(args.cores)
    
    with open(os.path.join(inputs['path_to_buffers'], inputs['subroutine'], f"proc_{inputs['proc_id']}", f"{inputs['buffer_id']}.csv")) as f:
        vals = np.array([float(x) for x in f.read().split()])
        vals = np.abs(vals)
    buf = vals[:10000]
    kfield = 500
    #KFIELD =          552
    #KFIELD =          552
    #KFIELD =          544
    #KFIELD =          544
    n_blocks = len(buf) / inputs['kfield']
    if n_blocks - int(n_blocks) != 0.0:
        ValueError(f'KFIELD is incorrect. The resulting number of blocks is {n_blocks}')
    n_blocks = int(n_blocks)
    i = kfield + 4  # array is indexed from 0
    jump_indices = []
    while i < len(buf):
        moving_average = np.mean(buf[i-4:i])
        if i % kfield == 0:
            jump_indices.append(i)
            i += 4
        elif np.abs(np.log10(moving_average)) > 5:
            jump_indices.append(i)
            i += 4
        else:
            i += 1
    with open(inputs['output_filename'], 'r') as f:
        json.dump({
            'jump_indices': jump_indices,
        }, f)
    finished_file = open('__finished__', 'w')
    finished_file.close()