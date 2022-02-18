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

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot([1, 2, 3], [3, 4, 1], 'o--')
    ax.grid()
    plt.savefig(inputs['figure_filename'], dpi=200)
    
    finished_file = open('__finished__', 'w')
    finished_file.close()
