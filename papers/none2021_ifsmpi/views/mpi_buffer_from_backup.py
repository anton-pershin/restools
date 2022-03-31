import os
import sys
sys.path.append(os.getcwd())
import time
import json

import numpy as np
import matplotlib.pyplot as plt

import restools

def median_filter(log_vals):
    log_vals_after_median_filter = np.zeros_like(log_vals)
    filter_half_width = 5
    log_vals_after_median_filter[0:filter_half_width] = log_vals[0:filter_half_width]
    log_vals_after_median_filter[-filter_half_width-1:-1] = log_vals[-filter_half_width-1:-1]
    for i in range(filter_half_width, len(log_vals) - filter_half_width):
        log_vals_after_median_filter[i] = np.median(log_vals[i-filter_half_width:i+filter_half_width+1])
    return log_vals_after_median_filter


def moving_average_filter(log_vals):
    log_vals_after_median_filter = np.zeros_like(log_vals)
    filter_half_width = 5
    log_vals_after_median_filter[0:filter_half_width] = log_vals[0:filter_half_width]
    log_vals_after_median_filter[-filter_half_width-1:-1] = log_vals[-filter_half_width-1:-1]
    for i in range(filter_half_width, len(log_vals) - filter_half_width):
        log_vals_after_median_filter[i] = np.mean(log_vals[i-filter_half_width:i+filter_half_width+1])
    return log_vals_after_median_filter


if __name__ == '__main__':
    #path_to_data = os.path.join('plots_xaver_255', 'trmtol', 'proc_1', '4.eps.data.npy')
    path_to_data = os.path.join('plots_xaver_639', 'trmtol', 'proc_1', '4.eps.data.npy')
    vals = np.load(path_to_data)
    with open('outputs.json', 'r') as f:
        jump_indices = json.load(f)['jump_indices']
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    #ax.plot(range(len(vals)), median_filter(np.log10(vals)), 'o')
    ax.plot(range(len(vals)), np.log10(vals), 'o')
    cut_i = 0
    for i in range(len(jump_indices)):
        if jump_indices[i] >= 4000:
            cut_i = i
            break
    ax.plot(jump_indices[:cut_i], np.log10(vals[jump_indices[:cut_i]]), 'ro', markersize=8)
    ax.grid()
#    start_i = 0
#    diff_array = np.abs(np.diff(np.log10(vals)))
#    for i, v in enumerate(diff_array):
#        if v != np.inf:
#            start_i = i
#            break
#    vals = vals[start_i:]
#    diff_array = diff_array[start_i:]
#    axes[0].semilogy(np.arange(len(vals)), vals, 'o', markersize=3)
#    axes[1].plot(np.arange(len(diff_array)), diff_array, 'o', markersize=3)
#    log_vals = np.log10(vals)
#    log_vals_after_median_filter = np.zeros_like(log_vals)
#    filter_half_width = 5
#    log_vals_after_median_filter[0:filter_half_width] = log_vals[0:filter_half_width]
#    log_vals_after_median_filter[-filter_half_width-1:-1] = log_vals[-filter_half_width-1:-1]
#    for i in range(filter_half_width, len(log_vals) - filter_half_width):
#        log_vals_after_median_filter[i] = np.median(log_vals[i-filter_half_width:i+filter_half_width+1])
#    axes[2].plot(np.arange(len(log_vals_after_median_filter)), log_vals_after_median_filter, 'o', markersize=3)
#    axes[3].plot(np.arange(len(log_vals_after_median_filter) - 1), np.abs(np.diff(log_vals_after_median_filter)), 'o', markersize=3)
#    for ax in axes:
#        ax.grid()
    plt.tight_layout()
    plt.show()
    #plt.savefig(savefig_path, dpi=200)
    print('qwer')
