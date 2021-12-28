import os
import sys
sys.path.append(os.getcwd())
import pickle
import json

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import restools
from papers.none2021_ecrad.data import Summary
from papers.none2021_ecrad.extensions import EcradIO, heating_rate_diff_rms, plot_data_on_mixed_linear_log_scale
from comsdk.research import Research
from comsdk.comaux import load_from_json, find_all_files_by_named_regexp
from reducedmodels.transition_to_turbulence import MoehlisFaisstEckhardtModel


def turn_to_int(s):
    return int(s.strip())


def turn_to_float(s):
    return float(s.strip())


if __name__ == '__main__':
    plt.style.use('resources/default.mplstyle')

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.fill_between(np.array([10**(-38), 10**38], dtype=np.float64), y1=0, y2=10)
    ax.fill_between([10**(-5), 10**5], y1=0, y2=20)
    ax.set_xscale('log')
    ax.set_xlabel(r'K $\times$ d$^{-1}$')
    ax.set_xticks(np.array([10**(-38), 10**(-35), 10**(-32), 10**(-7), 10**(-5), 10**(-3), 10**(-1), 10, 10**(3), 10**(5), 10**(7), 10**(32), 10**(35), 10**(38)], dtype=np.float64))
    plt.tight_layout()
#    plt.savefig(f'heating_rates_task_{task}.eps', dpi=200)
    plt.show()
