import os
import sys
from typing import List, Tuple, Any
from functools import reduce
sys.path.append(os.getcwd())

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import restools
from restools.timeintegration_builders import get_ti_builder
from restools.plotting import label_axes, build_zooming_axes_for_plotting_with_box, rasterise_and_save, reduce_eps_size
from papers.jfm2020_probabilistic_protocol.data import Summary, SingleConfiguration
from papers.jfm2020_probabilistic_protocol.extensions import RandomPerturbationFilenameJFM2020, OrthogonalComponentOfRandomPerturbationFilenameJFM2020
from comsdk.comaux import load_from_json, find_all_files_by_standardised_naming
from comsdk.research import Research
from thequickmath.field import read_field, L2_norms


if __name__ == '__main__':
    plt.style.use('resources/default.mplstyle')

    summary = load_from_json(Summary)
    ti_builder = get_ti_builder()
    unctrl_re_500_conf = summary.confs[1]
    any_energy_level = summary.energy_levels[9]
    res = Research.open(unctrl_re_500_conf.res_id)
    E_u = []
    E_v = []
    E_w = []
    # RandomPerturbationFilenameJFM2020 = A * U_perp + B * U_lam
    # OrthogonalComponentOfRandomPerturbationFilenameJFM2020 = U_perp
    filename_to_use = OrthogonalComponentOfRandomPerturbationFilenameJFM2020  
    for task in unctrl_re_500_conf.tasks:
        print(f'Reading task {task}...')
        task_path = res.get_task_path(task)
        file_datum = find_all_files_by_standardised_naming(filename_to_use, task_path)
        for file, data in file_datum:
            if data['energy_level'] == any_energy_level:
                f, _ = read_field(os.path.join(task_path, file))
                norms = L2_norms(f, normalize=True) 
                E_u.append(0.5*norms[0]**2)
                E_v.append(0.5*norms[1]**2)
                E_w.append(0.5*norms[2]**2)
    E_u = np.array(E_u)
    E_v = np.array(E_v)
    E_w = np.array(E_w)

    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    for ax, x, y, x_label, y_label in zip(axes, (E_u, E_u, E_v), 
                                                (E_v, E_w, E_w), 
                                                (r'$E_u$', r'$E_u$', r'$E_v$'),
                                                (r'$E_v$', r'$E_w$', r'$E_w$')):
        ax.scatter(x, y)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid()
#    u_lims = axes[0].get_xlim()
#    y_lim = -0.00007
#    axes[0].set_ylim((y_lim, 0.00145))
#    for ax in (axes[0], axes[1]):
#        ax.set_xlim((0.0075, u_lims[1]))
#    for ax in (axes[1], axes[2]):
#        ax.set_ylim((y_lim, 0.001))
    plt.tight_layout()
    plt.savefig('rp_distributions.png')
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    for ax, x, x_label in zip(axes, (E_u, E_v, E_w), (r'$E_u$', r'$E_v$', r'$E_w$')):
        ax.hist(x)
        ax.set_xlabel(x_label)
        ax.grid()
#    u_lims = axes[0].get_xlim()
#    y_lim = -0.00007
#    axes[0].set_ylim((y_lim, 0.00145))
#    for ax in (axes[0], axes[1]):
#        ax.set_xlim((0.0075, u_lims[1]))
#    for ax in (axes[1], axes[2]):
#        ax.set_ylim((y_lim, 0.001))
    plt.tight_layout()
#    plt.savefig('rp_distributions.png')
    plt.show()