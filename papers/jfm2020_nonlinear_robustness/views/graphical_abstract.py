import os
import sys
from typing import List
sys.path.append(os.getcwd())

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from restools.timeintegration_builders import get_ti_builder
from restools.flow_stats import Ensemble, BadEnsemble
from restools.plotting import label_axes, cell_heatmap, rasterise_and_save, reduce_eps_size
from papers.jfm2020_nonlinear_robustness.data import Summary
from comsdk.research import Research
from comsdk.comaux import load_from_json
from thequickmath.aux import index_for_almost_exact_coincidence


def _find_edgetracking_paths(path):
    files_or_dirs = os.listdir(path)
    prefix = 'T_'
    return [file_or_dir for file_or_dir in files_or_dirs if file_or_dir.startswith(prefix)]


if __name__ == '__main__':
    plt.style.use('resources/default.mplstyle')
    summary = load_from_json(Summary)
    ti_builder = get_ti_builder(cache=False)
    res = Research.open(summary.edge_states_info.res_id)
    a_i = index_for_almost_exact_coincidence(summary.edge_states_info.amplitudes, 0.3)
    omega_i = index_for_almost_exact_coincidence(summary.edge_states_info.frequencies, 1./8)

    # Plot xy-averaged KE of one laminarizing trajectory
    task = summary.edge_states_info.tasks[a_i][omega_i]
    ti = ti_builder.get_timeintegration(os.path.join(res.get_task_path(task), 'initial_conditions', 'data-0.5'))
    fig, ax = plt.subplots(figsize=(6, 5))
    ke_field = ti.ke_z
    max_time = ke_field.space.t[-1]
    X_, Y_ = np.meshgrid(ke_field.space.t[1660:], ke_field.space.z, indexing='ij')
    cvals = 400
    cont = ax.contourf(X_, Y_, ke_field[1660:, :], cvals, cmap=matplotlib.cm.turbo)#cmap=matplotlib.cm.jet)      turbo,
#    obj_to_rasterize = [cont]
    #ax.set_xlim((0, max_time))
    ax.set_ylabel('$z$', fontsize=16)
    ax.set_xlabel('$t$', fontsize=16)
    plt.axis('off')
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 1.0])
    #plt.subplots_adjust(top=0.93)
    plt.savefig('graphical_abstract.jpg', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()
