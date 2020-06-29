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
from restools.plotting import cell_heatmap
from restools.plotting import label_axes
from papers.jfm2020_nonlinear_robustness.data import Summary
from comsdk.research import Research
from comsdk.comaux import load_from_json
from thequickmath.aux import index_for_almost_exact_coincidence


def _plot_edge_state_energy(ax, edge_state_energy_means: List[float], edge_state_energy_stds: List[float],
                            freqs: List[float], label: str):
    indices_with_none = [i for i, ke in enumerate(edge_state_energy_means) if ke is None]
    indices_with_none.append(len(edge_state_energy_means))
    last_i_with_none = -1
    color = None
    for next_i_with_none in indices_with_none:  # let's plot parts separated by null values of ke_mean
        indices = slice(last_i_with_none + 1, next_i_with_none)
        eb_cont = ax.errorbar(freqs[indices], edge_state_energy_means[indices],
                              yerr=edge_state_energy_stds[indices], color=color, fmt='--o', linewidth=2, capsize=3,
                              capthick=2, label=label)
        if color is None:
            color = eb_cont.lines[0].get_color()
        if label is not None:
            label = None
        last_i_with_none = next_i_with_none


def _find_edgetracking_paths(path):
    files_or_dirs = os.listdir(path)
    prefix = 'T_'
    return [file_or_dir for file_or_dir in files_or_dirs if file_or_dir.startswith(prefix)]


if __name__ == '__main__':
    summary = load_from_json(Summary)
    ti_builder = get_ti_builder(cache=False)
    res = Research.open(summary.edge_states_info.res_id)
    min_freq = summary.edge_states_info.frequencies[0]
    max_freq = summary.edge_states_info.frequencies[-1]

    # Plot edge state energies

    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(nrows=1, ncols=3)
    ax_edge_energy = fig.add_subplot(gs[0, :2])
    ax_cells = fig.add_subplot(gs[0, 2])

    unctrl_task_path = res.get_task_path(summary.edge_states_info.task_for_uncontrolled_case)
    ens = Ensemble([ti_builder.get_timeintegration(os.path.join(unctrl_task_path, 'edge_trajectory_integrated'))],
                   max_ke_eps=0.02, initial_cutoff_time=1000.)
    ke_distr = ens.ke_distribution()
    ax_edge_energy.plot([min_freq, max_freq], [ke_distr.mean(), ke_distr.mean()], color='black', linewidth=2,
                        label=r'$A = 0$')
    for a_i, a in enumerate(summary.edge_states_info.amplitudes):
        edge_state_energy_means = []
        edge_state_energy_stds = []
        for omega_i, omega in enumerate(summary.edge_states_info.frequencies):
            task = summary.edge_states_info.tasks[a_i][omega_i]
            print('Processing task {}'.format(task))
            if task == -1:
                edge_state_energy_means.append(None)
                edge_state_energy_stds.append(None)
                continue
            task_path = res.get_task_path(task)
            ti = ti_builder.get_timeintegration(os.path.join(task_path, 'edge_trajectory_integrated'))
            number_of_periods = int(omega / (2.*np.pi) * (np.max(ti.T) - 1000.))
            t_min = np.max(ti.T) - int(2.*np.pi*number_of_periods / omega)  # we wish to collect values from this value
            # to t_max to ensure we time-average over the integer number of periods
            try:
                ens = Ensemble([ti], max_ke_eps=0.02, initial_cutoff_time=t_min)
                ke_distr = ens.ke_distribution()
            except BadEnsemble as e:
                print('Configuration "A = {}, omega = {} (task {})" is skipped because the edge trajectory is '
                      'too short'.format(a, omega, task))
                edge_state_energy_means.append(None)
                edge_state_energy_stds.append(None)
            else:
                edge_state_energy_means.append(ke_distr.mean())
                edge_state_energy_stds.append(ke_distr.std())

        _plot_edge_state_energy(ax_edge_energy, edge_state_energy_means, edge_state_energy_stds,
                                summary.edge_states_info.frequencies, r'$A = ' + str(a) + r'$')
    ax_edge_energy.set_xscale('log', basex=2)
    ax_edge_energy.set_xlabel(r'$\omega$', fontsize=16)
    ax_edge_energy.set_ylabel(r'$E_{edge}$', fontsize=16)
    ax_edge_energy.set_ylim((0.003, 0.035))
    ax_edge_energy.legend(loc='upper right', fontsize=14)
    ax_edge_energy.grid()

    # Plot classification map

    omegas = summary.edge_states_info.frequencies
    amps = summary.edge_states_info.amplitudes
    RPO_id = 1
    PO_id = 2
    EQ_id = 3.5
    NA_id = -1.5
    edge_state_types = np.zeros((len(omegas), len(amps)))
    edge_state_types[index_for_almost_exact_coincidence(omegas, 1./8)][:] = PO_id
    edge_state_types[index_for_almost_exact_coincidence(omegas, 1./4):][:] = EQ_id
    edge_state_types[index_for_almost_exact_coincidence(omegas, 1./32)][index_for_almost_exact_coincidence(amps, 0.05)] = RPO_id
    edge_state_types[index_for_almost_exact_coincidence(omegas, 1./16)][index_for_almost_exact_coincidence(amps, 0.05)] = RPO_id
    edge_state_types[index_for_almost_exact_coincidence(omegas, 1./16)][index_for_almost_exact_coincidence(amps, 0.1)] = RPO_id
    edge_state_types[index_for_almost_exact_coincidence(omegas, 1./32)][index_for_almost_exact_coincidence(amps, 0.05)] = RPO_id
    edge_state_types[index_for_almost_exact_coincidence(omegas, 1./4)][index_for_almost_exact_coincidence(amps, 0.5)] = NA_id
    edge_state_types[index_for_almost_exact_coincidence(omegas, 1./8)][index_for_almost_exact_coincidence(amps, 0.5)] = NA_id
    edge_state_types[index_for_almost_exact_coincidence(omegas, 1./16)][index_for_almost_exact_coincidence(amps, 0.5)] = NA_id
    edge_state_types[index_for_almost_exact_coincidence(omegas, 1./32)][index_for_almost_exact_coincidence(amps, 0.5)] = NA_id

    qrates = np.array(['EQ', 'PO', 'RPO', 'C', 'N/A'])
    norm = matplotlib.colors.BoundaryNorm(np.array([-1.5, -0.5, 0.5, 1.5, 2.5, 3.5]), 5)
    fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: qrates[::-1][norm(x)])
    omega_labels = [r'$2^{' + str(i) + r'}$' for i in np.arange(-7, 5)]
    im, cbar = cell_heatmap(edge_state_types, omega_labels, amps,
                            ax=ax_cells,
                            cmap=plt.get_cmap('viridis', 5),
                            cbar_kw=dict(ticks=np.arange(-1, 4),
                                         format=fmt))
    ax_cells.set_xlabel(r'$A$', fontsize=16)
    ax_cells.set_ylabel(r'$\omega$', fontsize=16)
    label_axes(ax_edge_energy, label='(a)', loc=(0.5, 1.03), fontsize=16)
    label_axes(ax_cells, label='(b)', loc=(0.5, 1.03), fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig('edge_energy.png')
    plt.show()

    # Plot edge states

    selected_edge_states = [
        {'a': 0.4, 'omega': 1., 'type': 'EQ', 'label': '(a)'},
        {'a': 0.3, 'omega': 1./8, 'type': 'PO', 'label': '(b)'},
        {'a': 0.1, 'omega': 1./16, 'type': 'RPO', 'label': '(c)'},
        {'a': 0.2, 'omega': 1./16, 'type': 'C', 'label': '(d)'},
    ]
    for es_info in selected_edge_states:
        a_i = index_for_almost_exact_coincidence(summary.edge_states_info.amplitudes, es_info['a'])
        omega_i = index_for_almost_exact_coincidence(summary.edge_states_info.frequencies, es_info['omega'])
        task = summary.edge_states_info.tasks[a_i][omega_i]
        print('Processing task {}'.format(task))
        task_path = res.get_task_path(task)
        edgetracking_paths = _find_edgetracking_paths(task_path)
        init_cs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        edgetracking_cs = [0.25, 0.5, 0.75]
        Nt = -1

        fig = plt.figure(figsize=(6, 5))
        gs = gridspec.GridSpec(nrows=3, ncols=1)
        ax_edgetracking = fig.add_subplot(gs[0:2, 0])
        ax_edgetracking.set_rasterization_zorder(1)
        ax_ke = fig.add_subplot(gs[2, 0])
        axes = [ax_edgetracking, ax_ke]
        colors = 'bgrcmy'
        shadowed_mode = True
        alpha = 1
        min_aver_ke = np.inf
        if shadowed_mode:
            alpha = 0.25
        for c in init_cs:  # plot gray initial trajectories for edge-tracking
            ti = ti_builder.get_timeintegration(os.path.join(task_path, 'initial_conditions', 'data-{}'.format(c)))
            ke = 0.5 * ti.L2U**2
            axes[0].plot(ti.T, ke, linewidth=1.5, alpha=alpha, color='black', zorder=0)
        for i, path in enumerate(edgetracking_paths):  # plot coloured edge-tracking iterations
            data_paths = [os.path.join(task_path, path, 'data-{}'.format(c_)) for c_ in edgetracking_cs]
            for data_path, c in zip(data_paths, edgetracking_cs):
                if (not os.path.exists(data_path)) or ('summary.txt' not in os.listdir(data_path)):
                    continue
                ti = ti_builder.get_timeintegration(data_path)
                ke = 0.5 * ti.L2U**2
                axes[0].plot(ti.T, ke, '{}'.format(colors[i - int(i // len(colors))*len(colors)]), linewidth=1.5,
                             alpha=alpha, label='$c = {}$'.format(c), zorder=0)
        # plot edge trajectory
        ti = ti_builder.get_timeintegration(os.path.join(task_path, 'edge_trajectory_integrated'))
        ke = 0.5 * ti.L2U**2
        axes[0].plot(ti.T, ke, '--', color='black', linewidth=1.5)
        max_time = 2000.
        axes[0].set_xlim((0, max_time))
        axes[0].set_ylim((0, 0.11))
        axes[0].set_ylabel(r'$E$', fontsize=16)
        axes[0].grid()
        # plot xy-averaged KE of edge trajectory
        ke_field = ti.ke_z
        X_, Y_ = np.meshgrid(ke_field.space.t, ke_field.space.z, indexing='ij')
        cvals = 100
        cont = axes[1].contourf(X_, Y_, ke_field, cvals, cmap=matplotlib.cm.jet)
        obj_to_rasterize = [cont]
        axes[1].set_xlim((0, max_time))
        axes[1].set_ylabel('$z$', fontsize=16)
        axes[1].set_xlabel('$t$', fontsize=16)
        label_axes(axes[0], label=es_info['label'], loc=(0.5, 1.05), fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
#        fname = 'edgetracking_C_A_02_omega_1_16.eps'
#        rasterise_and_save(fname, rasterise_list=obj_to_rasterize, fig=fig, dpi=300)
#        reduce_eps_size(fname)
        plt.show()
