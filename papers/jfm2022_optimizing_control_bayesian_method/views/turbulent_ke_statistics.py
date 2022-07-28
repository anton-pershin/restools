import os
import sys
from typing import List, Any
sys.path.append(os.getcwd())

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from restools.timeintegration_builders import get_ti_builder
from restools.flow_stats import Ensemble, BadEnsemble
from restools.plotting import label_axes
from papers.jfm2022_optimizing_control_bayesian_method.data import Summary
from papers.jfm2022_optimizing_control_bayesian_method.extensions import DistributionSummary
from comsdk.research import Research
from comsdk.misc import load_from_json


def _plot_ke_distribution(ax, freqs, obj_to_rasterize: List[Any], color, distr_summary: DistributionSummary):
    indices_with_none = [i for i, ke in enumerate(distr_summary.means) if ke is None]
    indices_with_none.append(len(distr_summary.means))
    last_i_with_none = -1
    for next_i_with_none in indices_with_none:  # let's plot parts separated by null values of ke_mean
        indices = slice(last_i_with_none + 1, next_i_with_none)
        ax.plot(freqs[indices], distr_summary.means[indices], linewidth=3, color=color)
        obj = ax.fill_between(freqs[indices], distr_summary.lower_quartiles[indices],
                              distr_summary.upper_quartiles[indices], color=color, alpha=0.5, zorder=-10)
        obj_to_rasterize.append(obj)
        obj = ax.fill_between(freqs[indices], distr_summary.lower_deciles[indices],
                              distr_summary.upper_deciles[indices], color=color, alpha=0.2, zorder=-10)
        obj_to_rasterize.append(obj)
        last_i_with_none = next_i_with_none


def _append_ke_statistics_and_return_block_extrema(res, task, data_path_generator, ti_builder, ke_summary,
                                                   **ensemble_kwargs):
    task_path = res.get_task_path(task)
    # first, collect statistics for turbulence
    paths = data_path_generator(task_path)
    tis = [ti_builder.get_timeintegration(p) for p in data_path_generator(task_path)]
    try:
        ens = Ensemble(tis, **ensemble_kwargs)
        ke_distr = ens.ke_distribution()
        print('\tNumber of data samples: {}. Assuming delta t = 0.5, this corresponds to {} time units '
              'of time series'.format(len(ke_distr.data_samples), 0.5*(len(ke_distr.data_samples) - 1)))
        block_minima, block_maxima = ens.block_extrema('L2U', transform=lambda d: 0.5*d**2)
    except BadEnsemble as e:
        print('Configuration "A = {}, omega = {} (task {})" is skipped because turbulent trajectories are '
              'too short'.format(a, omega, task))
        ke_summary.append()
        block_minima = None
        block_maxima = None
    else:
        quantiles = ke_distr.ppf([0.1, 0.25, 0.75, 0.9])
        ke_summary.append(mean=ke_distr.mean(), lower_decile=quantiles[0], lower_quartile=quantiles[1],
                          upper_quartile=quantiles[2], upper_decile=quantiles[3])
    return block_minima, block_maxima


def _plot_edge_and_turb_ke_statistics(ax, omega, task, ti_builder, turb_ke_distr, edge_ke_distr):
    # first, collect statistics for turbulence
    data_path_generator = lambda task_p: [os.path.join(task_p, 'initial_conditions', 'data-{}'.format(c))
                                for c in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
    block_minima, block_maxima = _append_ke_statistics_and_return_block_extrema(res, task,
                                                                                data_path_generator,
                                                                                ti_builder,
                                                                                turb_ke_distr,
                                                                                max_ke_eps=0.02)
    if block_minima is not None:
        ax.plot([omega]*len(block_minima), block_minima, 'o', color='blue', markersize=4, alpha=0.5, zorder=-10)
    if block_maxima is not None:
        ax.plot([omega]*len(block_maxima), block_maxima, 'o', color='red', markersize=4, alpha=0.5, zorder=-10)

    # second, collect statistics for edge states
    data_path_generator = lambda task_p: [os.path.join(task_p, 'edge_trajectory_integrated')]
    _append_ke_statistics_and_return_block_extrema(res, task, data_path_generator, ti_builder, edge_ke_distr,
                                                   initial_cutoff_time=1000., max_ke_eps=0.02)


if __name__ == '__main__':
    plt.style.use('resources/default.mplstyle')
    summary = load_from_json(Summary)
    res = Research.open(summary.edge_states_info.res_id)
    ti_builder = get_ti_builder(cache=True)
    obj_to_rasterize = []
    fig = plt.figure(figsize=(12, 6))
    cols_for_unctrl = 2
    cols_for_any_other_ctrl = 5
    gs = gridspec.GridSpec(nrows=1, ncols=cols_for_unctrl + cols_for_any_other_ctrl*4, figure=fig)
    ax_unctrl = fig.add_subplot(gs[0, :2])
    axes = [fig.add_subplot(gs[0, i : i + cols_for_any_other_ctrl])
            for i in range(cols_for_unctrl, cols_for_unctrl + cols_for_any_other_ctrl*4, cols_for_any_other_ctrl)]
    ylim = (0., 0.16)
    # plot uncontrolled turbulent ke and edge state ke
    print('Processing task {}. A = 0'.format(summary.edge_states_info.task_for_uncontrolled_case))
    turb_ke_distr = DistributionSummary()
    edge_ke_distr = DistributionSummary()
    _plot_edge_and_turb_ke_statistics(ax_unctrl, 0.5, summary.edge_states_info.task_for_uncontrolled_case, ti_builder,
                                      turb_ke_distr, edge_ke_distr)
    # copy the same data to make a plot nicer
    for s in (turb_ke_distr, edge_ke_distr):
        s.means.append(s.means[-1])
        s.lower_quartiles.append(s.lower_quartiles[-1])
        s.upper_quartiles.append(s.upper_quartiles[-1])
        s.lower_deciles.append(s.lower_deciles[-1])
        s.upper_deciles.append(s.upper_deciles[-1])
    _plot_ke_distribution(ax_unctrl, [0, 1], obj_to_rasterize, 'blue', turb_ke_distr)
    _plot_ke_distribution(ax_unctrl, [0, 1], obj_to_rasterize, 'green', edge_ke_distr)
    label_axes(ax_unctrl, label=r'$W_{osc} = 0$', loc=(-0.02, 1.03), fontsize=16)
    ax_unctrl.set_xticks([])
    ax_unctrl.set_ylim(ylim)
    ax_unctrl.grid()
    for a_i, a in enumerate(summary.edge_states_info.amplitudes[:4]):
        axes[a_i].plot([1./8, 1./8], ylim, 'k--', linewidth=2)
        turb_ke_distr = DistributionSummary()
        edge_ke_distr = DistributionSummary()
        freqs = []
        for omega_i, omega in enumerate(summary.edge_states_info.frequencies):
            task = summary.edge_states_info.tasks[a_i][omega_i]
            print('Processing task {}. A = {}, omega = {}'.format(task, a, omega))
            if task != -1:
                _plot_edge_and_turb_ke_statistics(axes[a_i], omega, task, ti_builder, turb_ke_distr, edge_ke_distr)
                freqs.append(omega)
        _plot_ke_distribution(axes[a_i], freqs, obj_to_rasterize, 'blue', turb_ke_distr)
        _plot_ke_distribution(axes[a_i], freqs, obj_to_rasterize, 'green', edge_ke_distr)
        label_axes(axes[a_i], label=r'$W_{osc} = ' + str(a) + r'$', loc=(0.3, 1.03), fontsize=16)
        axes[a_i].set_xscale('log', base=2)
        axes[a_i].set_xlabel(r'$\omega$', fontsize=16)
        axes[a_i].set_xlim((5*10**(-3), 20.))
        axes[a_i].set_xticks([2**(-7), 2**(-5), 2**(-3), 2**(-1), 2**(1), 2**(3)])
        axes[a_i].set_ylim(ylim)
        axes[a_i].set_yticklabels([])
        axes[a_i].set_rasterization_zorder(-5)
        axes[a_i].grid()
    ax_unctrl.set_ylabel(r'$E$', fontsize=16)
    ax_unctrl.set_rasterization_zorder(-5)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig('turb_attractor_estimate.eps')
    plt.show()
