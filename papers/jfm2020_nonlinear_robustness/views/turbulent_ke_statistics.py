import os
import sys
from typing import List, Any
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt

from restools.timeintegration_builders import get_ti_builder
from restools.flow_stats import Ensemble, BadEnsemble
from restools.plotting import label_axes
from papers.jfm2020_nonlinear_robustness.data import Summary
from comsdk.research import Research
from comsdk.comaux import load_from_json


class DistributionSummary:
    def __init__(self):
        self.means = []
        self.lower_quartiles = []
        self.upper_quartiles = []
        self.lower_deciles = []
        self.upper_deciles = []

    def append(self, mean=None, lower_quartile=None, upper_quartile=None, lower_decile=None, upper_decile=None):
        self.means.append(mean)
        self.lower_quartiles.append(lower_quartile)
        self.upper_quartiles.append(upper_quartile)
        self.lower_deciles.append(lower_decile)
        self.upper_deciles.append(upper_decile)


def _plot_ke_distribution(ax, obj_to_rasterize: List[Any], color, distr_summary: DistributionSummary):
    indices_with_none = [i for i, ke in enumerate(distr_summary.means) if ke is None]
    indices_with_none.append(len(distr_summary.means))
    last_i_with_none = -1
    for next_i_with_none in indices_with_none:  # let's plot parts separated by null values of ke_mean
        indices = slice(last_i_with_none + 1, next_i_with_none)
        axes[a_i].plot(freqs[indices], distr_summary.means[indices], linewidth=3, color=color)
        obj = ax.fill_between(freqs[indices], distr_summary.lower_quartiles[indices],
                              distr_summary.upper_quartiles[indices], color=color, alpha=0.5)
        obj_to_rasterize.append(obj)
        obj = ax.fill_between(freqs[indices], distr_summary.lower_deciles[indices],
                              distr_summary.upper_deciles[indices], color=color, alpha=0.2)
        obj_to_rasterize.append(obj)
        last_i_with_none = next_i_with_none


if __name__ == '__main__':
    summary = load_from_json(Summary)
    res = Research.open(summary.edge_states_info.res_id)
    ti_builder = get_ti_builder(cache=True)
    fig, axes = plt.subplots(1, 4, figsize=(12, 6))
    ylim = (0., 0.16)
    for a_i, a in enumerate(summary.edge_states_info.amplitudes[:4]):
        axes[a_i].plot([1./8, 1./8], ylim, 'k--', linewidth=2)
        turb_ke_distr = DistributionSummary()
        edge_ke_distr = DistributionSummary()
        freqs = []
        for omega_i, omega in enumerate(summary.edge_states_info.frequencies):
            task = summary.edge_states_info.tasks[a_i][omega_i]
            print('Processing task {}'.format(task))
            if task != -1:
                task_path = res.get_task_path(task)
                # first, collect statistics for turbulence
                tis = [ti_builder.get_timeintegration(os.path.join(task_path,
                                                                   'initial_conditions', 'data-{}'.format(c)))
                       for c in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
                try:
                    ens = Ensemble(tis, max_ke_eps=0.02)
                    ke_distr = ens.ke_distribution()
                    block_minima, block_maxima = ens.block_extrema('L2U', transform=lambda d: 0.5*d**2)
                except BadEnsemble as e:
                    print('Configuration "A = {}, omega = {} (task {})" is skipped because turbulent trajectories are '
                          'too short'.format(a, omega, task))
                    turb_ke_distr.append()
                else:
                    quantiles = ke_distr.ppf([0.1, 0.25, 0.75, 0.9])
                    turb_ke_distr.append(mean=ke_distr.mean(), lower_decile=quantiles[0], lower_quartile=quantiles[1],
                                         upper_quartile=quantiles[2], upper_decile=quantiles[3])
                    axes[a_i].plot([omega]*len(block_maxima), block_maxima, 'o', color='red', markersize=4, alpha=0.5)
                    axes[a_i].plot([omega]*len(block_minima), block_minima, 'o', color='blue', markersize=4, alpha=0.5)

                # second, collect statistics for edge states
                tis = [ti_builder.get_timeintegration(os.path.join(task_path, 'edge_trajectory_integrated'))]
                try:
                    ens = Ensemble(tis, initial_cutoff_time=1000., max_ke_eps=0.02)
                    ke_distr = ens.ke_distribution()
                except BadEnsemble as e:
                    print('Configuration "A = {}, omega = {} (task {})" is skipped because edge trajectory is '
                          'too short'.format(a, omega, task))
                    edge_ke_distr.append()
                else:
                    quantiles = ke_distr.ppf([0.1, 0.25, 0.75, 0.9])
                    edge_ke_distr.append(mean=ke_distr.mean(), lower_decile=quantiles[0], lower_quartile=quantiles[1],
                                         upper_quartile=quantiles[2], upper_decile=quantiles[3])
                freqs.append(omega)
        obj_to_rasterize = []
        _plot_ke_distribution(axes[a_i], obj_to_rasterize, 'blue', turb_ke_distr)
        _plot_ke_distribution(axes[a_i], obj_to_rasterize, 'green', edge_ke_distr)
        label_axes(axes[a_i], label=r'$A = ' + str(a) + r'$', loc=(0.3, 1.03), fontsize=16)
        axes[a_i].set_xscale('log', basex=2)
        axes[a_i].set_xlabel(r'$\omega$', fontsize=16)
        axes[a_i].set_xlim((5*10**(-3), 20.))
        axes[a_i].set_ylim(ylim)
        axes[a_i].grid()
    axes[0].set_ylabel(r'$E$', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig('turb_attractor_estimate.png')
    plt.show()
