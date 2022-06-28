import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from restools.timeintegration_builders import get_ti_builder
from restools.laminar_flows import PlaneCouetteFlow, PlaneCouetteFlowWithInPhaseSpanwiseOscillations
from restools.plotting import label_axes
from papers.jfm2022_optimizing_control_bayesian_method.data import Summary
from papers.jfm2022_optimizing_control_bayesian_method.extensions import turbulent_dissipation_rate, exponential_noise_distribution
from papers.jfm2020_probabilistic_protocol.data import Summary as SummaryProbProto
from papers.jfm2020_probabilistic_protocol.extensions import LaminarisationProbabilityFittingFunction2020JFM
from comsdk.comaux import load_from_json
from comsdk.research import Research

if __name__ == '__main__':
    plt.style.use('resources/default.mplstyle')
    summary = load_from_json(Summary)
    min_freq = summary.minimal_seed_info.frequencies[0]
    max_freq = summary.minimal_seed_info.frequencies[-1]

    # Plot edge state energies

    fig, ax = plt.subplots(figsize=(6, 5))
    minimal_seed_ke_unctrl = summary.minimal_seed_info.kinetic_energy_for_uncontrolled_case
    ax.plot([min_freq, max_freq], [minimal_seed_ke_unctrl, minimal_seed_ke_unctrl], color='black', linewidth=2,
            label=r'$W_{osc} = 0$')
    for a_i, a in enumerate(summary.minimal_seed_info.amplitudes):
        minimal_seed_ke = summary.minimal_seed_info.kinetic_energies[a_i]
        if a_i == 1:
            ax.plot(summary.minimal_seed_info.frequencies[-1:], minimal_seed_ke[-1:], '--o', color='tab:orange', linewidth=2, markersize=10,
                    label=r'$W_{osc} = ' + str(a) + r'$')
            ax.plot(summary.minimal_seed_info.frequencies[:-1], minimal_seed_ke[:-1], '--o', color='tab:orange', linewidth=2, markersize=10,
                    fillstyle='none', label=r'$W_{osc} = ' + str(a) + r'$' + '\n(not converged)')
            ax.plot(summary.minimal_seed_info.frequencies[-2:], minimal_seed_ke[-2:], '--', color='tab:orange', linewidth=2)
        else:
            ax.plot(summary.minimal_seed_info.frequencies, minimal_seed_ke, '--o', linewidth=2, markersize=10,
                    label=r'$W_{osc} = ' + str(a) + r'$')

    ax.set_xscale('log', basex=2)
    ax.set_xlabel(r'$\omega$', fontsize=16)
    ax.set_ylabel(r'$E_c$', fontsize=16)
    #ax.set_ylim((0.003, 0.035))
    ax.legend(loc='upper right', fontsize=14)
    ax.grid()
    label_axes(ax, label='(a)', loc=(0.47, 1.05), fontsize=16)
    plt.tight_layout()
#    plt.subplots_adjust(top=0.9, right=0.86, wspace=0.08)
    fname = 'minimal_seed_ke.eps'
    plt.savefig(fname)
#    rasterise_and_save(fname, rasterise_list=obj_to_rasterize, fig=fig, dpi=300)
#    reduce_eps_size(fname)
    plt.show()
