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
from comsdk.misc import load_from_json
from comsdk.research import Research

if __name__ == '__main__':
    plt.style.use('resources/default.mplstyle')

    summary = load_from_json(Summary)
    summary_prob_proto = load_from_json(SummaryProbProto)
    ti_builder = get_ti_builder()
    res_diss = Research.open(summary.simulations_with_full_fields_saved.res_id)
    energies = 0.5 * np.r_[[0.], summary_prob_proto.energy_levels]
    re = summary.p_lam_info.re

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    turb_diss = np.zeros((len(summary.p_lam_info.amplitudes), len(summary.p_lam_info.frequencies)))
    lam_scores = np.zeros((len(summary.p_lam_info.amplitudes), len(summary.p_lam_info.frequencies)))
    lam_scores_exp = np.zeros((len(summary.p_lam_info.amplitudes), len(summary.p_lam_info.frequencies)))
    min_turb_diss_rate = 1000.
    fitting_no_ctrl = LaminarisationProbabilityFittingFunction2020JFM.from_data(
        0.5 * np.array([0.] + summary_prob_proto.energy_levels), np.array([1.] + summary_prob_proto.confs[1].p_lam))
    lam_score_no_ctrl = fitting_no_ctrl.expected_probability()
    lam_score_exp_no_ctrl = fitting_no_ctrl.expected_probability(noise_distribution=exponential_noise_distribution)
    lam_diss_no_ctrl = re * PlaneCouetteFlow(re=re).dissipation_rate
    turb_diss_no_ctrl = turbulent_dissipation_rate(summary.simulations_with_full_fields_saved.task_for_uncontrolled_case,
                                                   0., 0., res_diss, ti_builder)
    for ax, lam_score in zip(axes, (lam_score_no_ctrl, lam_score_exp_no_ctrl)):
        ax.plot(summary.p_lam_info.frequencies,
                len(summary.p_lam_info.frequencies)*[lam_score * lam_diss_no_ctrl + (1. - lam_score) * turb_diss_no_ctrl],
                'k-', label=r'$W_{osc} = 0$')
    for a_i, amplitude in enumerate(summary.p_lam_info.amplitudes):
        for omega_i, frequency in enumerate(summary.p_lam_info.frequencies):
            if a_i < len(summary.simulations_with_full_fields_saved.amplitudes):
                task = summary.simulations_with_full_fields_saved.tasks[a_i][omega_i]
                turb_diss_ = turbulent_dissipation_rate(task, amplitude, frequency, res_diss, ti_builder)
            else:
                turb_diss_ = 0.
            if turb_diss_ is None:
                turb_diss_ = 0.
            if turb_diss_ != 0. and turb_diss_ < min_turb_diss_rate:
                min_turb_diss_rate = turb_diss_
            turb_diss[a_i][omega_i] = turb_diss_
            lam_scores[a_i][omega_i] = summary.p_lam_info.s[a_i][omega_i][1]
            lam_scores_exp[a_i][omega_i] = summary.p_lam_info.s_exp[a_i][omega_i][1]
    for a_i, amplitude in enumerate(summary.p_lam_info.amplitudes):
        for omega_i, frequency in enumerate(summary.p_lam_info.frequencies):
            if turb_diss[a_i][omega_i] == 0.:
                turb_diss[a_i][omega_i] = min_turb_diss_rate
        lam_diss = np.array([re * PlaneCouetteFlowWithInPhaseSpanwiseOscillations(
          re=re, a=amplitude, omega=frequency).dissipation_rate for omega in summary.p_lam_info.frequencies])
        axes[0].plot(summary.p_lam_info.frequencies, lam_scores[a_i] * lam_diss + (1. - lam_scores[a_i]) * turb_diss[a_i],
                     'o-', label=r'$W_{osc} = ' + str(amplitude) + r'$')
        axes[1].plot(summary.p_lam_info.frequencies,
                     lam_scores_exp[a_i] * lam_diss + (1. - lam_scores_exp[a_i]) * turb_diss[a_i],
                     'o-', label=r'$W_{osc} = ' + str(amplitude) + r'$')
    for ax, ylabel, title in zip(axes, (r'$S$', r'$E_a$', r'$E_{flex}$'), (r'(a)', r'(b)')):
        ax.grid()
        ax.set_xlabel(r'$\omega$')
        ax.set_xscale('log', basex=2)
        ax.set_xticks(summary.p_lam_info.frequencies)
        ax.set_xticklabels([r'$2^{' + str(int(np.log2(summary.p_lam_info.frequencies[i]))) + r'}$'
                            for i in range(len(summary.p_lam_info.frequencies))])
        label_axes(ax, label=title, loc=(0.47, 1.05), fontsize=16)
    #axes[0].set_ylabel(r'$Re \times \mathbb{E} [\epsilon | W_{osc}, \omega]$')
    axes[0].set_ylabel(r'$Re \times \bar{\epsilon}$')
    axes[1].legend(bbox_to_anchor=(0.75, 0.0, 0.9, 1.02), loc='upper center',
                   ncol=1, fancybox=True)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, right=0.86, wspace=0.08)
    fname = 'expected_diss_rate.eps'
    plt.savefig(fname)
#    rasterise_and_save(fname, rasterise_list=obj_to_rasterize, fig=fig, dpi=300)
#    reduce_eps_size(fname)
    plt.show()
