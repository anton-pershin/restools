import os
import sys
sys.path.append(os.getcwd())

import matplotlib
import matplotlib.pyplot as plt

from restools.timeintegration_builders import get_ti_builder
from restools.flow_stats import Ensemble, BadEnsemble
from restools.laminar_flows import PlaneCouetteFlow, PlaneCouetteFlowWithInPhaseSpanwiseOscillations
from restools.plotting import label_axes
from papers.jfm2020_nonlinear_robustness.data import Summary
from papers.jfm2020_nonlinear_robustness.extensions import turbulent_dissipation_rate
from comsdk.research import Research
from comsdk.comaux import load_from_json
from thequickmath.differentiation.finite_differences import fd
from thequickmath.field import average


def _plot_average_turbulent_profile(ax, task, task_path, label=None, color=None):
    tis = [ti_builder.get_timeintegration(os.path.join(task_path, 'initial_conditions', 'data-{}'.format(c)))
           for c in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
    try:
        ens = Ensemble(tis, debug=True, max_ke_eps=0.02)
        av_turb_field = ens.time_averaged_solution()
    except BadEnsemble as e:
        raise Exception('Data is wrong: turbulent trajectories are too short for task {}'.format(task))
    av_turb_u = average(average(av_turb_field, ['u'], 'x'), ['u'], 'z').u
    dUdy = fd(av_turb_field.space.y, av_turb_u)
    ax.plot(dUdy, av_turb_field.space.y, label=label, color=color)


if __name__ == '__main__':
    plt.style.use('resources/default.mplstyle')
    summary = load_from_json(Summary)
    ti_builder = get_ti_builder(cache=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot laminar and turbulent dissipation rates

    res = Research.open(summary.simulations_with_full_fields_saved.res_id)
    min_freq = summary.simulations_with_full_fields_saved.frequencies[0]
    max_freq = summary.simulations_with_full_fields_saved.frequencies[-1]
    freqs = summary.simulations_with_full_fields_saved.frequencies
    unctrl_task_path = res.get_task_path(summary.simulations_with_full_fields_saved.task_for_uncontrolled_case)
    tis = [ti_builder.get_timeintegration(os.path.join(unctrl_task_path, 'data-500'))]
    ens = Ensemble(tis)
    unctrl_turb_diss_mean = ens.dissipation_distribution().mean()
    axes[1].plot([min_freq, max_freq], [unctrl_turb_diss_mean, unctrl_turb_diss_mean], color='black')
    re = summary.simulations_with_full_fields_saved.re
    unctrl_lam_diss_rate = re * PlaneCouetteFlow(re=re).dissipation_rate
    axes[1].plot([min_freq, max_freq], [unctrl_lam_diss_rate, unctrl_lam_diss_rate], 's--', color='black')
    turb_diss_means_at_a = []
    for a_i, a in enumerate(summary.simulations_with_full_fields_saved.amplitudes):
        turb_diss_means = []
        for omega_i, omega in enumerate(summary.simulations_with_full_fields_saved.frequencies):
            task = summary.simulations_with_full_fields_saved.tasks[a_i][omega_i]
            turb_diss_means.append(turbulent_dissipation_rate(task, a, omega, res, ti_builder))
        lam_diss_rate_means = [re * PlaneCouetteFlowWithInPhaseSpanwiseOscillations(
          re=re, a=a, omega=omega).dissipation_rate for omega in summary.simulations_with_full_fields_saved.frequencies]
        lines = axes[1].plot(freqs, turb_diss_means, 'o-', label=r'$W_{osc} = ' + str(a) + r'$')
        axes[1].plot(summary.simulations_with_full_fields_saved.frequencies, lam_diss_rate_means, 's--',
                     color=lines[0].get_color())
    axes[1].set_xscale('log', basex=2)
    axes[1].set_xlabel(r'$\omega$')
    axes[1].set_ylabel(r'$Re \times \epsilon_{turb}$')
    axes[1].grid()
    axes[1].legend(loc='center left',
                fancybox=True, fontsize=12)
    label_axes(axes[1], label='(b)', loc=(0.5, 1.05), fontsize=16)

    # Plot average turbulent profiles

    res = Research.open(summary.edge_states_info.res_id)
    task_path = res.get_task_path(summary.edge_states_info.task_for_uncontrolled_case)
    _plot_average_turbulent_profile(axes[0], summary.edge_states_info.task_for_uncontrolled_case, task_path,
                                    color='black')
    axes[0].plot([1., 1.], [-1., 1.], 'k--')  # laminar solution
    a_i = 3  # A = 0.3
    for omega_i, omega in enumerate(summary.edge_states_info.frequencies):
        if 1./32 <= omega <= 1./2:
            task = summary.edge_states_info.tasks[a_i][omega_i]
            print('Processing task {}'.format(task))
            if task == -1:
                raise Exception('Data is wrong: task = -1 at a_i = 3, omega_i = {}'.format(omega_i))
            task_path = res.get_task_path(task)
            _plot_average_turbulent_profile(axes[0], task, task_path, label=r'$\omega = 1/' + str(int(1./omega)) + r'$')
    axes[0].set_xlabel(r'$\mathrm{d}U/\mathrm{d}y$')
    axes[0].set_ylabel(r'$y$')
    axes[0].grid()
    axes[0].legend()
    label_axes(axes[0], label='(a)', loc=(0.5, 1.05), fontsize=16)
    plt.tight_layout()
    plt.savefig('turb_prof_and_diss.eps')
    plt.show()
