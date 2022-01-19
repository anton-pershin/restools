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
from thequickmath.misc import np_index


def _plot_ke_and_turbulent_profile(ax_ke, ax_profile, task, task_path, c):
    ti = ti_builder.get_timeintegration(os.path.join(task_path, 'initial_conditions', 'data-{}'.format(c)))
    ti
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
    ti_builder = get_ti_builder()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot average turbulent profiles

    res = Research.open(summary.edge_states_info.res_id)
    a_i = summary.edge_states_info.amplitudes.index(0.3)
    omega_i = summary.edge_states_info.frequencies.index(1./8)
    task = summary.edge_states_info.tasks[a_i][omega_i]
    task_path = res.get_task_path(task)
    c = 0.4
#    _plot_average_turbulent_profile(axes[0], summary.edge_states_info.task_for_uncontrolled_case, task_path,
#                                    color='black')
    ti = ti_builder.get_timeintegration(os.path.join(task_path, 'initial_conditions', 'data-{}'.format(c)))
    ke = 0.5*ti.L2U**2
    axes[0].plot(ti.T, ke)
    axes[1].plot([1., 1.], [-1., 1.], 'k--')  # laminar solution
    selected_times = [{
        'T': T,
        'T_i': np_index(ti.T, T),
        'color': color,
    } for T, color in zip((1500, 1600, 1700, 1800, 1900), ('red', 'blue', 'green', 'magenta', 'cyan'))]
    for st in selected_times:
        axes[0].plot([ti.T[st['T_i']]], [ke[st['T_i']]], 'o', color=st['color'])
        solution_at_T = ti.solution(st['T'])
        av_turb_u = average(average(solution_at_T, ['u'], 'x'), ['u'], 'z').u
        av_lam_u = solution_at_T.space.y
        dUdy = fd(solution_at_T.space.y, av_turb_u + av_lam_u)
        axes[1].plot(dUdy, solution_at_T.space.y, label=f"T = {st['T']}", color=st['color'])
    axes[0].set_xlabel(r'$t$')
    axes[0].set_ylabel(r'$||\boldsymbol{u}||^2 / 2$')
    axes[1].set_xlabel(r'$\mathrm{d}U/\mathrm{d}y$')
    axes[1].set_ylabel(r'$y$')
    for ax in axes:
        ax.grid()
        ax.legend()
    label_axes(axes[0], label='(a)', loc=(0.5, 1.05), fontsize=16)
    plt.tight_layout()
    plt.savefig('instantaneous_profiles.eps')
    plt.show()
