import os
import sys
from typing import List, Tuple, Any
from functools import reduce
sys.path.append(os.getcwd())

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from restools.timeintegration_builders import get_ti_builder
from restools.plotting import label_axes, build_zooming_axes_for_plotting_with_box, rasterise_and_save, reduce_eps_size
from papers.jfm2020_probabilistic_protocol.data import Summary, SingleConfiguration
from papers.jfm2020_probabilistic_protocol.extensions import LaminarisationProbabilityFittingFunction2020JFM, plot_p_lam, plot_p_lam_from_conf
from comsdk.misc import load_from_json
from comsdk.research import Research


def _plot_traj(ax, res, task, data_dir, color):
    ti = ti_builder.get_timeintegration(os.path.join(res.get_task_path(task), data_dir))
    t_max = 2000
    Ulam = ti.L2Ulam[0]
    Bs = ti.UlamDotU[:t_max*2] / Ulam**2
    As = np.sqrt(ti.L2U[:t_max*2]**2 - Bs**2 * Ulam**2)
    ax.plot(As, Bs*Ulam, color=color, linewidth=2)


def _plot_ics(ax, conf: SingleConfiguration, res: Research, energy_levels_number: int, laminar_flow_ke: float,
              obj_to_rasterize: List[Any]) -> Tuple[np.ndarray, np.ndarray]:
    lam_rps = []
    turb_rps = []
    for e_i in range(energy_levels_number):
        for rp_info in conf.rps_info[e_i]:
            A_ = rp_info.A
            B_mod = rp_info.B * np.sqrt(2.*laminar_flow_ke)
            if rp_info.is_laminarised:
                lam_rps.append((A_, B_mod))
            else:
                turb_rps.append((A_, B_mod))
    lam_rps = np.array(lam_rps)
    turb_rps = np.array(turb_rps)
    for rps, color in zip((lam_rps, turb_rps), ('black', 'orange')):
        # DENSE CASE: markersize=1.6
        # NOT VERY DENSE: markersize=2.5
        lines = ax.plot(rps[:, 0], rps[:, 1], 'o', color=color, markersize=2.5)
        obj_to_rasterize.append(lines[0])
    _plot_traj(ax,
               res=res,
               task=conf.turb_trajectory.task,
               data_dir=conf.turb_trajectory.data_dir,
               color='tomato')
    return lam_rps, turb_rps


def _plot_rps_in_box(box_ax, rps, color, parent_box):
    A_vals = rps[:, 0]
    B_vals = rps[:, 1]
    A_in_box_cond = np.logical_and(parent_box[0] <= A_vals, A_vals <= parent_box[0] + parent_box[2])
    B_in_box_cond = np.logical_and(parent_box[1] <= B_vals, B_vals <= parent_box[1] + parent_box[3])
    filtered_rps = np.compress(np.logical_and(A_in_box_cond, B_in_box_cond), rps, axis=0)
    lines = box_ax.plot(filtered_rps[:, 0], filtered_rps[:, 1], 'o', color=color, markersize=3)
    return lines


if __name__ == '__main__':
    plt.style.use('resources/default.mplstyle')

    summary = load_from_json(Summary)
    ti_builder = get_ti_builder()
    res = {conf.res_id: Research.open(conf.res_id) for conf in summary.confs}

    # PLOT COLOURED INITIAL CONDITIONS FOR UNCONTROLLED CASE

    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    for ax in axes:
        ax.set_xlim((0., 0.41))
        ax.set_ylim((-0.35, 0.305))
        ax.set_xlabel(r'$A$', fontsize=16)
        ax.grid()
    axes[0].set_ylabel(r'$||\boldsymbol{U_{lam}}|| B$', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    obj_to_rasterize = []
    for ax, conf, label in zip(axes, summary.confs, (r'(a) $Re= 400$', r'(b)  $Re= 500$', r'(c)  $Re= 700$')):
        lam_rps, turb_rps = _plot_ics(ax, conf, res[conf.res_id], len(summary.energy_levels), summary.laminar_flow_ke,
                                      obj_to_rasterize)
        print(len(lam_rps) + len(turb_rps))
        if ax is axes[0]:
            parent_box = (0.217, 0.055, 0.025, 0.03)
            zoom_ax = build_zooming_axes_for_plotting_with_box(fig, ax,
                                                               parent_box=parent_box,
                                                               child_box=(0.283, 0.084, 0.115, 0.12),
                                                               parent_vertices=(3, 2),
                                                               child_vertices=(0, 1),
                                                               remove_axis=True)

            for rps, color in zip((lam_rps, turb_rps), ('black', 'orange')):
                lines = _plot_rps_in_box(zoom_ax, rps, color, parent_box)
                obj_to_rasterize.append(lines[0])
        label_axes(ax, label=label, loc=(0.33, 1.04), fontsize=20)
    #fname = 'ics_uniform_B.eps'
    #rasterise_and_save(fname, rasterise_list=obj_to_rasterize, fig=fig, dpi=300)
    #reduce_eps_size(fname)
    fname = 'ics_uniform_B.png'
    plt.savefig(fname, dpi=300)
    plt.show()

    # PLOT GRAPHICAL ABSTRACT

    fig = plt.figure(figsize=(6, 5), frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    obj_to_rasterize = []
    lam_rps, turb_rps = _plot_ics(ax, summary.confs[1], res[summary.confs[1].res_id], len(summary.energy_levels),
                                  summary.laminar_flow_ke, obj_to_rasterize)
    fname = 'graphical_abstract.png'
    plt.savefig(fname, dpi=300)
    plt.show()

    # PLOT P_LAM FOR UNCONTROLLED SYSTEM

    fig, axes = plt.subplots(3, 1, figsize=(12, 11))
    edge_energy_no_osc_at_Res = [0.01958, 0.0182, 0.0166]
    edge_energy_osc = 0.0115
    fittings = []
    for ax, conf, label in zip(axes, summary.confs, (r'(a) $Re= 400$', r'(b)  $Re= 500$', r'(c)  $Re= 700$')):
        fitting = LaminarisationProbabilityFittingFunction2020JFM.from_data(0.5 * np.array([0.] + summary.energy_levels),
                                                                            np.array([1.] + conf.p_lam))
        plot_p_lam_from_conf(ax, summary, conf)
        ax.plot([conf.edge_state_energy_mean, conf.edge_state_energy_mean], [0.0, 1.0], '--',
                linewidth=2,
                color='black',
                label=r'$E_{edge}$')
        fittings.append(fitting)
        ax.set_ylabel(r'$P_{lam}$', fontsize=16)
        ax.legend(loc='upper right', fontsize=16)
        ax.grid()
        label_axes(ax, label=label, loc=(0.45, 1.05), fontsize=16)
    Es = np.linspace(0., ax.get_xlim()[1], 200)
    for ax in axes:
        for fitting, color in zip(fittings, ('green', 'cyan', 'brown')):
            ax.plot(Es, fitting(Es), linewidth=2, color=color)  # red, green, yellowgreen, lime, brown are OK
    axes[-1].set_xlabel(r'$\frac{1}{2}||\boldsymbol{u}||^2$', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.96, hspace=0.3)
    fname = 'p_lam.eps'
    plt.savefig(fname)
    reduce_eps_size(fname)
    plt.show()

    # PLOT COLOURED INITIAL CONDITIONS FOR CONTROLLED CASE

    fig, ax = plt.subplots(1, 1, figsize=(5, 6))
    ax.set_xlim((0., 0.41))
    ax.set_ylim((-0.35, 0.305))
    ax.set_xlabel(r'$A$', fontsize=16)
    ax.grid()
    ax.set_ylabel(r'$||\boldsymbol{U_{lam}}|| B$', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    obj_to_rasterize = []
    lam_rps, turb_rps = _plot_ics(ax, summary.confs[-1], res[summary.confs[-1].res_id], len(summary.energy_levels),
                                  summary.laminar_flow_ke, obj_to_rasterize)
    parent_box=(0.217, 0.055, 0.025, 0.03)
    zoom_ax = build_zooming_axes_for_plotting_with_box(fig, ax,
                                                       parent_box=parent_box,
                                                       child_box=(0.283, 0.084, 0.115, 0.12),
                                                       parent_vertices=(3, 2),
                                                       child_vertices=(0, 1),
                                                       remove_axis=True)

    for rps, color in zip((lam_rps, turb_rps), ('black', 'orange')):
        lines = _plot_rps_in_box(zoom_ax, rps, color, parent_box)
        obj_to_rasterize.append(lines[0])
    #fname = 'ics_uniform_B_osc.eps'
    #rasterise_and_save(fname, rasterise_list=obj_to_rasterize, fig=fig, dpi=300)
    #reduce_eps_size(fname)
    fname = 'ics_uniform_B_osc.png'
    plt.savefig(fname, dpi=300)
    plt.show()

    # PLOT P_LAM FOR CONTROLLED SYSTEM

    conf_unctrl = summary.confs[1]
    conf_ctrl = summary.confs[3]
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    plot_p_lam_from_conf(ax, summary, conf_ctrl)
    fitting_ctrl = LaminarisationProbabilityFittingFunction2020JFM.from_data(0.5 * np.array([0.] + summary.energy_levels),
                                                                             np.array([1.] + conf_ctrl.p_lam))
    ax.plot([conf_unctrl.edge_state_energy_mean, conf_unctrl.edge_state_energy_mean], [0.0, 1.0], '--',
            linewidth=2,
            color='black',
            label=r'$E_{edge}$')
    ax.plot([conf_ctrl.edge_state_energy_mean, conf_ctrl.edge_state_energy_mean], [0.0, 1.0], '-',
            linewidth=2,
            color='black',
            label=r'$E_{edge}^{(osc)}$')
    ax.fill_between([conf_ctrl.edge_state_energy_mean - conf_ctrl.edge_state_energy_std,
                     conf_ctrl.edge_state_energy_mean + conf_ctrl.edge_state_energy_std],
                    [0.0, 0.0], [1.0, 1.0],
                    color='lightgray')
    ax.set_ylabel(r'$P_{lam}$', fontsize=16)
    ax.legend(loc='upper right', fontsize=16)
    ax.grid()
    Es = np.linspace(0., ax.get_xlim()[1], 200)
    ax.plot(Es, fittings[1](Es), linewidth=2, color='cyan')  # fitting for unctrl at Re = 500
    ax.plot(Es, fittings[0](Es), linewidth=2, color='green')  # fitting for unctrl at Re = 400
    ax.plot(Es, fitting_ctrl(Es), linewidth=2, color='red')
    ax.set_xlabel(r'$\frac{1}{2}||\boldsymbol{u}||^2$', fontsize=16)
    plt.tight_layout()
    fname = 'p_lam_osc_with_fitting.eps'
    plt.savefig(fname)
    reduce_eps_size(fname)
    plt.show()

    # PLOT P_LAM FOR RE = 500 FOR POSTER

    obj_to_rasterize = []
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    edge_energy_no_osc_at_Res = [0.01958, 0.0182, 0.0166]
    edge_energy_osc = 0.0115
    fittings = []
    conf = summary.confs[1]
    plot_p_lam_from_conf(ax, summary, conf, bar_alpha=0.4, obj_to_rasterize=obj_to_rasterize)
#    label_axes(ax, label=label, loc=(0.45, 1.05), fontsize=16)
    Es = np.linspace(0., ax.get_xlim()[1], 200)
    fittings = [LaminarisationProbabilityFittingFunction2020JFM.from_data(0.5 * np.array([0.] + summary.energy_levels),
                                                                        np.array([1.] + conf.p_lam)) for conf in summary.confs]
    for conf, color in zip(summary.confs, ('green', 'cyan', 'brown')):
        fitting  = LaminarisationProbabilityFittingFunction2020JFM.from_data(0.5 * np.array([0.] + summary.energy_levels),
                                                                    np.array([1.] + conf.p_lam))
        ax.plot(Es, fitting(Es), linewidth=2, color=color)  # red, green, yellowgreen, lime, brown are OK
        ax.plot([conf.edge_state_energy_mean, conf.edge_state_energy_mean], [0.0, 1.0], '--',
                linewidth=2,
                color=color)
                #label=r'$E_{edge}$')
    ax.set_xlabel(r'$E$', fontsize=16)
    ax.set_ylabel(r'$P_{lam}$', fontsize=16)
    ax.legend(loc='upper right', fontsize=16)
    ax.grid()
    plt.tight_layout()
    #fname = 'p_lam_re_500_poster.eps'
    #rasterise_and_save(fname, rasterise_list=obj_to_rasterize, fig=fig, dpi=300)
    #reduce_eps_size(fname)
    fname = 'p_lam_re_500_poster.png'
    plt.savefig(fname, dpi=300)
    plt.show()

    # PLOT P_LAM FOR UNCONTROLLED SYSTEM FOR PRESENTATION

    fig, axes = plt.subplots(3, 1, figsize=(12, 7))
    edge_energy_no_osc_at_Res = [0.01958, 0.0182, 0.0166]
    edge_energy_osc = 0.0115
    fittings = []
    for ax, conf, label in zip(axes, summary.confs, (r'(a) $Re= 400$', r'(b)  $Re= 500$', r'(c)  $Re= 700$')):
        fitting = LaminarisationProbabilityFittingFunction2020JFM.from_data(0.5 * np.array([0.] + summary.energy_levels),
                                                                            np.array([1.] + conf.p_lam))
        plot_p_lam_from_conf(ax, summary, conf)
        ax.plot([conf.edge_state_energy_mean, conf.edge_state_energy_mean], [0.0, 1.0], '--',
                linewidth=2,
                color='black',
                label=r'$E_{edge}$')
        fittings.append(fitting)
        ax.set_ylabel(r'$P_{lam}$', fontsize=16)
        ax.legend(loc='upper right', fontsize=16)
        ax.grid()
        label_axes(ax, label=label, loc=(0.45, 1.05), fontsize=16)
    Es = np.linspace(0., ax.get_xlim()[1], 200)
    for ax in axes:
        for fitting, color in zip(fittings, ('green', 'cyan', 'brown')):
            ax.plot(Es, fitting(Es), linewidth=2, color=color)  # red, green, yellowgreen, lime, brown are OK
    axes[-1].set_xlabel(r'$\frac{1}{2}||\boldsymbol{u}||^2$', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.96, hspace=0.3)
    fname = 'p_lam_pres.eps'
    plt.savefig(fname)
    reduce_eps_size(fname)
    plt.show()

    # PLOT P_LAM FOR CONTROLLED SYSTEM FOR POSTER

    obj_to_rasterize = []
    conf_unctrl = summary.confs[1]
    conf_ctrl = summary.confs[3]
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    plot_p_lam_from_conf(ax, summary, conf_ctrl, bar_alpha=0.4, obj_to_rasterize=obj_to_rasterize)
    fitting_ctrl = LaminarisationProbabilityFittingFunction2020JFM.from_data(0.5 * np.array([0.] + summary.energy_levels),
                                                                             np.array([1.] + conf_ctrl.p_lam))
    ax.plot([summary.confs[0].edge_state_energy_mean, summary.confs[0].edge_state_energy_mean], [0.0, 1.0], '--',
            linewidth=2,
            color='green')
    ax.plot([conf_unctrl.edge_state_energy_mean, conf_unctrl.edge_state_energy_mean], [0.0, 1.0], '--',
            linewidth=2,
            color='cyan')
    ax.plot([conf_ctrl.edge_state_energy_mean, conf_ctrl.edge_state_energy_mean], [0.0, 1.0], '--',
            linewidth=2,
            color='black')
    ax.fill_between([conf_ctrl.edge_state_energy_mean - conf_ctrl.edge_state_energy_std,
                     conf_ctrl.edge_state_energy_mean + conf_ctrl.edge_state_energy_std],
                    [0.0, 0.0], [1.0, 1.0],
                    color='lightgray', zorder=-2000)
    ax.set_ylabel(r'$P_{lam}$', fontsize=16)
    ax.legend(loc='upper right', fontsize=16)
    ax.grid()
    Es = np.linspace(0., ax.get_xlim()[1], 200)
#    ax.plot(Es, fittings[2](Es), linewidth=2, color='brown')  # fitting for unctrl at Re = 700
    ax.plot(Es, fittings[1](Es), linewidth=2, color='cyan')  # fitting for unctrl at Re = 500
    ax.plot(Es, fittings[0](Es), linewidth=2, color='green')  # fitting for unctrl at Re = 400
    ax.plot(Es, fitting_ctrl(Es), linewidth=2, color='black')
    ax.set_xlabel(r'$E$', fontsize=16)
    plt.tight_layout()
    #fname = 'p_lam_osc_with_fitting_poster.eps'
    #rasterise_and_save(fname, rasterise_list=obj_to_rasterize, fig=fig, dpi=300)
    #reduce_eps_size(fname)
    fname = 'p_lam_osc_with_fitting_poster.png'
    plt.savefig(fname, dpi=300)
    plt.show()
