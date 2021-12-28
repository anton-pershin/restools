import os
import sys
import argparse
import pickle
sys.path.append(os.getcwd())

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from restools.timeintegration_builders import get_ti_builder
from restools.plotting import label_axes
from restools.laminarisation_probability import LaminarisationStudy, LaminarisationProbabilityEstimation
from papers.jfm2020_nonlinear_robustness.extensions import DistributionSummary, exponential_noise_distribution
from papers.jfm2020_probabilistic_protocol.extensions import RandomPerturbationFilenameJFM2020, \
    DataDirectoryJFM2020AProbabilisticProtocol, LaminarisationProbabilityFittingFunction2020JFM
from thequickmath.stats import EmpiricalDistribution
from thequickmath.aux import np_index
from comsdk.comaux import load_from_json, dump_to_json
from comsdk.research import Research

#from ke_tools import *
#from relam_tools import build_relam

#def _dump_turbulent_fraction_by_re(task, Res):
#    min_tf = np.zeros((len(Res),))
#    domain_filling_time = np.zeros((len(Res),))
#    turb_fraction_by_re = {}
#    for i, Re in enumerate(Res):
#        print('Re = {}'.format(Re))
#        if isinstance(task, int):
#            ke_field = get_xy_averaged_ke(os.path.join(res.get_task_path(task), 'data-{}'.format(Re)))
#        else:
#            ke_field = get_xy_averaged_ke(os.path.join(task, 'data-{}'.format(Re)))
#        #outer_spacing, inner_spacings, lam_patch_number, turb_patch = laminar_patches_data(ke_field)
#        _, _, _, turb_patch = laminar_patches_data(ke_field)
#        turb_fraction = turb_patch / ke_field.space.z[-1]
#        turb_fraction_by_re[Re] = turb_fraction
#        #max_tf = np.max(turb_fraction)
#        #min_tf[i] = np.min(turb_fraction)
#        #if max_tf == 1.:
#        #    domain_filling_time[i] = ke_field.space.t[np_index(turb_fraction, 1.)]
#    if isinstance(task, int):
#        f = open(os.path.join(res.get_task_path(task), obj_name + '.pyo'),'w')
#    else:
#        f = open(os.path.join(task, obj_name + '.pyo'),'w')
#    pickle.dump(turb_fraction_by_re, f)
#    f.close()
#
#def _dump_turbulent_fraction_by_tasks(tasks, Res):
#    min_tf = np.zeros((len(Res),))
#    domain_filling_time = np.zeros((len(Res),))
#    turb_fraction_by_re = {}
#    for i in range(len(Res)):
#        Re = Res[i]
#        task = tasks[i]
#        print('Re = {}'.format(Re))
#        if isinstance(task, int):
#            ke_field = get_xy_averaged_ke(os.path.join(res.get_task_path(task), 'data'))
#        else:
#            ke_field = get_xy_averaged_ke(os.path.join(task, 'data'))
#        #outer_spacing, inner_spacings, lam_patch_number, turb_patch = laminar_patches_data(ke_field)
#        _, _, _, turb_patch = laminar_patches_data(ke_field)
#        turb_fraction = turb_patch / ke_field.space.z[-1]
#        turb_fraction_by_re[Re] = turb_fraction
#        #max_tf = np.max(turb_fraction)
#        #min_tf[i] = np.min(turb_fraction)
#        #if max_tf == 1.:
#        #    domain_filling_time[i] = ke_field.space.t[np_index(turb_fraction, 1.)]
#    with open(os.path.join(res.research_path, obj_name + '.pyo'),'w') as f:
#        pickle.dump(turb_fraction_by_re, f)
#
#def _plot_laminar_patches_data_at_re(res, task, Re):
#    print('Re = {}'.format(Re))
#    if isinstance(task, int):
#        ke_field = get_xy_averaged_ke(os.path.join(res.get_task_path(task), 'data-{}'.format(Re)))
#    else:
#        ke_field = get_xy_averaged_ke(os.path.join(task, 'data-{}'.format(Re)))
#    _, _, lam_patch_number, turb_patch = laminar_patches_data(ke_field, ke_lam_threshold=3*1e-2, min_lam_patch_width=7.)
#    turb_fraction = turb_patch / ke_field.space.z[-1]
#    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
#    _plot_contours(ke_field, axes[0])
#    axes[1].plot(ke_field.space.t, turb_fraction, linewidth=2)
#    axes[2].plot(ke_field.space.t, lam_patch_number, linewidth=2)
#    for ax in axes[1:]:
#        ax.grid()
#    axes[-1].set_xlabel(r'$t$', fontsize=16)
#    axes[0].set_ylabel(r'$z$', fontsize=16)
#    axes[1].set_ylabel(r'$F_{turb}$', fontsize=16)
#    axes[2].set_ylabel(r'$N_{spots}$', fontsize=16)
#    plt.tight_layout()
#    plt.show()
#
#def _plot_turbulent_fraction_data_from_dump(task):
#    if isinstance(task, int):
#        f = open(os.path.join(res.get_task_path(task), obj_name + '.pyo'),'r')
#    else:
#        f = open(os.path.join(task, obj_name + '.pyo'),'r')
#    turb_fraction_by_re = pickle.load(f)
#    f.close()
#    Res = sorted(turb_fraction_by_re.keys())
#    min_tf = np.zeros((len(Res),))
#    tf_growth_rate = np.zeros((len(Res),))
#    domain_filling_time = np.zeros((len(Res),))
#    plt.show()
#    for i, Re in enumerate(Res):
#        turb_fraction = turb_fraction_by_re[Re]
#        max_tf = np.max(turb_fraction)
#        min_tf[i] = np.min(turb_fraction)
#        if max_tf == 1.:
#            domain_filling_i = np_index(turb_fraction, 1.)
#            domain_filling_time[i] = domain_filling_i / 2.
#            tf_growth_rate[i] = (turb_fraction[domain_filling_i] - turb_fraction[0]) / domain_filling_time[i]
#        else:
#            tf_growth_rate[i] = (turb_fraction[-1] - turb_fraction[0]) / (len(turb_fraction) * 2)
#    fig, axes = plt.subplots(2, 1, sharex=True)
#    #axes[0].plot(Res, min_tf, 'o')
#    axes[0].plot(Res, tf_growth_rate, 'o')
#    axes[1].plot(Res, domain_filling_time, 'o')
#    axes[1].set_xlabel(r'$Re$')
#    axes[0].set_ylabel(r'$min(T_f)$')
#    axes[1].set_ylabel(r'$t_{fill}$')
#    axes[0].grid()
#    axes[1].grid()
#    plt.show()

def _plot_turbulent_fraction_growth_rate_from_dump(ax, res, task, dump_obj_name, color=None, label=None):
    if isinstance(task, int):
        with open(os.path.join(res.get_task_path(task), f'{dump_obj_name}.pyo'), 'rb') as f:
            turb_fraction_by_re = pickle.load(f, encoding='latin1')
    else:
        print(os.path.join(task, f'{dump_obj_name}.pyo'))
        with open(os.path.join(task, f'{dump_obj_name}.pyo'), 'rb') as f:
            turb_fraction_by_re = pickle.load(f, encoding='latin1')
    res = sorted(turb_fraction_by_re.keys())
    tf_growth_rate = np.zeros((len(res),))
    for i, re in enumerate(res):
        turb_fraction = turb_fraction_by_re[re]
        t_begin_i = 500*2
        t_max_i = len(turb_fraction) - 1
        if t_max_i < t_begin_i*2:
            t_begin_i = 0
        max_tf = np.max(turb_fraction)
        if max_tf == 1.:
            domain_filling_i = np_index(turb_fraction, 1.)
            domain_filling_time = domain_filling_i / 2.
            tf_growth_rate[i] = (turb_fraction[domain_filling_i] - turb_fraction[t_begin_i]) / domain_filling_time
        else:
            tf_growth_rate[i] = (turb_fraction[t_max_i] - turb_fraction[t_begin_i]) / (len(turb_fraction) * 2)
    tf_growth_rate *= 32.*np.pi  # to turn turbulent fraction to front speed with respect to the domain size
    ax.plot(res, tf_growth_rate, 'o-', linewidth=2, color=color, label=label)
    ax.set_xlabel(r'$Re$', fontsize=16)
    ax.set_ylabel(r'$\langle c \rangle$', fontsize=16)
#    ax.set_xticks(Res)
    ax.grid()
    return res, tf_growth_rate

#def _plot_domain_filling_time_from_dump(ax, task):
#    if isinstance(task, int):
#        f = open(os.path.join(res.get_task_path(task), obj_name + '.pyo'),'r')
#    else:
#        f = open(os.path.join(task, obj_name + '.pyo'),'r')
#    turb_fraction_by_re = pickle.load(f)
#    f.close()
#    Res = np.array(sorted(turb_fraction_by_re.keys()))
#    min_tf = np.zeros((len(Res),))
#    domain_filling_time = np.zeros((len(Res),))
#    for i, Re in enumerate(Res):
#        turb_fraction = turb_fraction_by_re[Re]
#        max_tf = np.max(turb_fraction)
#        min_tf[i] = np.min(turb_fraction)
#        if max_tf == 1.:
#            domain_filling_time[i] = np_index(turb_fraction, 1.) / 2.
#    cond = domain_filling_time > 0.
#    p = np.polyfit(Res[cond], domain_filling_time[cond], 1)
#    print(p)
#    ax.plot(Res, p[0]*Res + p[1], '-', color='lightblue', linewidth=2)
#    ax.plot(Res[cond], domain_filling_time[cond], 'o')
#    #axes[1].set_xlabel(r'$Re$')
#    #axes[0].set_ylabel(r'$min(T_f)$')
#    #axes[1].set_ylabel(r'$t_{fill}$')
#    ax.set_ylim((0, 8000))
#    ax.grid()
#    #axes[1].grid()
#
#def _plot_min_turbulent_fraction_from_dump(ax, task):
#    if isinstance(task, int):
#        f = open(os.path.join(res.get_task_path(task), obj_name + '.pyo'),'r')
#    else:
#        f = open(os.path.join(task, obj_name + '.pyo'),'r')
#    turb_fraction_by_re = pickle.load(f)
#    f.close()
#    Res = np.array(sorted(turb_fraction_by_re.keys()))
#    min_tf = np.zeros((len(Res),))
#    domain_filling_time = np.zeros((len(Res),))
#    for i, Re in enumerate(Res):
#        turb_fraction = turb_fraction_by_re[Re]
#        max_tf = np.max(turb_fraction)
#        min_tf[i] = np.min(turb_fraction)
#        if max_tf == 1.:
#            domain_filling_time[i] = np_index(turb_fraction, 1.) / 2.
#    cond = domain_filling_time > 0
#    p = np.polyfit(Res[cond], min_tf[cond], 1)
#    ax.plot(Res, p[0]*Res + p[1], '-', color='lightblue', linewidth=2)
#    ax.plot(Res[cond], min_tf[cond], 'o')
#    #axes[1].set_xlabel(r'$Re$')
#    #axes[0].set_ylabel(r'$min(T_f)$')
#    #axes[1].set_ylabel(r'$t_{fill}$')
#    ax.set_ylim((0.1, 0.55))
#    ax.grid()
#    #axes[1].grid()
#
#def _plot_turbulent_fraction_at_re_from_dump(task, Re):
#    if isinstance(task, int):
#        f = open(os.path.join(res.get_task_path(task), obj_name + '.pyo'),'r')
#        ke_field = get_xy_averaged_ke(os.path.join(res.get_task_path(task), 'data-{}'.format(Re)))
#    else:
#        f = open(os.path.join(task, obj_name + '.pyo'),'r')
#        ke_field = get_xy_averaged_ke(os.path.join(task, 'data-{}'.format(Re)))
#
#    turb_fraction_by_re = pickle.load(f)
#    f.close()
#    turb_fraction = turb_fraction_by_re[Re]
#    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
#    _plot_contours(ke_field, axes[0])
#    axes[1].plot(ke_field.space.t, turb_fraction)
#    axes[1].set_xlabel(r'$t$')
#    axes[0].set_ylabel(r'$z$')
#    axes[1].set_ylabel(r'$T_f$')
#    axes[1].grid()
#    plt.show()
#
#def _get_turbulent_fraction_growth_rate_at_Re_from_dump(res, task, Re):
#    if isinstance(task, int):
#        f = open(os.path.join(res.get_task_path(task), obj_name + '.pyo'),'r')
#    else:
#        f = open(os.path.join(task, obj_name + '.pyo'),'r')
#    turb_fraction_by_re = pickle.load(f)
#    f.close()
#    turb_fraction = turb_fraction_by_re[Re]
#    t_begin_i = 500*2
#    t_max_i = len(turb_fraction) - 1
#    if t_max_i < t_begin_i*2:
#        t_begin_i = 0
#    max_tf = np.max(turb_fraction)
#    if max_tf == 1.:
#        domain_filling_i = np_index(turb_fraction, 1.)
#        domain_filling_time = domain_filling_i / 2.
#        tf_growth_rate = (turb_fraction[domain_filling_i] - turb_fraction[t_begin_i]) / domain_filling_time
#    else:
#        tf_growth_rate = (turb_fraction[t_max_i] - turb_fraction[t_begin_i]) / (len(turb_fraction) * 2)
#    return tf_growth_rate


if __name__ == '__main__':
    plt.style.use('resources/default.mplstyle')
    parser = argparse.ArgumentParser(description='Plots front speed')
    parser.add_argument('mode', metavar='MODE', nargs='?', choices=['loadfromdump', 'dump'], default='loadfromdump',
                        help='decides where the data should be taken from and whether the data is dumped '
                             '(must be either loadfromdump or dump)')
    args = parser.parse_args()
    #summary = load_from_json(Summary)
    #summary_prob_proto = load_from_json(SummaryProbProto)
    ti_builder = get_ti_builder()
    res_no_ctrl = Research.open('EQ')
    res_ctrl = Research.open('WIDE_SIMS_IN_PHASE')

    tasks = [
        [res_ctrl.get_task_path(15)],  # A = 0.1, omega = 1/16
        [res_ctrl.get_task_path(6)],  # A = 0.2, omega = 1/16
        [res_ctrl.get_task_path(4)],  # A = 0.3, omega = 1/16
        [res_ctrl.get_task_path(5)],  # A = 0.4, omega = 1/16
    ]
    labels = [
        r'$A = 0.1, \omega = 1/16$',
        r'$A = 0.2, \omega = 1/16$',
        r'$A = 0.3, \omega = 1/16$',
        r'$A = 0.4, \omega = 1/16$',
    ]

    #tasks_S5 = [[res.get_task_path(422)]]
    #tasks_S13 = [[res.get_task_path(430)]]
    #tasks_S5_A_01_inphase = [
    #    [res_osc.get_task_path(152)], # A = 0.1, omega = 1/2, in-phase
    #    [res_osc.get_task_path(153)], # A = 0.1, omega = 1/4, in-phase
    #    [res_osc.get_task_path(154)], # A = 0.1, omega = 1/8, in-phase
    #    [res_osc.get_task_path(86), res_osc.get_task_path(87), res_osc.get_task_path(88)], # A = 0.1, omega = 1/16, in-phase
    #    [res_osc.get_task_path(156)], # A = 0.1, omega = 1/32, in-phase
    #    [res_osc.get_task_path(157)], # A = 0.1, omega = 1/64, in-phase
    #    [res_osc.get_task_path(158)], # A = 0.1, omega = 1/128, in-phase
    #]
    #
    #tasks_S5_A_02_inphase = [
    #    [res_osc.get_task_path(159)], # A = 0.2, omega = 1/2, in-phase
    #    [res_osc.get_task_path(160)], # A = 0.2, omega = 1/4, in-phase
    #    [res_osc.get_task_path(161)], # A = 0.2, omega = 1/8, in-phase
    #    [res_osc.get_task_path(162)], # A = 0.2, omega = 1/16, in-phase
    #    [res_osc.get_task_path(163)], # A = 0.2, omega = 1/32, in-phase
    #    [res_osc.get_task_path(164)], # A = 0.2, omega = 1/64, in-phase
    #    [res_osc.get_task_path(165)], # A = 0.2, omega = 1/128, in-phase
    #]
    #
    #tasks_S5_A_03_inphase = [
    #    [res_osc.get_task_path(166)], # A = 0.3, omega = 1/2, in-phase
    #    [res_osc.get_task_path(167)], # A = 0.3, omega = 1/4, in-phase
    #    [res_osc.get_task_path(168)], # A = 0.3, omega = 1/8, in-phase
    #    [res_osc.get_task_path(169)], # A = 0.3, omega = 1/16, in-phase
    #    [res_osc.get_task_path(170)], # A = 0.3, omega = 1/32, in-phase
    #    [res_osc.get_task_path(171)], # A = 0.3, omega = 1/64, in-phase
    #    [res_osc.get_task_path(172)], # A = 0.3, omega = 1/128, in-phase
    #]
    #
    #tasks_S5_A_04_inphase = [
    #    [res_osc.get_task_path(173)], # A = 0.4, omega = 1/2, in-phase
    #    [res_osc.get_task_path(174)], # A = 0.4, omega = 1/4, in-phase
    #    [res_osc.get_task_path(175)], # A = 0.4, omega = 1/8, in-phase
    #    [res_osc.get_task_path(176)], # A = 0.4, omega = 1/16, in-phase
    #    [res_osc.get_task_path(177)], # A = 0.4, omega = 1/32, in-phase
    #    [res_osc.get_task_path(178)], # A = 0.4, omega = 1/64, in-phase
    #    [res_osc.get_task_path(179)], # A = 0.4, omega = 1/128, in-phase
    #]
    #
    #tasks_S5_A_05_inphase = [
    #    [res_osc.get_task_path(180)], # A = 0.5, omega = 1/2, in-phase
    #    [res_osc.get_task_path(181)], # A = 0.5, omega = 1/4, in-phase
    #    [res_osc.get_task_path(182)], # A = 0.5, omega = 1/8, in-phase
    #    [res_osc.get_task_path(183)], # A = 0.5, omega = 1/16, in-phase
    #    [res_osc.get_task_path(184)], # A = 0.5, omega = 1/32, in-phase
    #    [res_osc.get_task_path(186)], # A = 0.5, omega = 1/64, in-phase
    #    [res_osc.get_task_path(187)], # A = 0.5, omega = 1/64, in-phase
    #]
    #
    #tasks_S13_A_01_inphase = [
    #    [res_osc.get_task_path(188)], # A = 0.1, omega = 1/2, in-phase
    #    [res_osc.get_task_path(189)], # A = 0.1, omega = 1/4, in-phase
    #    [res_osc.get_task_path(190)], # A = 0.1, omega = 1/8, in-phase
    #    [res_osc.get_task_path(89), res_osc.get_task_path(90), res_osc.get_task_path(91)], # A = 0.1, omega = 1/16, in-phase
    #    [res_osc.get_task_path(191)], # A = 0.1, omega = 1/32, in-phase
    #    [res_osc.get_task_path(192)], # A = 0.1, omega = 1/64, in-phase
    #    [res_osc.get_task_path(193)], # A = 0.1, omega = 1/128, in-phase
    #]
    #
    #tasks_S13_A_02_inphase = [
    #    [res_osc.get_task_path(194)], # A = 0.2, omega = 1/2, in-phase
    #    [res_osc.get_task_path(195)], # A = 0.2, omega = 1/4, in-phase
    #    [res_osc.get_task_path(196)], # A = 0.2, omega = 1/8, in-phase
    #    [res_osc.get_task_path(197)], # A = 0.2, omega = 1/16, in-phase
    #    [res_osc.get_task_path(198)], # A = 0.2, omega = 1/32, in-phase
    #]
    #tasks_S5_A_005_inphase = [
    #    res_osc.get_task_path(47), # A = 0.05, omega = 1/16, in-phase
    #    res_osc.get_task_path(54), # A = 0.05, omega = 1/16, in-phase
    #    res_osc.get_task_path(57), # A = 0.05, omega = 1/16, in-phase
    #]
    #tasks_S5_A_01_antiphase = [
    #    res_osc_antiphase.get_task_path(5), # A = 0.1, omega = 1/16, antiphase
    #    res_osc_antiphase.get_task_path(6), # A = 0.1, omega = 1/16, antiphase
    #    res_osc_antiphase.get_task_path(7), # A = 0.1, omega = 1/16, antiphase
    #    res_osc_antiphase.get_task_path(8), # A = 0.1, omega = 1/16, antiphase
    #]
    #tasks_S5_A_001_antiphase = [
    #    res_osc_antiphase.get_task_path(11), # A = 0.01, omega = 1/16, antiphase
    #    res_osc_antiphase.get_task_path(12), # A = 0.01, omega = 1/16, antiphase
    #]
    #tasks_S5_A_001_inphase = [
    #    res_osc.get_task_path(25), # A = 0.01, omega = 1/16
    #    res_osc.get_task_path(28), # A = 0.01, omega = 1/16
    #    res_osc.get_task_path(29), # A = 0.01, omega = 1/16
    #]
    #tasks_S13_A_01_inphase = [
    #    res_osc.get_task_path(89), # A = 0.1, omega = 1/16
    #    res_osc.get_task_path(90), # A = 0.1, omega = 1/16
    #    res_osc.get_task_path(91), # A = 0.1, omega = 1/16
    #]
    #tasks_S13_A_001_inphase = [
    #    res_osc.get_task_path(39), # A = 0.01, omega = 1/16
    #    res_osc.get_task_path(44), # A = 0.01, omega = 1/16
    #    res_osc.get_task_path(92), # A = 0.01, omega = 1/16
    #    res_osc.get_task_path(93), # A = 0.01, omega = 1/16
    #]
    #
    #tasks_S13_A_0001_inphase = [
    #    res_osc.get_task_path(70), # A = 0.001, omega = 1/16
    #    res_osc.get_task_path(97), # A = 0.001, omega = 1/16
    #    res_osc.get_task_path(98), # A = 0.001, omega = 1/16
    #    res_osc.get_task_path(99), # A = 0.001, omega = 1/16
    #]

    #labels = [r'$A = 0$', r'$A = 0.01, \omega = 1/16$', r'$A = 0.05, \omega = 1/16$']
    #labels = [r'$A = 0$', r'$A = 5 \times 10^{-2}$']

    #tasks = [
    #    res.get_task_path(430), # A = 0
    #    res_osc.get_task_path(38), # A = 0.01, omega = 1/16
    #    res_osc.get_task_path(39), # A = 0.01, omega = 1/16
    #]

    fig, ax = plt.subplots(1, 1, figsize=(6, 5.5))
    ax.plot([300, 350], [0.02, 0.02], '--', linewidth=4, color='black', label=r'$A = 0$')
    colors = ['cyan', 'blue', 'red', 'green', 'orange', 'magenta', 'grey', 'lime', 'navy', 'teal', 'goldenrod']
    obj_name = 'turb_fraction_by_re'
    tf_growth_rates = []
    for k, tasks in enumerate(tasks):
        Res_tmp = np.array([])
        tf_growth_rate_tmp = np.array([])
        for i, task in enumerate(tasks):
            #_plot_turbulent_fraction_growth_rate_from_dump(ax, task, color='red' if i != 0 else None)
            Res, tf_growth_rate = _plot_turbulent_fraction_growth_rate_from_dump(ax, res_ctrl, task, obj_name, color=None, label=labels[k] if i == 0 else None)
    #        Res_tmp = np.r_[Res_tmp, Res]
    #        tf_growth_rate_tmp = np.r_[tf_growth_rate_tmp, tf_growth_rate]
        #Re_filter = (Res_tmp >= 287) & (Res_tmp < 340)
    #    Re_filter = (Res_tmp >= 280)
    #    Ress.append(Res_tmp[Re_filter])
    #    tf_growth_rates.append(tf_growth_rate_tmp[Re_filter])
    x_lims = ax.get_xlim()
    ax.plot(x_lims, [0., 0.], linewidth=4, color='black')

    #ax.set_ylim((-0.0004, 0.0004))
    #ax.set_xlim((340, 400))
    #ax.legend(loc='lower right', fontsize=12)
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True)
    ax.set_xlabel(r'$Re$', fontsize=16)
    ax.set_ylabel(r'$\langle c \rangle$', fontsize=16)
    ax.set_xticks(range(300, 505, 20))
    plt.tight_layout()
    plt.savefig('front_speed.eps', dpi=200)
    plt.show()
