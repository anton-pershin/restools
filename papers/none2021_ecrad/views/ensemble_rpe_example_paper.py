import os
import sys
sys.path.append(os.getcwd())
import pickle
import json

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import restools
from restools.plotting import build_zooming_axes_for_plotting_with_box
from papers.none2021_ecrad.data import Summary, ReducedPrecisionVersion
from comsdk.research import Research
from comsdk.misc import dump_to_json, load_from_json, find_all_files_by_named_regexp


def turn_to_int(s):
    return int(s.strip())


def turn_to_float(s):
    return float(s.strip())


class RpeEnsembleData(object):
    """docstring for RpeEnsembleData"""
    def __init__(self):
        self.means = []
        self.stds = []
        self.raw_ensembles = []

    def add_ensemble_update(self, ens):
        self.means.append(np.mean(ens))
        self.stds.append(np.std(ens))
        self.raw_ensembles.append(ens)


if __name__ == '__main__':
    #plt.style.use('resources/default.mplstyle')

#    s = Summary(res_id='ECRAD',
#                l137_file='/home/tony/projects/oxford/pershin/misctools/L137.csv',
#                oxford_input_path='/network/aopp/chaos/pred/shared/ecRad_data',
#                oxford_output_path='/network/aopp/chaos/pred/pershin/ecrad/outputs',
#                solvers=['tripleclouds', 'mcica'],
#                rp_versions=[ReducedPrecisionVersion(name='', descr='Vanilla reduced precision, without any improvements. Simply all the variables were turned to given precision (except for those leading to floating-point exceptions)'),
#                             ReducedPrecisionVersion(name='all_flux_vars_single_precision', descr='''
#Like vanilla reduced precision, but mixed precision is introduced where single precision (23 bits) is used for several important variables (mostly those summing up fluxes):
#- radiation_interface.F90: flux%lw_up, flux%lw_dn, flux%sw_up, flux%sw_dn, flux%sw_dn_direct, flux%lw_up_clear, flux%lw_dn_clear, flux%sw_up_clear, flux%sw_dn_clear, flux%sw_dn_direct_clear
#- radiation_tripleclouds_lw.F90: flux_dn, flux_dn_below, flux_up, flux_dn_clear, flux_up_clear
#- radiation_tripleclouds_sw.F90: flux_dn, flux_dn_below, flux_up, direct_dn, direct_dn_below, direct_dn_clear, flux_dn_clear, flux_up_clear'''),
#                             ReducedPrecisionVersion(name='ieee_half', descr='''
#Like all_flux_vars_single_precision, but the code is modified so that IEEE half precision rules are respected, so the
#exponent is also controlled. Again we needed to turn few variables to single precision:
#- radiation_interface.F90: od_lw
#- radiation_tripleclouds_lw.F90: reflectance, transmittance
#- radiation_tripleclouds_sw.F90: reflectance, transmittance
#- radiation_two_stream.F90: calc_reflectance_transmittance_lw is in double precision
#- radiation_overlap.F90: one_over_cf, cf_upper, cf_lower in calc_alpha_overlap_matrix
#''')])
#    dump_to_json(s)
#    raise Exception('qwer')

    summary = load_from_json(Summary)
    task = 11
    res_id = summary.res_id
    res = Research.open(res_id)
    task_path = res.get_task_path(task)
    dump_path = os.path.join(task_path, 'dumps')
    files = os.listdir(dump_path)
    files.remove('summary')
    max_index = max(map(lambda f: int(os.path.splitext(f)[0]),
                        files))
    rp_data = RpeEnsembleData()
    ref_data = RpeEnsembleData()
    for i in range(1, max_index + 1):
        f = open(os.path.join(dump_path, f'{i}.csv'), 'r')
        for d in (rp_data, ref_data):
            d.add_ensemble_update([float(n) for n in f.readline().split(',')])
    #print(np.max(np.abs(df_exact['x_i'].to_numpy() - df_rp['x_i'].to_numpy())))
    indices = list(range(len(rp_data.means)))
#    fig, (ax_ens, ax_rel_error) = plt.subplots(1, 2, figsize=(12, 6))

    fig = plt.figure(figsize=(12, 3.5))
    gs = fig.add_gridspec(1, 3)
    ax_ens = fig.add_subplot(gs[0, :2])
    ax_rel_error = fig.add_subplot(gs[0, 2])
    ax_rel_error.semilogx(np.abs(np.array(rp_data.means) - np.array(ref_data.means)) / np.abs(np.array(ref_data.means)), indices, '--o', linewidth=1, label='Reduced precision')
    ax_rel_error.semilogx([2**(-10), 2**(-10)], [min(indices), max(indices)], 'k', linewidth=2, label='Reduced precision')
    for i, rp_mean, ref_mean, ref_std in zip(indices, rp_data.means, ref_data.means, ref_data.stds):
        ax_ens.errorbar([10**(-16), ref_mean], [i, i], xerr=3*np.array([0, ref_std]), fmt='o--', 
                        color='tab:blue', capsize=2.0, elinewidth=2, capthick=2, linewidth=2, markersize=8,
                        label='Double precision' if i == 0 else None)
        ax_ens.errorbar([10**(-16), rp_mean], [i, i], xerr=3*np.array([0, ref_std]), fmt='o--', 
                        color='tab:orange', capsize=2.0, elinewidth=2, capthick=2, linewidth=2, markersize=8,
                        label='Half precision' if i == 0 else None)
    ax_ens.fill_betweenx([min(indices) - 1, max(indices) + 1], 10**(-9), x2=2**(-14), color='#ddd')
    ax_ens.text(3*10**(-9), 0.6, 'Subnormal numbers', fontdict=dict(fontsize=12, color='#555'))
    ax_ens.set_xscale('log')

    divider = make_axes_locatable(ax_ens)
    ax_ens_lin = divider.append_axes('right', size=2.0, pad=0)
    for i, rp_mean, ref_mean, ref_std in zip(indices, rp_data.means, ref_data.means, ref_data.stds):
        ax_ens_lin.errorbar([10**(-16), ref_mean], [i, i], xerr=3*np.array([0, ref_std]), fmt='o--', 
                            color='tab:blue', capsize=2.0, elinewidth=2, capthick=2, linewidth=2, markersize=8, 
                            label='Double precision' if i == 0 else None)
        ax_ens_lin.errorbar([10**(-16), rp_mean], [i, i], xerr=3*np.array([0, ref_std]), fmt='o--', 
                            color='tab:orange', capsize=2.0, elinewidth=2, capthick=2, linewidth=2, markersize=8,
                            label='Half precision' if i == 0 else None)

#    axins = build_zooming_axes_for_plotting_with_box(fig, ax_ens,
#                                                     parent_box=[3.5*10**(-8), 4.25, 6.2*10**(-8), -0.5],
#                                                     child_box=[10**(-8), 3.2, 8*10**(-7), -1.5],
#                                                     parent_vertices=[1, 2],
#                                                     child_vertices=[0, 3],
#                                                     remove_axis=False)
#    #axins = ax_ens.inset_axes([0.5, 0.5, 0.4, 0.2])
#    #indices[-3] = -5*10**4
#    #indices[-1] = 5*10**8
#    axins.errorbar(rp_data.means, indices, xerr=3*np.array(rp_data.stds), fmt='o--', capsize=2.0, elinewidth=2, capthick=2, linewidth=1)
#    axins.errorbar(ref_data.means, indices, xerr=3*np.array(ref_data.stds), fmt='o--', capsize=2.0, elinewidth=2, capthick=2, linewidth=1)
#    #axins.plot(ref_data.raw_ensembles[4], [4]*len(ref_data.raw_ensembles[4]), 'ok', markersize=1, zorder=50)
#    axins.set_xlim((5.0*10**(-8), 6.3*10**(-8)))
#    axins.set_ylim((4.1, 3.9))
#    axins.set_xticks([])
#    axins.set_yticks([])


    ax_ens.set_xlim((10**(-9), 3.))
    ax_ens_lin.set_xlim((3., 5.1))

    ax_rel_error.set_yticklabels([])
    ax_ens.set_yticks(indices)
    ax_ens.set_yticklabels([
        '1   x(1) = 3.6',
        '2   x(2) = 4.9',
        '3   m(1) = 5.3e-4',
        '4   m(2) = 6.3e-4',
        '5   m = m / 10000.0',
        '6   com = (m(1)*x(1) + m(2)*x(2)) /&\n              &(m(1) + m(2))',
    ], fontsize=8, usetex=False)
    ax_ens.set_xlabel('Assignment values', fontsize=12, usetex=False)
    ax_ens_lin.legend(loc='center right')
    ax_rel_error.set_xlabel('Relative error per assignment', fontsize=12, usetex=False)
    for ax in (ax_ens, ax_ens_lin, ax_rel_error):
        ax.grid()
        ax.set_ylim((indices[-1] + 0.5, indices[0] - 0.5))
    for label in ax_ens.get_yticklabels():
        label.set_horizontalalignment('left')
    ax_ens_lin.spines['left'].set_visible(False)  # removes bottom axis line
    ax_ens_lin.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left=False,      # ticks along the bottom edge are off
        right=False,         # ticks along the top edge are off
        labelbottom=False)
    ax_ens_lin.set_yticklabels([])

    ax_ens.tick_params(axis='y', which='major', pad=160)
    ax_rel_error.set_xticks([10**(-5), 10**(-4), 10**(-3), 10**(-2), 10**(-1)])
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 1.0])
    plt.savefig(f'ensemble_rpe_example.eps', dpi=200)
    plt.show()




def plot_data_on_mixed_linear_log_scale(fig, ax, x_data_list, y_data_list, label_list, ylabel='Pressure (hPa)',
                                        xscale='log', ylim_linear=(1000, 101), ylim_log=(101, 0.007), 
                                        ylabel_shirt=-0.02, **kwargs):
    for x_data, y_data, label in zip(x_data_list, y_data_list, label_list):
        ax.plot(x_data, y_data, linewidth=2, label=label, **kwargs)
    ax.grid()
    ax.set_ylim(ylim_linear)
    ax.set_xscale(xscale)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    divider = make_axes_locatable(ax)
    ax_log = divider.append_axes("top", size=2.0, pad=0, sharex=ax)
    for x_data, y_data, label in zip(x_data_list, y_data_list, label_list):
        ax_log.plot(x_data, y_data, linewidth=2, label=label, **kwargs)
    ax_log.set_yscale('log')
    ax_log.set_ylim(ylim_log)
    ax_log.grid()
#    ax_log.legend()
    ax_log.spines['bottom'].set_visible(False)  # removes bottom axis line
    #ax_log.xaxis.set_ticks_position('top')
    ax_log.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)
    fig.text(ylabel_shirt, 0.55, ylabel, va='center', rotation='vertical', fontsize=16)
    return ax, ax_log
