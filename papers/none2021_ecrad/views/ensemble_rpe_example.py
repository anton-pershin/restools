import os
import sys
sys.path.append(os.getcwd())
import pickle
import json

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import restools
from restools.plotting import build_zooming_axes_for_plotting_with_box
from papers.none2021_ecrad.data import Summary, ReducedPrecisionVersion
from comsdk.research import Research
from comsdk.comaux import dump_to_json, load_from_json, find_all_files_by_named_regexp


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
    plt.style.use('resources/default.mplstyle')

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
    #ax_ens.semilogx(rp_data.means, indices, '--o', linewidth=1, label='Reduced precision')
    #ax_ens.semilogx(ref_data.means, indices, '--o', linewidth=1, label='Reference')
    ax_ens.errorbar(rp_data.means, indices, xerr=3*np.array(ref_data.stds), fmt='--o', capsize=2.0, elinewidth=2, capthick=2, linewidth=1, label='Half precision')
    ax_ens.errorbar(ref_data.means, indices, xerr=3*np.array(ref_data.stds), fmt='--o', capsize=2.0, elinewidth=2, capthick=2, linewidth=1, label='Double precision')
    #ax_ens.text(1.5*10**(-5), 3.5, 'Subnormal numbers', fontdict=dict(fontsize=12, color='#555'))
    ax_ens.text(1.5*10**(-8), 0.5, 'Subnormal numbers', fontdict=dict(fontsize=12, color='#555'))
    ax_ens.set_xscale('log')
    #ax_ens.fill_betweenx(indices, np.array(ref_data.means) - 3*np.array(ref_data.stds), np.array(ref_data.means) + 3*np.array(ref_data.stds))
    ax_rel_error.set_yticklabels([])
    ax_ens.set_yticks(indices)
#    ax_ens.set_yticklabels([
#        '1   x(1) = 3.6',
#        '2   x(2) = 4.9',
#        '3   m(1) = 5.3e-4',
#        '4   m(2) = 6.3e-4',
#        '5   call perturb_ensemble(m)',
#        '...',
#        '6   m = m / 10000.0',
#        '...',
#        '7   com = (m(1)*x(1) + m(2)*x(2)) /&\n              &(m(1) + m(2))',
#    ], fontsize=8, usetex=False)
    ax_ens.set_yticklabels([
        '1   x(1) = 3.6',
        '2   x(2) = 4.9',
        '3   m(1) = 5.3e-4',
        '4   m(2) = 6.3e-4',
        '5   m = m / 10000.0',
        '6   com = (m(1)*x(1) + m(2)*x(2)) /&\n              &(m(1) + m(2))',
    ], fontsize=8, usetex=False)
    ax_ens.set_xlabel('Assignment values', fontsize=12, usetex=False)
    ax_ens.fill_betweenx([min(indices) - 1, max(indices) + 1], 10**(-9), x2=2**(-14), color='#ddd')
    ax_ens.set_xlim((4*10**(-9), 10**(2)))
    ax_ens.legend(loc='center right')
    ax_rel_error.set_xlabel('Relative error per assignment', fontsize=12, usetex=False)
    for ax in (ax_ens, ax_rel_error):
        ax.grid()
        ax.set_ylim((indices[-1] + 0.5, indices[0] - 0.5))
    for label in ax_ens.get_yticklabels():
        label.set_horizontalalignment('left')
    ax_ens.tick_params(axis='y', which='major', pad=160)
    ax_rel_error.set_xticks([10**(-5), 10**(-4), 10**(-3), 10**(-2), 10**(-1)])
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 1.0])

    axins = build_zooming_axes_for_plotting_with_box(fig, ax_ens,
                                                     #parent_box=[3.5*10**(-8), 7.25, 6.2*10**(-8), -1.5],
                                                     #child_box=[10**(-8), 4.6, 8*10**(-7), -1.5],
                                                     parent_box=[3.5*10**(-8), 4.25, 6.2*10**(-8), -0.5],
                                                     child_box=[10**(-8), 3.2, 8*10**(-7), -1.5],
                                                     parent_vertices=[1, 2],
                                                     child_vertices=[0, 3],
                                                     remove_axis=False)
    #axins = ax_ens.inset_axes([0.5, 0.5, 0.4, 0.2])
    indices[-3] = -5*10**4
    indices[-1] = 5*10**8
    axins.errorbar(rp_data.means, indices, xerr=3*np.array(rp_data.stds), fmt='o--', capsize=2.0, elinewidth=2, capthick=2, linewidth=1)
    axins.errorbar(ref_data.means, indices, xerr=3*np.array(ref_data.stds), fmt='o--', capsize=2.0, elinewidth=2, capthick=2, linewidth=1)
    #axins.plot(ref_data.raw_ensembles[4], [4]*len(ref_data.raw_ensembles[4]), 'ok', markersize=1, zorder=50)
    axins.set_xlim((5.0*10**(-8), 6.3*10**(-8)))
    axins.set_ylim((4.1, 3.9))
    axins.set_xticks([])
    axins.set_yticks([])

    #x1, x2, y1, y2 = 4*10**(-8), 7*10**(-8), 5.5, 7.5
    #axins.set_xlim(x1, x2)
    #axins.set_ylim(y1, y2)
    #axins.set_xticklabels('')
    #axins.set_yticklabels('')
    #ax_ens.indicate_inset_zoom(axins, edgecolor="black")

    plt.savefig(f'ensemble_rpe_example.eps', dpi=200)
    plt.show()
