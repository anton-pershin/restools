import os
import sys
sys.path.append(os.getcwd())
import pickle
import json

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import restools
from papers.none2021_ecrad.data import Summary, ReducedPrecisionVersion
from papers.none2021_ecrad.extensions import VariableData, collect_variable_data, plot_variable_histogram
from comsdk.research import Research
from comsdk.comaux import dump_to_json, load_from_json, find_all_files_by_named_regexp


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
    task = 1
    res_id = summary.res_id
    #res_id = 'IFSMPI'
    res = Research.open(res_id)
    task_path = res.get_task_path(task)
    #subroutine_name = 'calc_overlap_matrices'
    #subroutine_name = 'calc_two_stream_gammas_lw_rpe'
    #subroutine_name = 'calc_two_stream_gammas_sw_rpe'
    subroutine_name = 'solver_tripleclouds_lw'
    #subroutine_name = 'solver_tripleclouds_sw'
    #subroutine_name = 'radiation_interface'
    #subroutine_name = 'lorenz'
    all_vars_path = os.path.join(task_path, 'rp_dumps', subroutine_name)
    #all_vars_path = os.path.join(task_path, 'rp_dumps')

    fig, ax = plt.subplots(figsize=(12, 6))
    variables = collect_variable_data(all_vars_path, group_by='variable_only', count_zero_assignments=False)
    for var_name, var_data in variables.items():
        print(f'{var_name}: {var_data.n_ops} assignments')
    im = plot_variable_histogram(fig, ax, variables, figname=f'variables_range_{subroutine_name}.eps')
    #fig.colorbar(im, ax=ax)
    #plt.tight_layout()
    #plt.show()
