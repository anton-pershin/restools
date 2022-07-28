import os
import sys
sys.path.append(os.getcwd())
import pickle
import json

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import restools
from papers.none2021_ifsmpi.data import Summary, ReducedPrecisionVersion
from papers.none2021_ecrad.extensions import VariableData, collect_variable_data, plot_variable_histogram, \
    plot_single_variable_histogram
from comsdk.research import Research
from comsdk.misc import dump_to_json, load_from_json, find_all_files_by_named_regexp


if __name__ == '__main__':
    plt.style.use('resources/default.mplstyle')

#    s = Summary(res_id='IFSMPI',
#                oxford_output_path='/network/aopp/chaos/pred/pershin/oifs/oifsruns',
#                rp_versions=[ReducedPrecisionVersion(name='52bits', descr='''
#24-hours weather prediction for double-precision MPI comminication (trltom and trmtol). Ground truth for us'''),
#                             ReducedPrecisionVersion(name='23bits', descr='''
#24-hours weather prediction for single-precision MPI comminication (trltom and trmtol)'''),
#                             ReducedPrecisionVersion(name='16bits', descr='''
#24-hours weather prediction for 16-sbits-precision MPI comminication (trltom and trmtol)'''),
#                             ReducedPrecisionVersion(name='10bits', descr='''
#24-hours weather prediction for half-precision MPI comminication (trltom and trmtol) without the control of exponent.
#This is a vanilla half-precision version without any improvements'''),
#                             ReducedPrecisionVersion(name='10bits_zero_mode_52bits_in_direct_and_inverse',
#descr='''24-hours weather prediction for half-precision MPI comminication (trltom and trmtol) without the control of
#exponent. This is an improved version where parts of the communicated buffers corresponding to zero Fourier modes
#are kept in double precision. This gives an accuracy comparable to a single-precision version'''),
#                             ReducedPrecisionVersion(name='10bits_zero_mode_52bits_in_direct_and_inverse_IEEE_half_precision',
#descr='''24-hours weather prediction for IEEE half-precision MPI comminication (trltom and trmtol) with the control of
#exponent. TO BE UPDATED!!!'''),])
#    dump_to_json(s)
#    raise Exception('qwer')

    summary = load_from_json(Summary)
    task = 2
    res_id = summary.res_id
    res = Research.open(res_id)
    task_path = res.get_task_path(task)
    #subroutine_name = 'trltom'
    subroutine_name = 'trmtol'
    all_vars_path = os.path.join(task_path, 'rp_dumps', subroutine_name)
    buffers_path = os.path.join(task_path, 'buffer_dumps', subroutine_name, 'proc_4')

    # PRINT BUFFER VALUES

#    with open(os.path.join(buffers_path, '9.csv')) as f:
#        vals = np.array([float(x) for x in f.read().split()])
#        vals = np.abs(vals)
#    last_moving_average = None
#    next_moving_average = None
#    change_indices = []
#    kfields = 454
#    i = kfields + 4
#    while i < 2*kfields - 4:
#        last_moving_average = np.mean(vals[i-4:i])
#        next_moving_average = np.mean(vals[i:i+4])
#        if np.abs(np.log10(last_moving_average) - np.log10(vals[i])) > 5:
#            change_indices.append(i)
#            i += 4
#        i += 1
#
#    change_indices = np.array([i - kfields for i in change_indices], dtype=int)
#    #n_patches = int(len(vals) / kfields)
#    #for i in range(n_patches):
#    #    vals[i*kfields:i*kfields + change_indices[0]] *= 10**7
#    #    vals[i*kfields + change_indices[0]:i*kfields + change_indices[1]] *= 10**(-5)
#    #    vals[i*kfields + change_indices[1]:(i+1)*kfields] *= 10**2
#
#    print(f'change_indices: {change_indices}')
#    print(f'Max: {np.max(np.abs(vals))}, min: {np.min(np.abs(vals[vals > 0.0]))}')
#    fig, ax = plt.subplots(figsize=(12, 5))
#    #sns.histplot(vals[vals > 0.0], bins=50, log_scale=True, ax=ax)
#
#    n_patches = int(len(vals[:4000]) / kfields)
#    for i in range(n_patches):
#        vals[i*kfields:i*kfields + change_indices[0]] *= 10**7
#        vals[i*kfields + change_indices[0]:i*kfields + change_indices[1]] *= 10**(-5)
#        vals[i*kfields + change_indices[1]:(i+1)*kfields] *= 10**2
#        ax.semilogy(np.arange(i*kfields, i*kfields + change_indices[0]), vals[i*kfields:i*kfields + change_indices[0]], 'o',
#                    color='#20639b', markersize=3)
#        ax.semilogy(np.arange(i*kfields + change_indices[0], i*kfields + change_indices[1]), vals[i*kfields + change_indices[0]:i*kfields + change_indices[1]], 'o',
#                    color='#ed553b', markersize=3)
#        ax.semilogy(np.arange(i*kfields + change_indices[1], (i+1)*kfields), vals[i*kfields + change_indices[1]:(i+1)*kfields], 'o',
#                    color='#3caea3', markersize=3)
#
#
#    #ax.semilogy(range(4000), vals[:4000], 'o')
#    ax.set_ylim((10**(-12), 10**7))
#    ylims = ax.get_ylim()
#    for i in range(1, n_patches):
#        ax.semilogy([i*kfields, i*kfields], ylims, '-', color='gray')
#    #ax.semilogy([i*kfields for i in range(1, 6)], [vals[i*kfields] for i in range(1, 6)], 'o')
#    #for j in range(4):
#    #    ax.semilogy(change_indices + j*kfields, [vals[i] for i in change_indices + j*kfields], 'go')
#    #ax.grid()
#    ax.set_xlabel('Buffer index', usetex=False)
#    ax.set_ylabel('Value', usetex=False)
#    plt.tight_layout()
#    plt.savefig('trmtol_buffer_values_type_ii_scaled.eps', dpi=200)
#    plt.show()


    # PRINT MEAN VALUES FOR EACH MPI CALL

#    variables = collect_variable_data(all_vars_path, group_by='variable_and_call_number', count_zero_assignments=True)
#    mean_values = []
#    max_values = []
#    for proc_name, d in variables.items():
#        print(f'Processor {proc_name}')
#        for call_num, var_data in d.items():
#            print(f'Call #{call_num}: mean = {var_data.mean}, max = {var_data.max}, min = {var_data.min}, '
#                  f'n_ops = {var_data.n_ops}')
#            fig, ax = plt.subplots(figsize=(8, 3))
#            plot_single_variable_histogram(fig, ax, var_data, figname=f'hist_for_{subroutine_name}_call_{call_num}.eps')
#            if var_data.mean != 0 and var_data.max != 0:
#                mean_values.append(var_data.mean)
#                max_values.append(var_data.max)
#
#    mean_values = np.array(mean_values)
#    max_values = np.array(max_values)
#
#    fig, ax = plt.subplots(figsize=(6, 3))
#    sns.histplot(data={
#        'Mean': mean_values,
#        'Max': max_values,
#            #'Mean': np.log10(mean_values),
#            #'Max': np.log10(max_values),
#        }, ax=ax, element='step', log_scale=True)
#    #xticks = ax.get_xticks()
#    #ax.set_xticklabels([r'$10^{' + str(int(x)) + r'}$' for x in xticks])
#    #sns.histplot(data=np.log10(max_values), ax=ax, element='step')
#    #ax.hist(np.log(mean_values))
#    #ax.hist(np.log(max_values))
#    #ax.set_xscale('log')
#    ax.set_xlabel('Buffer values')
#    plt.tight_layout()
#    plt.savefig(f'mean_max_hist_{subroutine_name}.eps', dpi=200)
#    plt.show()

    # PLOT HISTOGRAMS FOR ALL THE PROCESSORS

    fig, ax = plt.subplots(figsize=(6, 2))
    variables = collect_variable_data(all_vars_path, group_by='variable_only', count_zero_assignments=True)
    for var_name, var_data in variables.items():
        dp_mem_consumption = var_data.n_ops * 8. / 10**9
        dp_mem_consumption_per_step = dp_mem_consumption / 32.
        hp_mem_consumption = dp_mem_consumption / 4.
        hp_mem_consumption_per_step = hp_mem_consumption / 32.
        print(f'{var_name}: {var_data.n_ops} assignments with memory consumption of {dp_mem_consumption:.2f} GBs and '
              f'{dp_mem_consumption_per_step:.3f} GBs/step (double precision) or {hp_mem_consumption:.2f} GBs '
              f'and {hp_mem_consumption_per_step:.3f} GBs/step (half precision) GBs')
    plot_variable_histogram(fig, ax, variables, figname=f'variables_range_{subroutine_name}.eps')
