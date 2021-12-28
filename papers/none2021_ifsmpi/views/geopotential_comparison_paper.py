import os
import sys

from matplotlib import colors
sys.path.append(os.getcwd())
import pickle
import json

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import DivergingNorm
import iris
import iris.plot as iplt

import restools
from papers.none2021_ifsmpi.data import Summary
from papers.none2021_ecrad.extensions import IfsIO, extract_or_interpolate, get_ifs_rel_diff
from comsdk.research import Research
from comsdk.comaux import load_from_json, find_all_files_by_named_regexp


#def plot_comparison(ifs_io, ifs_io_rps, avail_data, quantity='geopotential', vmin=-0.00015, vmax=0.00015, 
#                    pressure=500., comp_func=get_rel_diff):
#    n_rows = len(ifs_io_rps)
#    n_cols = len(ifs_io)
#    avail_dirs = list(avail_data.keys())
#    titles = list(avail_data.values())
#
#    fig = plt.figure(figsize=(12, 7))
#    for row in range(1, n_rows + 1):
#        ifs_io_rp = ifs_io_rps[row - 1]
#        for col in range(1, n_cols + 1):
#            ts_i = col - 1
#            comp = comp_func(ifs_io, ifs_io_rp, ts_i, avail_dirs[row - 1], 
#                             quantity=quantity, pressure=pressure)
#            plt.subplot(n_rows, n_cols, (row - 1)*n_cols + col)
#            cf = iplt.contourf(comp, 32, norm=DivergingNorm(vmin=vmin, vcenter=0., vmax=vmax), 
#                               cmap=plt.get_cmap('seismic'))
#            plt.gca().coastlines()
#            plt.gca().set_title(f'+{ifs_io.time_shift(ts_i)}')
#            if col == 1:
#                plt.gca().text(-0.1, 0.5, f'{titles[row - 1]}', transform=plt.gca().transAxes, 
#                               va='center', fontsize=14, rotation='vertical')
##        cbar = plt.gca().colorbar()
##        cbar.set_ticks([min_value, 0., max_value])
#    colorbar_axes = plt.gcf().add_axes([0.35, 0.05, 0.3, 0.05])
#    colorbar = plt.colorbar(cf, colorbar_axes, orientation='horizontal')
#    colorbar.locator = matplotlib.ticker.MaxNLocator(3)
#    colorbar.update_ticks()
#    plt.show()


if __name__ == '__main__':
    plt.style.use('resources/default.mplstyle')

    summary = load_from_json(Summary)
    res_id = summary.res_id
    res = Research.open(res_id)
    task_path = res.get_task_path(summary.oifs_runs_task)
    step_shifts = (0, 16, 32)
    #oifs_runs = ('23bits', '10bits', '10bits_zero_mode_52bits_in_direct_and_inverse_IEEE_half_precision_final')
    #oifs_run_labels = ('Single precision', '10 sbits', 'IEEE half precision')
    oifs_ref_run = '52bits'
    oifs_runs = ('23bits', '10bits_zero_mode_52bits_in_direct_and_inverse_IEEE_half_precision_final')
    oifs_run_labels = ('Single precision', 'IEEE half precision')
    n_timesteps = len(step_shifts)
    n_runs = len(oifs_runs)
    quantity='geopotential_height'
    vmin = -0.001
    vcenter = 0.
    vmax = 0.001
    pressure = 500.

    ifs_io_ref = IfsIO([os.path.join(task_path, 'hgom', oifs_ref_run, 'sh', f'{id_}.nc') for id_ in step_shifts],
                       [os.path.join(task_path, 'hgom', oifs_ref_run, 'gg', f'{id_}.nc') for id_ in step_shifts],
                       l91_file=summary.l91_file)
    #fig = plt.figure(figsize=(12, 7))
    fig, axes = plt.subplots(n_timesteps, n_runs, figsize=(8, 6))
    cf_ref = None
    for run_i in range(n_runs):
        ifs_io = IfsIO([os.path.join(task_path, 'hgom', oifs_runs[run_i], 'sh', f'{id_}.nc') for id_ in step_shifts],
                       [os.path.join(task_path, 'hgom', oifs_runs[run_i], 'gg', f'{id_}.nc') for id_ in step_shifts],
                        l91_file=summary.l91_file)
        for ts_i in range(n_timesteps):
            q_rel_diff = get_ifs_rel_diff(ifs_io_ref, ifs_io, ts_i, oifs_runs[run_i], quantity=quantity, pressure=pressure)
            plt.sca(axes[ts_i][run_i])
#            cf = iplt.contourf(q, 32, norm=DivergingNorm(vmin=vmin, vcenter=vcenter, vmax=vmax), 
#                                cmap=plt.get_cmap('seismic'))
            #if cf_ref is None:
            cf = iplt.contourf(q_rel_diff, 16, norm=DivergingNorm(vmin=vmin, vcenter=0., vmax=vmax), 
                               cmap=plt.get_cmap('seismic'))

            #    cf_ref = cf
            #else:
            #    cf = iplt.contourf(q_rel_diff, 16, cmap=plt.get_cmap('coolwarm'), vmin=cf_ref.zmin, vmax=cf_ref.zmax, levels=cf_ref.levels, coords=['latitude', 'longitude'])

            #iplt.xticks([range()])

            plt.gca().coastlines()
            if ts_i == 0:
                plt.gca().set_title(oifs_run_labels[run_i], usetex=False, fontsize=12)
            if run_i == 0:
                plt.gca().text(-0.1, 0.5, f'+{ifs_io.time_shift(ts_i)}', transform=plt.gca().transAxes,
                                va='center', fontsize=12, rotation='vertical')
#        cbar = plt.gca().colorbar()
#        cbar.set_ticks([min_value, 0., max_value])
    
    colorbar_axes = plt.gcf().add_axes([0.35, 0.05, 0.3, 0.05])
    colorbar = plt.colorbar(cf, colorbar_axes, orientation='horizontal')
    colorbar.locator = matplotlib.ticker.MaxNLocator(3)
    colorbar.update_ticks()

#avail_data = {
#    'ecrad_ieee_half_precision_v2_16bits': 'ecrad 16 bits',
#    'ecrad_ieee_half_precision_v2': 'ecrad IEEE 10 bits',
#} # dir_name -> descr
#ifs_io_rps = [IfsIO([os.path.join(ifs_experiments_path, f'{dir_}', f'{id_}.nc') for id_ in step_shifts]) 
#              for dir_ in avail_data.keys()]
#plot_comparison(ifs_io, ifs_io_rps, avail_data, quantity='temperature', vmin=-0.05, vmax=0.05)

    plt.tight_layout(rect=[0, 0.1, 1, 1], h_pad=0.5, w_pad=1.0)
    plt.savefig(f'geopotential_height_rel_error_oifs.png', dpi=200)
    plt.show()
