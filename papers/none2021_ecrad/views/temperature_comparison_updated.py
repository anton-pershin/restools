import os
import sys
sys.path.append(os.getcwd())
import pickle
import json

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import iris
import iris.plot as iplt

import restools
from papers.none2021_ecrad.data import Summary
from papers.none2021_ecrad.extensions import ERA5Data, IfsIO, extract_or_interpolate
from comsdk.research import Research
from comsdk.misc import load_from_json, find_all_files_by_named_regexp
from thequickmath.reduced_models.transition_to_turbulence import MoehlisFaisstEckhardtModel


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
    #task_path = res.get_task_path(summary.task_for_oifs_results)
    task_path = res.get_task_path(18)
    step_shifts = (0, 64, 128, 192, 256, 320)
    #ecrad_runs = ('ecrad_mcica_52bits', 'ecrad_tripleclouds_52bits', 'ecrad_tripleclouds_mixed_precision')
    ecrad_runs = ('era5', 'ecrad_tripleclouds_52bits')
    n_cols = len(step_shifts)
    n_rows = len(ecrad_runs)
    quantity='temperature'
    #vmin = -0.00015
    #vcenter = 0.
    #vmax = 0.00015
    vmin = 200
    vmax = 300
    vcenter = (vmin + vmax) / 2.
    pressure=500.

    #fig = plt.figure(figsize=(12, 7))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 7))
    for run_i in range(n_rows):
        for ts_i in range(n_cols):
            if run_i == 0:
                ifs_io = ERA5Data('/Volumes/Seagate/Leeds/PhD/research/2021-05-18_ecRad_reduced_precision/17-ERA5DataForEcRad/anton_T_t_pl_days.nc')
            else:
                ifs_io = IfsIO(time_series_sh_files=[os.path.join(task_path, 'hgom', ecrad_runs[run_i], 'sh', f'{id_}.nc') for id_ in step_shifts],
                               time_series_gg_files=[os.path.join(task_path, 'hgom', ecrad_runs[run_i], 'sh', f'{id_}.nc') for id_ in step_shifts],
                               l91_file=summary.l91_file)
            q = extract_or_interpolate(getattr(ifs_io, quantity)(ts_i), pressure)
            #print(np.min(q.data), np.max(q.data))
            #plt.subplot(1, n_cols, col)
            plt.sca(axes[run_i][ts_i])
            cf = iplt.contourf(q, 32, norm=TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax), 
                                cmap=plt.get_cmap('seismic'))
            plt.gca().coastlines()
            plt.gca().set_title(f'+{ifs_io.time_shift(ts_i)}')
#        if col == 1:
#            plt.gca().text(-0.1, 0.5, f'{titles[row - 1]}', transform=plt.gca().transAxes, 
#                            va='center', fontsize=14, rotation='vertical')
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




    plt.tight_layout()
    plt.savefig(f'temperature_rel_error.eps', dpi=200)
    plt.show()
