import os
import sys
sys.path.append(os.getcwd())
import pickle
import json

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import iris

import restools
from papers.none2021_ecrad.data import Summary
from papers.none2021_ecrad.extensions import EcradIO, heating_rate_diff_rms, plot_data_on_mixed_linear_log_scale
from comsdk.research import Research
from comsdk.misc import load_from_json, find_all_files_by_named_regexp
from reducedmodels.transition_to_turbulence import MoehlisFaisstEckhardtModel


def turn_to_int(s):
    return int(s.strip())


def turn_to_float(s):
    return float(s.strip())


if __name__ == '__main__':
    plt.style.use('resources/default.mplstyle')

    summary = load_from_json(Summary)
    task = 15
    res_id = summary.res_id
    res = Research.open(res_id)
    task_path = res.get_task_path(task)
    e_tripleclouds = EcradIO(input_nc_file=os.path.join(task_path, 'inputs', 'era5_2001-06-09-18.nc'),
                       output_nc_file=os.path.join(task_path, 'tripleclouds', 'pure', 'era5_2001-06-09-18_output.nc'),
                       convert_columns_to_latitude_and_longitude=False,
                       l137_file=summary.l137_file)
    e_mcica = EcradIO(input_nc_file=os.path.join(task_path, 'inputs', 'era5_2001-06-09-18.nc'),
                      output_nc_file=os.path.join(task_path, 'tripleclouds', '52bits', 'era5_2001-06-09-18_output.nc'),
                      convert_columns_to_latitude_and_longitude=False,
                      l137_file=summary.l137_file)
#    e_half_stochastic = EcradIO(input_nc_file=os.path.join(task_path, 'era5slice.nc'),
#                                output_nc_file=os.path.join(task_path, 'era5slice_output.nc'),
#                                convert_columns_to_latitude_and_longitude=False,
#                                l137_file=summary.l137_file)
#    e_half = EcradIO(input_nc_file=os.path.join(res.get_task_path(9), 'era5slice.nc'),
#                     output_nc_file=os.path.join(res.get_task_path(9), 'era5slice_output.nc'),
#                     convert_columns_to_latitude_and_longitude=False,
#                     l137_file=summary.l137_file)
#    task_path = '/home/tony/projects/oxford/pershin/ecrad/practical'
#    e_half_stochastic = EcradIO(input_nc_file=os.path.join(task_path, 'era5slice.nc'),
#                                output_nc_file=os.path.join(task_path, 'era5slice_output_1.nc'),
#                                convert_columns_to_latitude_and_longitude=False,
#                                l137_file=summary.l137_file)
#    e_half = EcradIO(input_nc_file=os.path.join(task_path, 'era5slice.nc'),
#                     output_nc_file=os.path.join(task_path, 'era5slice_output_2.nc'),
#                     convert_columns_to_latitude_and_longitude=False,
#                     l137_file=summary.l137_file)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, type_, ax_title in zip(axes, ('lw', 'sw'), ('Longwave', 'Shortwave')):
        x_data_list = []
        y_data_list = []
        #label_list = ('binary64', 'binary16', 'binary16+stochastic')
        #for e in (e_double, e_half, e_half_stochastic):
        label_list = ('Tripleclouds (original)', 'Tripleclouds (fixed for neg. mix. ratio)')
        for e in (e_tripleclouds, e_mcica):
            heating_rate = heating_rate_diff_rms(e_tripleclouds, e, type=type_, keep_air_pressure=True)
            hr_cube = e.ecrad_output_as_iris_cube(f'heating_rate_{type_}')
            hr_cube = hr_cube.collapsed(['latitude'], iris.analysis.RMS)
            x_data_list.append(hr_cube.data)
            y_data_list.append(hr_cube.coord('air_pressure').points / 10**2)
        ax_lin, ax_log = plot_data_on_mixed_linear_log_scale(fig, ax, x_data_list, y_data_list, label_list, xscale='linear',
                                                         ylim_linear=(1000, 101), ylim_log=(101, 0.007))
        ax_log.set_title(ax_title, fontsize=18)
#        ylim = ax.get_ylim()
        #ax.set_xlim((-17, 2.5))
#        ax.set_ylim((ylim[1], ylim[0]))
        #ax.tick_params(labelsize='14')
        ax_lin.set_xlabel(r'K $\times$ d$^{-1}$')
        ax_lin.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'heating_rates_task_{task}.eps', dpi=200)
    plt.show()
