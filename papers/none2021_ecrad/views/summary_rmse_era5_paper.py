import os
import sys

from matplotlib import colors
sys.path.append(os.getcwd())
import pickle
import json

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import iris
import iris.plot as iplt

import restools
from papers.none2021_ecrad.data import Summary
from papers.none2021_ecrad.extensions import ERA5Data, IfsIO, extract_or_interpolate, get_ifs_rel_diff, get_ifs_abs_rel_diff, get_ifs_rmse
from comsdk.research import Research
from comsdk.misc import load_from_json, find_all_files_by_named_regexp
from thequickmath.reduced_models.transition_to_turbulence import MoehlisFaisstEckhardtModel


if __name__ == '__main__':
    plt.style.use('resources/default.mplstyle')

    summary = load_from_json(Summary)
    res_id = summary.res_id
    res = Research.open(res_id)
    task_path = res.get_task_path(summary.task_for_oifs_results)
    era5_task_path = res.get_task_path(summary.task_for_era5_data)
    step_shifts = (0, 64, 128, 192, 256, 320)
    #step_shifts = (0, 160, 320)
    ecrad_ref_run = 'ecrad_tripleclouds_52bits'
    ifs_era5_data = ERA5Data(os.path.join(era5_task_path, 'anton_T_t_pl_days.nc'),
                            any_ref_nc_file_containing_same_lat_and_lon=os.path.join(task_path, 'hgom', ecrad_ref_run, 'sh', '0.nc'),
                            ref_lat_lon_naming='short')
    ifs_era5_data_with_t2m = ERA5Data(os.path.join(era5_task_path, 'anton_T_t_sfc_days.nc'),
                                      any_ref_nc_file_containing_same_lat_and_lon=os.path.join(task_path, 'hgom', ecrad_ref_run, 'sh', '0.nc'),
                                      ref_lat_lon_naming='short')
    ifs_io_ref = IfsIO([os.path.join(task_path, 'hgom', ecrad_ref_run, 'sh', f'{id_}.nc') for id_ in step_shifts],
                       [os.path.join(task_path, 'hgom', ecrad_ref_run, 'gg', f'{id_}.nc') for id_ in step_shifts],
                       l91_file=summary.l91_file)
    ecrad_runs = ('ecrad_mcica_52bits', 'ecrad_tripleclouds_23bits', 'ecrad_tripleclouds_mixed_precision')
    ecrad_run_labels = ('McICA (52 sbits)', 'Tripleclouds (23 sbits)', 'Tripleclouds (mixed precision)')
    ecrad_ref_run_label = 'ERA5'
    quantity_info = [
        {
            'quantity': 'temperature',
            'pressure': 300.,
            'title': '300-hPa temperature',
        },
        {
            'quantity': 'temperature',
            'pressure': 700.,
            'title': '700-hPa temperature',
        },
        {
            'quantity': 'temperature_at_2m',
            'pressure': None,
            'title': '2-m temperature',
        },
    ]
    n_timesteps = len(step_shifts)
    n_runs = len(ecrad_runs)
    n_quantities = len(quantity_info)

    #vmin = -0.00015
    #vcenter = 0.
    #vmax = 0.00015
    vmin = 47500
    vmax = 58400
    vcenter = (vmin + vmax) / 2.

    fig, axes = plt.subplots(1, len(quantity_info), figsize=(12, 6))
    for q_i in range(n_quantities):
        for run_i in range(n_runs):
            ifs_io = IfsIO([os.path.join(task_path, 'hgom', ecrad_runs[run_i], 'sh', f'{id_}.nc') for id_ in step_shifts],
                           [os.path.join(task_path, 'hgom', ecrad_runs[run_i], 'gg', f'{id_}.nc') for id_ in step_shifts],
                            l91_file=summary.l91_file)
            vals_in_percent = []
            days = []
            for ts_i in range(n_timesteps):
                time = ifs_io.times()[ts_i]
                ifs_era5_data_ref = ifs_era5_data_with_t2m if quantity_info[q_i]['quantity'] == 'temperature_at_2m' else ifs_era5_data
                ref_rmse = get_ifs_rmse(ifs_era5_data_ref, ifs_io_ref, time, ecrad_runs[run_i], quantity=quantity_info[q_i]['quantity'], 
                                        pressure=quantity_info[q_i]['pressure'])
                rmse = get_ifs_rmse(ifs_era5_data_ref, ifs_io, time, ecrad_runs[run_i], quantity=quantity_info[q_i]['quantity'], 
                                    pressure=quantity_info[q_i]['pressure'])
                rel_rmse_deviation = (rmse.data.item() - ref_rmse.data.item()) / ref_rmse.data.item()
                vals_in_percent.append(rel_rmse_deviation * 100)
                days.append(ifs_io.time_shift(ts_i).days)
            axes[q_i].plot(days, vals_in_percent, 'o--', label=ecrad_run_labels[run_i], 
                           linewidth=2 if ecrad_runs[run_i] == 'ecrad_tripleclouds_23bits' else 3)
        axes[q_i].set_xticks([0, 2, 4, 6, 8, 10])
        axes[q_i].set_xlabel('Forecast day', fontsize=16)
        axes[q_i].set_title(quantity_info[q_i]['title'], fontsize=16)
        axes[q_i].set_ylim((-5, 5))
        axes[q_i].grid()
        axes[q_i].legend()
    axes[0].set_ylabel(r'Change in RMS error (\%)', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'summary_rmse_era5.eps', dpi=200)
    plt.show()
