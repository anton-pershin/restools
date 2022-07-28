import os
import sys
sys.path.append(os.getcwd())
import pickle
import json

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import iris
import iris.plot as iplt
import iris.quickplot as qplt

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
    e_double = EcradIO(input_nc_file=os.path.join(task_path, 'inputs', 'era5_2001-06-01-00.nc'),
                       output_nc_file=os.path.join(task_path, 'tripleclouds', '52bits', 'era5_2001-06-01-00_output.nc'),
                       convert_columns_to_latitude_and_longitude=True,
                       l137_file=summary.l137_file)
    e_half = EcradIO(input_nc_file=os.path.join(task_path, 'inputs', 'era5_2001-06-01-00.nc'),
                     output_nc_file=os.path.join(task_path, 'tripleclouds', '10bits_all_flux_vars_single_precision', 'era5_2001-06-01-00_output.nc'),
                     convert_columns_to_latitude_and_longitude=True,
                     l137_file=summary.l137_file)

    fig, ax = plt.subplots(figsize=(6, 6))
    x_data_list = []
    y_data_list = []
    for i, e in enumerate((e_double, e_half)):
        heating_rate_sw = e.ecrad_output_as_iris_cube('heating_rate_sw').extract(constraint=iris.Constraint(latitude=lambda l: -5 < l < -2,
                                                                                                            longitude=lambda l: 130 < l < 133))
        x_data_list.append(heating_rate_sw.data)
        y_data_list.append(heating_rate_sw.coord('air_pressure').points / 10**2)
    ax_lin, ax_log = plot_data_on_mixed_linear_log_scale(fig, ax, x_data_list, y_data_list, ['Double precision', 'Mixed precision'], 
                                                         ylabel='Pressure (hPa)', xscale='linear',
                                                         ylim_linear=(1000, 101), ylim_log=(101, 0.007),
                                                         ylabel_shirt=0.02)
    cloud_fraction = e_double.ecrad_input_as_iris_cube('cloud_fraction').extract(constraint=iris.Constraint(latitude=lambda l: -5 < l < -2,
                                                                                                            longitude=lambda l: 130 < l < 133))
    min_x, max_x = ax_lin.get_xlim()
    for ax_ in (ax_lin, ax_log):
        ax_.fill_betweenx(cloud_fraction.coord('air_pressure').points, x1=min_x, x2=cloud_fraction.data * 10 + min_x, color='#aaa')
        #ax_.plot(cloud_fraction.data * 10 + min_x, cloud_fraction.coord('air_pressure').points, 'k--')
        ax_.set_xlim((min_x, max_x))
        #ax.plot(heating_rate_lw.data, heating_rate_lw.coord('air_pressure').points)
#        qplt.plot(res)
#        ax.set_title('Longwave heating rate' if i == 0 else '')
#        ax.set_ylabel('')
#        ax = plt.subplot(2, 3, 3*i + 3)
#        res = e.ecrad_output_as_iris_cube('heating_rate_sw').extract(constraint=iris.Constraint(latitude=lambda l: -5 < l < -2,
#                                                                                            longitude=lambda l: 130 < l < 133))
#        qplt.plot(res)
#        ax.set_title('Shortwave heating rate' if i == 0 else '')
#        ax.set_ylabel('')
    #    ax.set_ylim((-10, 30))
    ax.set_xlabel(r'Shortwave heating rate (K $\times$ d$^{-1}$)')
    #ax.grid()
    ax_log.legend(loc='upper left', fontsize=16)
    plt.tight_layout(rect=[0.03, 0, 1, 1])
    plt.savefig('erroneous_heating_rates.eps', dpi=200)
    plt.show()