import os
import sys
from dataclasses import dataclass
from typing import Any

from matplotlib import colors
sys.path.append(os.getcwd())
import pickle
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import iris
import iris.plot as iplt

import cartopy.crs as ccrs

import restools
from papers.none2021_ecrad.data import Summary
from papers.none2021_ecrad.extensions import IfsIO, extract_or_interpolate, get_ifs_rel_diff, get_ifs_abs_rel_diff
from comsdk.research import Research
from comsdk.misc import load_from_json, find_all_files_by_named_regexp
from thequickmath.reduced_models.transition_to_turbulence import MoehlisFaisstEckhardtModel


@dataclass
class EcradRunInfo:
    name: str
    label: str


@dataclass
class FieldInfo:
    name: str
    label: str
    units: str
    pressure: str = None
    vmin: float = None
    vmax: float = None
    levels: Any = None
    cf: Any = None


if __name__ == '__main__':
    plt.style.use('resources/default.mplstyle')

    summary = load_from_json(Summary)
    res_id = summary.res_id
    res = Research.open(res_id)
    task_path = res.get_task_path(summary.task_for_oifs_results)
    #step_shifts = (0, 64, 128, 192, 256, 320)
    step_shifts = (0, 192, 320)
    ts_i = 2
    ecrad_runs = (
        EcradRunInfo(name='ecrad_tripleclouds_52bits', label='Tripleclouds (52 sbits)'),
        EcradRunInfo(name='ecrad_mcica_52bits', label='McICA (52 sbits)'),
        EcradRunInfo(name='ecrad_tripleclouds_23bits', label='Tripleclouds (23 sbits)'),
        EcradRunInfo(name='ecrad_tripleclouds_mixed_precision', label='Tripleclouds (mixed precision)'),
    )
    fields_to_plot = (
        FieldInfo(name='geopotential_height',
                  label='Geopotential height',
                  units='m',
                  pressure=500.),
        FieldInfo(name='temperature_at_2m',
                  label='2-m temperature',
                  units='K'),
        FieldInfo(name='surface_downwards_shortwave_radiation',
                  label=r'\begin{center}Surface downwelling\\shortwave radiation\end{center}',
                  units=r'J m$^{-2}$'),
        FieldInfo(name='surface_downwards_longwave_radiation',
                  label=r'\begin{center}Surface downwelling\\longwave radiation\end{center}',
                  units=r'J m$^{-2}$'),
    )
    n_timesteps = len(step_shifts)
    n_runs = len(ecrad_runs)
    n_fields = len(fields_to_plot)
    #relative_error_level = 10**(-2)
    relative_error_level = 10**(-1)

    ifs_io_ref = IfsIO([os.path.join(task_path, 'hgom', ecrad_runs[0].name, 'sh', f'{id_}.nc') for id_ in step_shifts],
                       [os.path.join(task_path, 'hgom', ecrad_runs[0].name, 'gg', f'{id_}.nc') for id_ in step_shifts],
                       l91_file=summary.l91_file)
    #fig, axes = plt.subplots(n_fields, n_runs, figsize=(12, 6))
    fig = plt.figure(figsize=(12, 7))
    for run_i in range(n_runs):
        ifs_io = IfsIO([os.path.join(task_path, 'hgom', ecrad_runs[run_i].name, 'sh', f'{id_}.nc') for id_ in step_shifts],
                       [os.path.join(task_path, 'hgom', ecrad_runs[run_i].name, 'gg', f'{id_}.nc') for id_ in step_shifts],
                        l91_file=summary.l91_file)
        for field_i in range(n_fields):
            q = getattr(ifs_io, fields_to_plot[field_i].name)(ts_i)
            if fields_to_plot[field_i].pressure is not None:
                q = extract_or_interpolate(q, fields_to_plot[field_i].pressure)
            cs = q.coord_system("CoordSystem")
            q_projection = iplt.default_projection(q)
            ax = plt.subplot(n_fields, n_runs, 1 + field_i*n_fields + (run_i - int(run_i//n_runs)*n_runs), projection=q_projection)
            if run_i == 0:
                cf = iplt.contourf(q, 16, cmap=plt.get_cmap('coolwarm'), coords=['longitude', 'latitude'], axes=ax)
            else:
                q_ref = getattr(ifs_io_ref, fields_to_plot[field_i].name)(ts_i)
                if fields_to_plot[field_i].pressure is not None:
                    q_ref = extract_or_interpolate(q_ref, fields_to_plot[field_i].pressure)
                #divnorm = colors.TwoSlopeNorm(vcenter=0.)
                #cf = iplt.contourf(q_ref - q, 16, cmap=plt.get_cmap('bwr'), norm=divnorm, 
                #                       coords=['longitude', 'latitude'], axes=ax)
                divnorm = colors.TwoSlopeNorm(vcenter=0.)
                if fields_to_plot[field_i].vmin is None:
                    cf = iplt.contourf(q_ref - q, 16, cmap=plt.get_cmap('bwr'), 
                                       norm=divnorm,
                                       coords=['longitude', 'latitude'],
                                       axes=ax)
                    fields_to_plot[field_i].vmin = cf.zmin
                    fields_to_plot[field_i].vmax = cf.zmax
                    fields_to_plot[field_i].levels = cf.levels
                    fields_to_plot[field_i].cf = cf
                else:
                    cf = iplt.contourf(q_ref - q, 16, cmap=plt.get_cmap('bwr'), 
                                       vmin=fields_to_plot[field_i].vmin, 
                                       vmax=fields_to_plot[field_i].vmax, 
                                       levels=fields_to_plot[field_i].levels, 
                                       norm=divnorm,
                                       coords=['longitude', 'latitude'],
                                       axes=ax)

            # Add a contour, and put the result in a variable called contour.
            #q_rel_diff = get_ifs_abs_rel_diff(ifs_io_ref, ifs_io, ts_i, ecrad_runs[run_i].name, quantity=fields_to_plot[field_i].name, 
            #                                  pressure=fields_to_plot[field_i].pressure)
            #q_rel_diff.data[np.isnan(q_rel_diff.data)] = 0.
            #iplt.contour(q_rel_diff, levels=np.array([relative_error_level], dtype=np.float64), colors=['#00ff00'], axes=ax)
            # JUST COMMENTED THIS (ERROR): iplt.xticks([range()])
            #plt.gca().coastlines()
            ax.coastlines()
            if field_i == 0:
                #plt.gca().set_title(ecrad_runs[run_i].label, usetex=False, fontsize=12)
                ax.set_title(ecrad_runs[run_i].label, usetex=False, fontsize=12)
#            if run_i == 0:
#                plt.gca().text(-0.1, 0.5, f'+{ifs_io.time_shift(ts_i)}', transform=plt.gca().transAxes,
#                                va='center', fontsize=12, rotation='vertical')
            if run_i == 0:
                #plt.gca().text(-0.1, 0.5, f'{ fields_to_plot[field_i].label}', transform=plt.gca().transAxes,
                ax.text(-0.2, 0.5, f'{ fields_to_plot[field_i].label}', transform=plt.gca().transAxes,
                                va='center', fontsize=12, rotation='vertical')
#        cbar = plt.gca().colorbar()
#        cbar.set_ticks([min_value, 0., max_value])

    plt.tight_layout(rect=[0, 0, 0.9, 1], h_pad=0.03, w_pad=1.0)

    for field_i, ax in enumerate(plt.gcf().axes[-4:]):
        base_rect = list(ax.get_position().bounds)
        base_rect[0] += base_rect[2] + 0.02  # shift along the x-coord
        base_rect[2] = 0.015  # set tiny x-extent
        colorbar_axes = plt.gcf().add_axes(base_rect)
        colorbar = plt.colorbar(fields_to_plot[field_i].cf, colorbar_axes)
        colorbar.locator = matplotlib.ticker.MaxNLocator(3)
        colorbar.update_ticks()
        colorbar.set_label(fields_to_plot[field_i].units) #, rotation=270.)
                           #labelpad=8.0,
                           #rotation='horizontal')
    #colorbar_axes = plt.gcf().add_axes([0.9, 0.9, 0.05, 0.2])
    #colorbar = plt.colorbar(fields_to_plot[0].cf, colorbar_axes)
    #colorbar.locator = matplotlib.ticker.MaxNLocator(3)
    #colorbar.update_ticks()

    #plt.tight_layout(rect=[0, 0.1, 1, 1], h_pad=0.05, w_pad=1.0)
    plt.savefig(f'fields_comparison_and_rel_error.png', dpi=200)
    plt.show()
