import json
import os
from functools import partial
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Literal, Sequence
from iris import coord_systems

import numpy as np
import pandas as pd
import netCDF4
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import seaborn as sns
import iris
from iris.analysis.calculus import differentiate
import iris.plot as iplt
import iris.quickplot as qplt
import cf_units

from comsdk.misc import find_all_files_by_named_regexp

tripleclouds_truth_path = '/network/aopp/chaos/pred/pershin/ecrad/outputs/tripleclouds/52bits'
tripleclouds_all_path = '/network/aopp/chaos/pred/pershin/ecrad/outputs/tripleclouds'
mcica_truth_path = '/network/aopp/chaos/pred/pershin/ecrad/outputs/mcica/52bits'
mcica_all_path = '/network/aopp/chaos/pred/pershin/ecrad/outputs/mcica'
test_truth_path = '/network/aopp/chaos/pred/pershin/ecrad/outputs/test/52bits'
test_all_path = '/network/aopp/chaos/pred/pershin/ecrad/outputs/test'
ecrad_input_path = '/network/aopp/chaos/pred/shared/ecRad_data'


@dataclass
class VariableData:
    n_ops: int = 0
    bins: List[int] = field(default_factory=lambda: [0 for _ in range(11)])
    max: float = 0.
    min: float = np.inf
    mean: float = 0.


class EcradIO:
    def __init__(self, input_nc_file=None, output_nc_file=None, convert_columns_to_latitude_and_longitude=True,
                 time=None, l137_file='L137.csv'):
        self.input_nc_dataset = None if input_nc_file is None else \
            netCDF4.Dataset(input_nc_file, 'r', format='NETCDF4')
        self.output_nc_dataset = None if output_nc_file is None else \
            netCDF4.Dataset(output_nc_file, 'r', format='NETCDF4')
        self.time = time
        self.convert_columns_to_latitude_and_longitude = convert_columns_to_latitude_and_longitude
        half_pressure = np.r_[[0.], pd.read_csv(l137_file).ph.to_numpy()]
        full_pressure = pd.read_csv(l137_file).pf.to_numpy()
        self.iris_full_pressure = iris.coords.DimCoord(full_pressure,
                                                       standard_name='air_pressure', units='hPa')
        self.iris_half_pressure = iris.coords.DimCoord(half_pressure,
                                                       standard_name='air_pressure', units='hPa')
        self.iris_interface_pressure = iris.coords.DimCoord(full_pressure[:-1],
                                                            standard_name='air_pressure', units='hPa')
        if self.convert_columns_to_latitude_and_longitude:
            self.iris_latitude = iris.coords.DimCoord(np.array(self.input_nc_dataset['latitude']).reshape((61, 120))[:, 0],
                                                      standard_name='latitude', units='degree_north')
            self.iris_longitude = iris.coords.DimCoord(np.array(self.input_nc_dataset['longitude']).reshape((61, 120))[0, :],
                                                      standard_name='longitude', units='degree_east', circular=True)

    def list_input_variables(self):
        return list(self.input_nc_dataset.variables.keys())

    def list_output_variables(self):
        return list(self.output_nc_dataset.variables.keys())

    def ecrad_input_as_iris_cube(self, var):
        if var not in self.input_nc_dataset.variables.keys():
            raise ValueError(f'Variable {var} not found in input dataset')
        return self._ecrad_var_as_iris_cube(var, self.input_nc_dataset)

    def ecrad_output_as_iris_cube(self, var):
        if var == 'flux_net_lw':
            return self._flux_net_as_iris_cube(type='lw')
        elif var == 'flux_net_sw':
            return self._flux_net_as_iris_cube(type='sw')
        elif var == 'cloud_radiative_effect_lw':
            return self._cloud_radiative_effect_as_iris_cube(type='lw')
        elif var == 'cloud_radiative_effect_sw':
            return self._cloud_radiative_effect_as_iris_cube(type='sw')
        elif var == 'heating_rate_lw':
            return self._heating_rate_as_iris_cube(type='lw')
        elif var == 'heating_rate_sw':
            return self._heating_rate_as_iris_cube(type='sw')
        if var not in self.output_nc_dataset.variables.keys():
            raise ValueError(f'Variable {var} not found in output dataset')
        return self._ecrad_var_as_iris_cube(var, self.output_nc_dataset)

    def _ecrad_var_as_iris_cube(self, var, original_dataset, use_Pa=False):
        if len(original_dataset[var].dimensions) == 0:
            raise ValueError(f'Variable {var} is scalar. No need to use iris cube')
        new_shape = []
        dim_coords_and_dims = []

        def convert_to_Pa_if_necessary(pressure_as_iris_coord):
            if use_Pa:
                iris_coord = pressure_as_iris_coord.copy()
                iris_coord.convert_units('Pa')
                return iris_coord
            else:
                return pressure_as_iris_coord

        for i, dim_name in enumerate(original_dataset[var].dimensions):
            if dim_name == 'column':
                if self.convert_columns_to_latitude_and_longitude:
                    dim_coords_and_dims += [(self.iris_latitude, len(new_shape)), (self.iris_longitude, len(new_shape) + 1)]
                    new_shape += [61, 120]
                else:
                    s = original_dataset.dimensions[dim_name].size
                    iris_fake_latitude = iris.coords.DimCoord(np.linspace(0, 180, s),
                                                              standard_name='latitude', units='degree_north')
                    dim_coords_and_dims.append((iris_fake_latitude, len(new_shape)))
                    new_shape.append(s)
            elif dim_name == 'level':
                iris_coord = convert_to_Pa_if_necessary(self.iris_full_pressure)
                dim_coords_and_dims.append((iris_coord, len(new_shape)))
                new_shape.append(137)
            elif dim_name == 'half_level':
                iris_coord = convert_to_Pa_if_necessary(self.iris_half_pressure)
                dim_coords_and_dims.append((iris_coord, len(new_shape)))
                new_shape.append(138)
            elif dim_name == 'level_interface':
                iris_coord = convert_to_Pa_if_necessary(self.iris_interface_pressure)
                dim_coords_and_dims.append((iris_coord, len(new_shape)))
                new_shape.append(136)
        var_as_np_array = np.array(original_dataset[var]).astype(np.float64).reshape(tuple(new_shape))
        cube = iris.cube.Cube(var_as_np_array, dim_coords_and_dims=dim_coords_and_dims)
        if self.time is not None:
            time_coord = iris.coords.DimCoord(self.time, standard_name='time', units='hours since 1970-01-01 00:00:00')
            cube.add_aux_coord(time_coord)
        return cube

    def _flux_net_as_iris_cube(self, type='lw', use_Pa=False):
        if f'flux_net_{type}' in self.output_nc_dataset.variables:
            return self._ecrad_var_as_iris_cube(f'flux_net_{type}', self.output_nc_dataset, use_Pa=use_Pa)
        else:
            flux_dn_cube = self._ecrad_var_as_iris_cube(f'flux_dn_{type}', self.output_nc_dataset, use_Pa=use_Pa)
            flux_up_cube = self._ecrad_var_as_iris_cube(f'flux_up_{type}', self.output_nc_dataset, use_Pa=use_Pa)
            return flux_dn_cube - flux_up_cube

    def _cloud_radiative_effect_as_iris_cube(self, type='lw'):
        flux_dn_cube = self._ecrad_var_as_iris_cube(f'flux_dn_{type}', self.output_nc_dataset)
        flux_dn_clear_cube = self._ecrad_var_as_iris_cube(f'flux_dn_{type}_clear', self.output_nc_dataset)
        flux_up_cube = self._ecrad_var_as_iris_cube(f'flux_up_{type}', self.output_nc_dataset)
        flux_up_clear_cube = self._ecrad_var_as_iris_cube(f'flux_up_{type}_clear', self.output_nc_dataset)
        return (flux_dn_cube - flux_dn_clear_cube) - (flux_up_cube - flux_up_clear_cube)

    def _heating_rate_as_iris_cube(self, type='lw'):
        c = 24*3600*(9.81/1004.)
        flux_net_cube = self._flux_net_as_iris_cube(type=type, use_Pa=True)
        return -1*c*differentiate(flux_net_cube, 'air_pressure')


class IfsIO:
    def __init__(self, time_series_sh_files: Sequence[str], time_series_gg_files: Sequence[str], l91_file='L91.csv'):
        self.n_time_steps = len(time_series_sh_files)
        self.time_series_sh_datasets = [netCDF4.Dataset(filename, 'r', format='NETCDF4')
                                        for filename in time_series_sh_files]
        self.time_series_gg_datasets = [netCDF4.Dataset(filename, 'r', format='NETCDF4')
                                        for filename in time_series_gg_files]
        pressure_full_levels = pd.read_csv(l91_file).pf.to_numpy()
        pressure_full_levels = pressure_full_levels[~np.isnan(pressure_full_levels)]
        self.iris_pressure_levels = {
            'plev': iris.coords.DimCoord(np.array(self.time_series_sh_datasets[0]['plev']),
                                         standard_name='air_pressure', units='Pa'),
            'lev_2': iris.coords.DimCoord(pressure_full_levels,
                                          standard_name='air_pressure', units='hPa'),
        }
        for _, dimcoord in self.iris_pressure_levels.items():
            dimcoord.convert_units('hPa')
        cs = iris.coord_systems.GeogCS(6371229)
        self.iris_latitude = iris.coords.DimCoord(np.array(self.time_series_sh_datasets[0]['lat']),
                                                  standard_name='latitude', units='degree_north',
                                                  coord_system=cs)
        self.iris_longitude = iris.coords.DimCoord(np.array(self.time_series_sh_datasets[0]['lon']),
                                                   standard_name='longitude', units='degree_east', circular=True,
                                                   coord_system=cs)

    def __len__(self):
        return len(self.time_series_sh_datasets)

    def times(self) -> Sequence[int]:
        return [int(ds['time'][0]) for ds in self.time_series_sh_datasets]

    def time_shift(self, time_step_i) -> timedelta:
        initial_ds = self.time_series_sh_datasets[0]
        ds = self.time_series_sh_datasets[time_step_i]
        initial_datetime = cf_units.num2date(float(initial_ds['time'][0]), 'hours since 1970-01-01 00:00:00', cf_units.CALENDAR_STANDARD)
        current_datetime = cf_units.num2date(float(ds['time'][0]), 'hours since 1970-01-01 00:00:00', cf_units.CALENDAR_STANDARD)
        return current_datetime - initial_datetime

    def geopotential(self, time_step_i):
        return self._ifs_var_as_iris_cube('Z', 'geopotential', self.time_series_sh_datasets[time_step_i], units='m2 s-2')

    def geopotential_height(self, time_step_i):
        geopotential_cube = self.geopotential(time_step_i)
        g = iris.coords.AuxCoord(9.80665,
                          long_name='gravitational acceleration',
                          units='m s-2')
        geopotential_height_cube = geopotential_cube / g
        geopotential_height_cube.rename('geopotential height')
        return geopotential_height_cube

    def temperature(self, time_step_i):
        return self._ifs_var_as_iris_cube('T', 'air_temperature', self.time_series_sh_datasets[time_step_i])

    def temperature_at_2m(self, time_step_i):
        return self._ifs_var_as_iris_cube('T2M', 'air_temperature', self.time_series_gg_datasets[time_step_i])

    def u_wind(self, time_step_i):
        return self._ifs_var_as_iris_cube('u', 'x_wind', self.time_series_sh_datasets[time_step_i])

    def v_wind(self, time_step_i):
        return self._ifs_var_as_iris_cube('v', 'y_wind', self.time_series_sh_datasets[time_step_i])

    def w_wind(self, time_step_i):
        return self._ifs_var_as_iris_cube('W', 'upward_air_velocity', self.time_series_sh_datasets[time_step_i])

    def surface_pressure(self, time_step_i):
        return self._ifs_var_as_iris_cube('lnsp', 'surface_air_pressure', self.time_series_sh_datasets[time_step_i], units='Pa')

    def surface_downwards_shortwave_radiation(self, time_step_i):
        #'SSRD', J m-2
        return self._ifs_var_as_iris_cube('SSRD', None, self.time_series_gg_datasets[time_step_i], units='J m-2')
    
    def surface_downwards_longwave_radiation(self, time_step_i):
        # 'STRD', J m-2
        return self._ifs_var_as_iris_cube('STRD', None, self.time_series_gg_datasets[time_step_i], units='J m-2')

    def _ifs_var_as_iris_cube(self, var, standard_name, original_dataset, units=None):
        if len(original_dataset[var].dimensions) == 0:
            raise ValueError(f'Variable {var} is scalar. No need to use iris cube')
        new_shape = []
        dim_coords_and_dims = []
        time = None
        surface = False

        for i, dim_name in enumerate(original_dataset[var].dimensions):
            if dim_name == 'plev':
                dim_coords_and_dims.append((self.iris_pressure_levels['plev'], len(dim_coords_and_dims)))
            elif dim_name == 'lev_2':
                dim_coords_and_dims.append((self.iris_pressure_levels['lev_2'], len(dim_coords_and_dims)))
            elif dim_name == 'lev_3':
                surface = True
            elif dim_name == 'lat':
                dim_coords_and_dims.append((self.iris_latitude, len(dim_coords_and_dims)))
            elif dim_name == 'lon':
                dim_coords_and_dims.append((self.iris_longitude, len(dim_coords_and_dims)))
            elif dim_name == 'time':
                time = float(original_dataset['time'][0])
            else:
                raise ValueError(f'Unknown dimension: {dim_name}')
        if surface:
            var_as_np_array = np.array(original_dataset[var])[0, 0, ...]
        else:
            var_as_np_array = np.array(original_dataset[var])[0, ...]
        if var == 'lnsp':
            var_as_np_array = np.exp(var_as_np_array)
        cube = iris.cube.Cube(var_as_np_array, standard_name=standard_name, dim_coords_and_dims=dim_coords_and_dims, units=units)
        if var == 'lnsp':
            cube.convert_units('hPa')
        if time is not None:
            time_coord = iris.coords.DimCoord(time, standard_name='time', units='hours since 1970-01-01 00:00:00')
            cube.add_aux_coord(time_coord)
        return cube


class ERA5Data:
    def __init__(self, timed_nc_file: str, any_ref_nc_file_containing_same_lat_and_lon=None, ref_lat_lon_naming='long',
                 shift_time_to_zero=True):
        self.nc_dataset = netCDF4.Dataset(timed_nc_file, 'r', format='NETCDF4')
        self.n_time_steps = len(self.nc_dataset['time'])
        if any_ref_nc_file_containing_same_lat_and_lon is not None:
            self.any_ref_nc_dataset_containing_same_lat_and_lon = netCDF4.Dataset(any_ref_nc_file_containing_same_lat_and_lon, 'r', format='NETCDF4')
        else:
            self.any_ref_nc_dataset_containing_same_lat_and_lon = None
        self.ref_lat_lon_naming = ref_lat_lon_naming
        self.shift_time_to_zero = shift_time_to_zero

#        self.n_time_steps = len(time_series_sh_files)
#        self.time_series_sh_datasets = [netCDF4.Dataset(filename, 'r', format='NETCDF4')
#                                        for filename in time_series_sh_files]
#        self.time_series_gg_datasets = [netCDF4.Dataset(filename, 'r', format='NETCDF4')
#                                        for filename in time_series_gg_files]
#        pressure_full_levels = pd.read_csv(l91_file).pf.to_numpy()
#        pressure_full_levels = pressure_full_levels[~np.isnan(pressure_full_levels)]
#        self.iris_pressure_levels = {
#            'plev': iris.coords.DimCoord(np.array(self.time_series_sh_datasets[0]['plev']),
#                                         standard_name='air_pressure', units='Pa'),
#            'lev_2': iris.coords.DimCoord(pressure_full_levels,
#                                          standard_name='air_pressure', units='hPa'),
#        }
#        for _, dimcoord in self.iris_pressure_levels.items():
#            dimcoord.convert_units('hPa')
#        self.iris_latitude = iris.coords.DimCoord(np.array(self.time_series_sh_datasets[0]['lat']),
#                                                  standard_name='latitude', units='degree_north')
#        self.iris_longitude = iris.coords.DimCoord(np.array(self.time_series_sh_datasets[0]['lon']),
#                                                   standard_name='longitude', units='degree_east', circular=True)

    def __len__(self):
        return len(self.n_time_steps)

    def times(self) -> Sequence[int]:
        times = np.array(self.nc_dataset['time'])
        if self.shift_time_to_zero:
            times -= self.nc_dataset['time'][0]
        return list(times)

    def time_shift(self, time_step_i) -> timedelta:
        initial_datetime = cf_units.num2date(float(self.nc_dataset['time'][0]), 'hours since 1970-01-01 00:00:00', cf_units.CALENDAR_STANDARD)
        current_datetime = cf_units.num2date(float(self.nc_dataset['time'][time_step_i]), 'hours since 1970-01-01 00:00:00', cf_units.CALENDAR_STANDARD)
        return current_datetime - initial_datetime

    def temperature(self, time_step_i):
        return self._ifs_var_as_iris_cube('t', 'air_temperature', time_step_i)

    def temperature_at_2m(self, time_step_i):
        return self._ifs_var_as_iris_cube('t2m', 'air_temperature', time_step_i)

    def _ifs_var_as_iris_cube(self, var, standard_name, time_step_i):
        dim_coords_and_dims = []
        #standard_name = 'air_temperature'
        units = None
        #var = 't'
        time = None
        surface = False
        nc_dataset_to_take_lat_and_lon_from = self.any_ref_nc_dataset_containing_same_lat_and_lon if self.any_ref_nc_dataset_containing_same_lat_and_lon is not None else self.nc_dataset
        lat_name = 'latitude' if self.ref_lat_lon_naming == 'long' else 'lat'
        lon_name = 'longitude' if self.ref_lat_lon_naming == 'long' else 'lon'
        cs = iris.coord_systems.GeogCS(6371229)
        for i, dim_name in enumerate(self.nc_dataset[var].dimensions):
            if dim_name == 'level':
                dim_coords_and_dims.append((iris.coords.DimCoord(np.array(self.nc_dataset['level']),
                                                standard_name='air_pressure', units='hPa'),
                                            len(dim_coords_and_dims)))
            elif dim_name == 'latitude':
                dim_coords_and_dims.append((iris.coords.DimCoord(np.array(nc_dataset_to_take_lat_and_lon_from[lat_name]),
                                                standard_name='latitude', units='degree_north', coord_system=cs), 
                                            len(dim_coords_and_dims)))
            elif dim_name == 'longitude':
                dim_coords_and_dims.append((iris.coords.DimCoord(np.array(nc_dataset_to_take_lat_and_lon_from[lon_name]),
                                                standard_name='longitude', units='degree_east', circular=True, coord_system=cs), 
                                            len(dim_coords_and_dims)))
            elif dim_name == 'time':
                time = float(self.nc_dataset['time'][time_step_i])
                if self.shift_time_to_zero:
                    time -= float(self.nc_dataset['time'][0])
            else:
                raise ValueError(f'Unknown dimension: {dim_name}')
        var_as_np_array = np.array(self.nc_dataset[var])[time_step_i, ...]
        cube = iris.cube.Cube(var_as_np_array, standard_name=standard_name, dim_coords_and_dims=dim_coords_and_dims, units=units)
        if var == 'lnsp':
            cube.convert_units('hPa')
        if time is not None:
            time_coord = iris.coords.DimCoord(time, standard_name='time', units='hours since 1970-01-01 00:00:00')
            cube.add_aux_coord(time_coord)
        return cube

#    def time_shift(self, time_step_i) -> timedelta:
#        initial_ds = self.time_series_sh_datasets[0]
#        ds = self.time_series_sh_datasets[time_step_i]
#        initial_datetime = cf_units.num2date(float(initial_ds['time'][0]), 'hours since 1970-01-01 00:00:00', cf_units.CALENDAR_STANDARD)
#        current_datetime = cf_units.num2date(float(ds['time'][0]), 'hours since 1970-01-01 00:00:00', cf_units.CALENDAR_STANDARD)
#        return current_datetime - initial_datetime



class ComparisonResults:
    def __init__(self, data_path_with_labels_to_be_compared, handlers_with_labels):
        self.handlers_with_labels = handlers_with_labels
        self.data_path_with_labels_to_be_compared = data_path_with_labels_to_be_compared
        self.results = {}

    def set_result(self, data_label, handler_label, obj):
        if data_label not in self.results:
            self.results[data_label] = {handler_label: obj}
        else:
            self.results[data_label][handler_label] = obj

    def get_result(self, data_label, handler_label):
        return self.results[data_label][handler_label]

    def available_handlers(self):
        return list(self.handlers_with_labels.keys())

    def available_data(self):
        return list(self.data_path_with_labels_to_be_compared.keys())

    def data_path(self, label):
        return self.data_path_with_labels_to_be_compared[label]

    def handler(self, label):
        return self.handlers_with_labels[label]

    def init_results(self, cls_initialiser):
        for d in self.available_data():
            for h in self.available_handlers():
                self.set_result(d, h, cls_initialiser())

    def subset(self, dt: datetime):
        subset_car = ComparisonResults(self.data_path_with_labels_to_be_compared, self.handlers_with_labels)
        for d in self.available_data():
            for h in self.available_handlers():
                res = self.get_result(d, h)
                subset_car.set_result(d, h, self.get_result(d, h).extract(iris.Constraint(time=lambda cell: cell.point == dt)))
        return subset_car


def load_simulations_with_respect_reference(comp_against_ref, reference_path, input_path, collapse_time_axis_through_rms=False, collapse_time_axis_through_mean=False, debug=False, time_bounds=None):
    """
    Reference path contains a set of ecRad outputs for reference simulations (usually those with double precision)
    Low precision paths is a list of paths each of which contains ecRad outputs for a given precision
    Handlers is a list of functions (EcradIO, EcradIO) -> whatever.
    Returns a list with rank (len(low_prec_paths), len(handlers), # of outputs)
    """
    comp_against_ref.init_results(iris.cube.CubeList)
    year = 2001
    months = range(1, 12+1)
    days = range(1, 31+1)
    hours = [0, 6, 12, 18]

    for m in months:
        if debug:
            print(f'Process month #{m}')
        for d in days:
            for h in hours:
                try:
                    dt = datetime(year, m, d, h, 0, 0)
                except ValueError as e:
                    if debug:
                        print(f'Ignore year {year}, month {m}, day {d}, hour {h} for the reason "{e}"')
                    continue
                if time_bounds is not None:
                    if not (time_bounds[0] <= dt <= time_bounds[1]):
                        continue
                dt_str = dt.strftime('%Y-%m-%d-%H')
                input_nc_file = os.path.join(input_path, f'era5_{dt_str}.nc')
                output_nc_file_ref = os.path.join(reference_path, f'era5_{dt_str}_output.nc')
                if not os.path.exists(output_nc_file_ref):
                    continue
                e_ref = EcradIO(input_nc_file=input_nc_file,
                                output_nc_file=output_nc_file_ref,
                                time=cf_units.date2num(dt, 'hours since 1970-01-01 00:00:00',
                                                       cf_units.CALENDAR_STANDARD))

                for data_label in comp_against_ref.available_data():
                    comp_against_ref.data_path(data_label)
                    output_nc_file_low_prec = os.path.join(comp_against_ref.data_path(data_label), f'era5_{dt_str}_output.nc')
                    e_low_prec = EcradIO(input_nc_file=input_nc_file,
                                         output_nc_file=output_nc_file_low_prec,
                                         time=cf_units.date2num(dt, 'hours since 1970-01-01 00:00:00',
                                                                cf_units.CALENDAR_STANDARD))
                    for handler_label in comp_against_ref.available_handlers():
                        comp_against_ref.get_result(data_label, handler_label).append(comp_against_ref.handler(handler_label)(e_ref, e_low_prec))
    for data_label in comp_against_ref.available_data():
        for handler_label in comp_against_ref.available_handlers():
            if collapse_time_axis_through_rms:
                comp_against_ref.set_result(data_label, handler_label, comp_against_ref.get_result(data_label, handler_label).merge_cube().collapsed(['time'], iris.analysis.RMS))
            elif collapse_time_axis_through_mean:
                comp_against_ref.set_result(data_label, handler_label, comp_against_ref.get_result(data_label, handler_label).merge_cube().collapsed(['time'], iris.analysis.MEAN))
            else:
                comp_against_ref.set_result(data_label, handler_label, comp_against_ref.get_result(data_label, handler_label).merge_cube())


def load_simulations_at_fixed_time(comp_against_ref, input_path, dt, debug=False):
    dt_str = dt.strftime('%Y-%m-%d-%H')
    for data_label in comp_against_ref.available_data():
        comp_against_ref.data_path(data_label)
        input_nc_file = os.path.join(input_path, f'era5_{dt_str}_output.nc')
        output_nc_file_low_prec = os.path.join(comp_against_ref.data_path(data_label), f'era5_{dt_str}_output.nc')
        e_low_prec = EcradIO(input_nc_file=input_nc_file,
                             output_nc_file=output_nc_file_low_prec,
                             time=cf_units.date2num(dt, 'hours since 1970-01-01 00:00:00',
                                                    cf_units.CALENDAR_STANDARD))
        for handler_label in comp_against_ref.available_handlers():
            comp_against_ref.set_result(data_label, handler_label, comp_against_ref.handler(handler_label)(e_low_prec))


def _build_coord_list(cube, keep_latitude, keep_longitude, keep_air_pressure):
    avail_coords = [c.name() for c in cube.coords()]

    coords = []
    if not keep_latitude and 'latitude' in avail_coords:
        coords.append('latitude')
    if not keep_longitude and 'longitude' in avail_coords:
        coords.append('longitude')
    if not keep_air_pressure and 'air_pressure' in avail_coords:
        coords.append('air_pressure')
    return coords


def heating_rate_mean(e, type='lw', keep_latitude=False, keep_longitude=False, keep_air_pressure=False):
    diff_cube = e.ecrad_output_as_iris_cube(f'heating_rate_{type}')
    coords_to_mean_about = _build_coord_list(diff_cube, keep_latitude, keep_longitude, keep_air_pressure)
    return diff_cube.collapsed(coords_to_mean_about, iris.analysis.MEAN) if len(coords_to_mean_about) != 0 else diff_cube


def heating_rate_diff_mean(e_ref, e, type='lw', keep_latitude=False, keep_longitude=False, keep_air_pressure=False):
    diff_cube = e_ref.ecrad_output_as_iris_cube(f'heating_rate_{type}') - e.ecrad_output_as_iris_cube(f'heating_rate_{type}')
    coords_to_mean_about = _build_coord_list(diff_cube, keep_latitude, keep_longitude, keep_air_pressure)
    return diff_cube.collapsed(coords_to_mean_about, iris.analysis.MEAN) if len(coords_to_mean_about) != 0 else diff_cube


def heating_rate_diff_rms(e_ref, e, type='lw', keep_latitude=False, keep_longitude=False, keep_air_pressure=False):
    diff_cube = e_ref.ecrad_output_as_iris_cube(f'heating_rate_{type}') - e.ecrad_output_as_iris_cube(f'heating_rate_{type}')
    coords_to_rms_about = _build_coord_list(diff_cube, keep_latitude, keep_longitude, keep_air_pressure)
    return diff_cube.collapsed(coords_to_rms_about, iris.analysis.RMS) if len(coords_to_rms_about) != 0 else diff_cube


def flux_net_diff_rms(e_ref, e, type='lw', keep_latitude=False, keep_longitude=False, keep_air_pressure=False):
    diff_cube = e_ref.ecrad_output_as_iris_cube(f'flux_net_{type}') - e.ecrad_output_as_iris_cube(f'flux_net_{type}')
    coords_to_rms_about = _build_coord_list(diff_cube, keep_latitude, keep_longitude, keep_air_pressure)
    return diff_cube.collapsed(coords_to_rms_about, iris.analysis.RMS) if len(coords_to_rms_about) != 0 else diff_cube


def get_profile_at_fixed_latitude(e: EcradIO, lat: float, quantity: str):
    q = e.ecrad_output_as_iris_cube(quantity)
    q_at_fixed_lat = q.extract(iris.Constraint(latitude=lambda x: lat-0.5 < x < lat+0.5))
    return q_at_fixed_lat


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


def turn_to_int(s):
    return int(s.strip())


def turn_to_float(s):
    return float(s.strip())


def collect_variable_data(root_path, group_by: Literal['variable_only', 'variable_and_call_number'],
                          count_zero_assignments=False, be_aware_of_grouping_by_processor=False, debug=False):
    variables = {}
    for var_name in os.listdir(root_path):
        if debug:
            print(f'Reading variable {var_name}')
        variables[var_name] = {}
        var_path = os.path.join(root_path, var_name)
        proc_path_list = [var_path]
        proc_names = ['proc_1']
        if be_aware_of_grouping_by_processor:
            proc_names = os.listdir(var_path)
            proc_path_list = [os.path.join(var_path, proc_dir) for proc_dir in proc_names]
        for proc_name, proc_path in zip(proc_names, proc_path_list):
            variables[var_name][proc_name] = {}
            filename_and_params = find_all_files_by_named_regexp(r'^(?P<num>\d+).csv$', proc_path)
            for filename, params in filename_and_params:
                p = os.path.join(proc_path, filename)
                call_num = params['num']
                with open(p, 'r') as f:
                    lines = list(f)
                    if len(lines) == 0:
                        print(f'Empty file encountered (skip it): {p}')
                        continue
                    variables[var_name][proc_name][call_num] = VariableData()
                    name_to_index_mapping = {name.strip(): i for i, name in enumerate(lines[0].split(','))}
                    values = lines[1].split(',')
                    try:
                        variables[var_name][proc_name][call_num].min = min(variables[var_name][proc_name][call_num].min,
                                                         turn_to_float(values[name_to_index_mapping['min_nonzero_abs']]))
                    except ValueError as e:
                        print(f'Ignore error: {e}')
                        variables[var_name][proc_name][call_num].min = 10**(-20)
                    if variables[var_name][proc_name][call_num].min < 10**(-20):
                        variables[var_name][proc_name][call_num].min = 10**(-20)
                    variables[var_name][proc_name][call_num].max = max(variables[var_name][proc_name][call_num].max,
                                                     turn_to_float(values[name_to_index_mapping['max_abs']]))
                    variables[var_name][proc_name][call_num].mean += turn_to_float(values[name_to_index_mapping['mean_abs']])
                    variables[var_name][proc_name][call_num].n_ops += turn_to_int(values[name_to_index_mapping['n_ops']])
                    if not count_zero_assignments:
                        variables[var_name][proc_name][call_num].n_ops -= turn_to_int(values[name_to_index_mapping[f'bin0']])
                    for i in range(11):
                        variables[var_name][proc_name][call_num].bins[i] += turn_to_int(values[name_to_index_mapping[f'bin{i+1}']])
        if group_by == 'variable_only':
            var_data_for_var_only = VariableData()
            n_files = 0
            for _, proc_data in variables[var_name].items():
                n_files += len(proc_data)
                for _, var_data in proc_data.items():
                    var_data_for_var_only.min = min(var_data_for_var_only.min, var_data.min)
                    var_data_for_var_only.max = max(var_data_for_var_only.max, var_data.max)
                    var_data_for_var_only.mean += var_data.mean
                    var_data_for_var_only.n_ops += var_data.n_ops
                    for i in range(11):
                        var_data_for_var_only.bins[i] += var_data.bins[i]
            var_data_for_var_only.mean /= n_files
            variables[var_name] = var_data_for_var_only
    return variables


def plot_variable_histogram(fig, ax, variable_datum: Dict[str, VariableData], figname=None):
    bin_edges = np.array([10**(-20), 10**(-16), 10**(-7), 10**(-5), 10**(-3), 10**(-1), 10, 10**3, 10**5, 10**7, 10**16, 10**20],
                         dtype=np.float64)
    fake_horizontal_values = np.arange(len(variable_datum) + 1)
    string_horizontal_values = []
    prob_values = []
    min_values = []
    mean_values = []
    max_values = []
    for var_name, var_data in variable_datum.items():
        string_horizontal_values.append(var_name)
        prob_values.append(np.array(var_data.bins) / var_data.n_ops)
        min_values.append(var_data.min)
        mean_values.append(var_data.mean)
        max_values.append(var_data.max)
    prob_values = np.array(prob_values)
    im = ax.pcolormesh(bin_edges, fake_horizontal_values, prob_values, cmap=plt.get_cmap('ocean_r'))
    for lower_edge, min_, mean, max_ in zip(fake_horizontal_values[:-1], min_values, mean_values, max_values):
        ax.plot([min_, min_], [lower_edge + 0.1, lower_edge + 0.9], color='red', linewidth=1)
        ax.plot([max_, max_], [lower_edge + 0.1, lower_edge + 0.9], color='red', linewidth=1)
        ax.plot([mean], [lower_edge + 0.5], 'ro', markersize=6)
    ax.set_xscale('log')
    ax.set_yticks(fake_horizontal_values[:-1] + 0.5)
    ax.set_yticklabels(string_horizontal_values, fontsize=10, usetex=False)
    ax.grid(axis='x')
    if figname is not None:
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(figname, dpi=200)
        plt.show()
    return im


def plot_single_variable_histogram(fig, ax, var_data: VariableData, figname=None):
    bin_edges = np.array([10**(-20), 10**(-16), 10**(-7), 10**(-5), 10**(-3), 10**(-1), 10, 10**3, 10**5, 10**7, 10**16, 10**20],
                         dtype=np.float64)
#    fake_horizontal_values = np.arange(len(variable_datum) + 1)
#    string_horizontal_values = []
    prob_values = np.array(var_data.bins) / var_data.n_ops
#    min_values = []
#    mean_values = []
#    max_values = []
#    for var_name, var_data in variable_datum.items():
#        string_horizontal_values.append(var_name)
#        prob_values.append(np.array(var_data.bins) / var_data.n_ops)
#        min_values.append(var_data.min)
#        mean_values.append(var_data.mean)
#        max_values.append(var_data.max)
#    prob_values = np.array(prob_values)

    for i in range(len(bin_edges)):
        if i == 0:
            ax.plot([bin_edges[i], bin_edges[i]], [0, prob_values[i]], color='tab:blue', linewidth=3)
            ax.plot([bin_edges[i], bin_edges[i + 1]], [prob_values[i], prob_values[i]], color='tab:blue', linewidth=3)
        elif i == len(bin_edges) - 1:
            ax.plot([bin_edges[i], bin_edges[i]], [prob_values[i - 1], 0], color='tab:blue', linewidth=3)
        else:
            ax.plot([bin_edges[i], bin_edges[i]], [prob_values[i - 1], prob_values[i]], color='tab:blue', linewidth=3)
            ax.plot([bin_edges[i], bin_edges[i + 1]], [prob_values[i], prob_values[i]], color='tab:blue', linewidth=3)

    for val, marker, label in zip((var_data.min, var_data.max, var_data.mean), ('>', '<', '^'), ('Min', 'Max', 'Mean')):
        ax.plot([val, val], [0, 0.1], '-', color='tab:red', linewidth=2)
        ax.plot([val], [0.1], marker, color='tab:red', markersize=12, label=label)

#    im = ax.pcolormesh(bin_edges, fake_horizontal_values, prob_values, cmap=plt.get_cmap('ocean_r'))
#    for lower_edge, min_, mean, max_ in zip(fake_horizontal_values[:-1], min_values, mean_values, max_values):
#        ax.plot([min_, min_], [lower_edge + 0.1, lower_edge + 0.9], color='red', linewidth=1)
#        ax.plot([max_, max_], [lower_edge + 0.1, lower_edge + 0.9], color='red', linewidth=1)
#        ax.plot([mean], [lower_edge + 0.5], 'ro', markersize=6)
    ax.set_xscale('log')
    ax.set_xlabel('Buffer values')
    ax.set_ylabel('Probability')
    ax.set_xticks(bin_edges)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
#    ax.set_yticks(fake_horizontal_values[:-1] + 0.5)
#    ax.set_yticklabels(string_horizontal_values, fontsize=10, usetex=False)
    ax.legend()
    ax.grid()
#    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    if figname is not None:
        plt.savefig(figname, dpi=200)
    plt.show()


def extract_or_interpolate(cube, pressure):
    if pressure is None:
        return cube
    subcube = cube.extract(iris.Constraint(air_pressure=pressure))
    if subcube is None:
        subcube = cube.interpolate([('air_pressure', pressure)], 
                                   iris.analysis.Linear())
    return subcube


def get_ifs_diff(ifs_io, ifs_io_rp, ts_i, dir_, quantity='geopotential', pressure=500.):
    if pressure is None:
        q = getattr(ifs_io, quantity)(ts_i)
        q_rp = getattr(ifs_io_rp, quantity)(ts_i)
    q = extract_or_interpolate(getattr(ifs_io, quantity)(ts_i), pressure)
    q_rp = extract_or_interpolate(getattr(ifs_io_rp, quantity)(ts_i), pressure)
    diff = q - q_rp
    max_value = np.max(diff.data)
    min_value = np.min(diff.data)
    print(f'Dir: {dir_}, time shift: +{ifs_io.time_shift(ts_i)}, max: {max_value}, min: {min_value}')
    return diff


def get_ifs_rel_diff(ifs_io, ifs_io_rp, ts_i, dir_, quantity='geopotential', pressure=500.):
    if pressure is None:
        q = getattr(ifs_io, quantity)(ts_i)
        q_rp = getattr(ifs_io_rp, quantity)(ts_i)
    q = extract_or_interpolate(getattr(ifs_io, quantity)(ts_i), pressure)
    q_rp = extract_or_interpolate(getattr(ifs_io_rp, quantity)(ts_i), pressure)
    rel_diff = (q - q_rp) / q
    max_value = np.max(rel_diff.data)
    min_value = np.min(rel_diff.data)
    print(f'Dir: {dir_}, time shift: +{ifs_io.time_shift(ts_i)}, max: {max_value}, min: {min_value}')
    return rel_diff


def get_ifs_abs_rel_diff(ifs_io, ifs_io_rp, ts_i, dir_, quantity='geopotential', pressure=500.):
    if pressure is None:
        q = getattr(ifs_io, quantity)(ts_i)
        q_rp = getattr(ifs_io_rp, quantity)(ts_i)
    q = extract_or_interpolate(getattr(ifs_io, quantity)(ts_i), pressure)
    q_rp = extract_or_interpolate(getattr(ifs_io_rp, quantity)(ts_i), pressure)
    zero_indices = (q.data == 0.)
    rel_diff = iris.analysis.maths.abs((q - q_rp) / q)
    rel_diff.data[zero_indices] = 0.
    max_value = np.max(rel_diff.data)
    min_value = np.min(rel_diff.data)
    print(f'Dir: {dir_}, time shift: +{ifs_io.time_shift(ts_i)}, max: {max_value}, min: {min_value}')
    return rel_diff


def get_ifs_rmse(ifs_io, ifs_io_rp, time, dir_, quantity='geopotential', pressure=500.):
    ts_i = ifs_io.times().index(time)
    ts_rp_i = ifs_io_rp.times().index(time)
    if pressure is None:
        q = getattr(ifs_io, quantity)(ts_i)
        q_rp = getattr(ifs_io_rp, quantity)(ts_rp_i)
    q = extract_or_interpolate(getattr(ifs_io, quantity)(ts_i), pressure)
    q_rp = extract_or_interpolate(getattr(ifs_io_rp, quantity)(ts_rp_i), pressure)
    diff = q - q_rp
    rmse = diff.collapsed(['latitude', 'longitude'], iris.analysis.RMS)
    print(f'Dir: {dir_}, time shift: +{time}, RMSE: {rmse.data}, RMS-averaged original value: {q.collapsed(["latitude", "longitude"], iris.analysis.RMS).data}')
    return rmse

