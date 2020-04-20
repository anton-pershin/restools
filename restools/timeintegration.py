from abc import ABC, abstractmethod
from typing import Callable, Optional, SupportsFloat, Mapping
from typing_extensions import Literal

from thequickmath.aux import index_for_almost_exact_coincidence
from thequickmath.field import *
from comsdk.comaux import parse_datafile, parse_timed_numdatafile, parse_by_named_regexp
from restools.function import Function
from restools.data_access_strategies import free_data_after_access_strategy


class TimeIntegration(ABC):
    """
    Class TimeIntegration is a general representation of the data produced by time-integration. The current
    implementation regards a particular case of 3D fluid simulations. This class wraps up the real data (e.g., *.h5,
    *.nc and *.txt files) and splits into two main data representations: (1) solution (3D) time series, (2) other data
    (possibly, time series) and (3) simulations configuration. The former is a list of full flow
    fields saved at separate instants of time which can be assessed via method
    solution(t). Other data are attributes of the class instance, differentiated by their so-called Data ID, and
    depend on a particular derived class (see docs of the corresponding class). Simulations configuration can be
    accessed via property simulation_configuration. All the data is accessed according to the lazy initialisation
    pattern (i.e., files are loaded only when they are needed). At the same time, whether the data is cached/stored when
    it has been loaded depends on DataAccessStrategy defined for solution fields (solution_access_strategy) and other
    data (other_data_access_strategy). Since Data ID is passed to main strategy methods, one is encouraged to derive
    their own classes implementing different 'caching' strategy based on different Data IDs. Any other data can be
    transformed right after they are loaded through the function transform ('upload' and 'transform' functions are
    essentially glued).

    To create instances of the class, we follow Builder pattern so one is encouraged to use one of the derived
    classes of TimeIntegrationBuilder (or its own builder derived from TimeIntegrationBuilder) which should be passed
    to TimeIntegrationBuildDirector. In addition, there are few functions encapsulating this process. It allows for the
    safe and complete creation of TimeIntegration instance.

    One should also note that TimeIntegration is a base class so one has to use one of its derived classes.
    For example, this package contains class TimeIntegrationInOldChannelFlow adapted for the data produced by old
    (non-MPI) version of channelflow.
    """

    def __init__(self, data_path) -> None:
        self.t = None
        self.other_data_access_strategy = None
        self.solution_access_strategy = None
        self.transform = None
        self._data_path = data_path
        self._solution_time_series = {}
        self._other_data = {}

    def __getattr__(self, data_id):
        data = self.other_data_access_strategy.access_data(self._other_data, data_id,
                                                           partial(upload_and_transform_any,
                                                                   self.upload_data,
                                                                   self.transform),
                                                           data_id)
        return data

    @property
    def data_path(self) -> str:
        """
        Returns the absolute data path where all data files are assumed to be
        :return: data path as a string
        """
        return self._data_path

    @property
    def simulation_configuration(self) -> dict:
        """
        Returns the simulation configuration: numerical resolution, domain size, parameter values, etc.
        :return: dictionary of configuration values
        """
        sim_conf = self.other_data_access_strategy.access_data(self._other_data, 'simulation_configuration',
                                                               self.upload_simulation_configuration)
        return sim_conf

    def solution(self, t) -> Field:
        """
        Returns the solution field at time t
        :param t: time at which solution is accessed
        :return: solution at time t as an instance of Field
        """
        if self.t is None:
            self._load_time_domain()
        t_i = index_for_almost_exact_coincidence(self.t, t)
        sol_field = self.solution_access_strategy.access_data(self._solution_time_series, t_i,
                                                              self.upload_solution, t_i)
        return sol_field

    def initial_condition(self) -> Field:
        """
        Returns
        :return: solution at time t = 0 as an instance of Field
        """
        ic_field = self._solution_time_series[0]
        self.solution_access_strategy.access_data(self._solution_time_series, 0,
                                                  self.upload_solution, 0)
        return ic_field

    def _load_time_domain(self) -> None:
        _, _, filenames = next(os.walk(self._data_path))
        regexp_pattern = self._get_regexp_for_solution_filenames()
        self.t = []
        for filename in filenames:
            params_dict = parse_by_named_regexp(regexp_pattern, filename)
            if params_dict is not None:
                self.t.append(float(params_dict['num']))
        self.t.sort()

    @abstractmethod
    def _get_regexp_for_solution_filenames(self) -> str:
        raise NotImplementedError('Must be implemented. It must return the regular expression with "num" as a '
                                  'parameter')

    @abstractmethod
    def upload_solution(self, t_i: SupportsFloat) -> Mapping[SupportsFloat, Field]:
        raise NotImplementedError('Must be implemented. It must return the solution as Field instance based on t_i'
                                  ' as an index in a time series')

    @abstractmethod
    def upload_data(self, data_id) -> dict:
        raise NotImplementedError('Must be implemented. It must return the dictionary containing all the data '
                                  'physically linked with the data identified by data_id (e.g., it may be all time '
                                  'series from the same file). The dictionary, apparently, must also contain the data '
                                  'identified by data_id ')

    @abstractmethod
    def upload_simulation_configuration(self) -> Mapping[Literal['simulation_configuration'],  dict]:
        raise NotImplementedError('Must be implemented. It must return the dictionary containing simulation '
                                  'configuration')


class TimeIntegrationInOldChannelFlow(TimeIntegration):
    """
    Class TimeIntegrationInOldChannelFlow represent time-integration performed by the old channelflow, non-mpi version.

    Its ``other data`` are
      - T (time)
      - L2U (time-evolution of the L2-norm of the flow field u, i.e., the fluctuation around the laminar solution)
      - L2u, L2v, L2w (time-evolution of the L2-norms of the components of the flow field)
      - D (time-evolution of dissipation)
      - max_KE (time-evolution of maximum pointwise kinetic energy)
      - LF (time-evolution of the location of the left front of the spot)
      - RF (time-evolution of the location of the right front of the spot)
      - UlamDotU (time-evolution of the scalar product <U_lam, u>, where U_lam is the laminar solution and u is the
      fluctuation around it)
      - L2Ulam (time-evolution of the L2-norm of the laminar solution)
      - L2UlamPlusU (time-evolution of the L2-norm of the full flow field (i.e., of U_lam + u))
      - DUlamPlusU (time-evolution of dissipation of the full flow field (i.e., of U_lam + u))
      - ke_z (time-evolution of the xy-averaged kinetic energy)
      - u_z (time-evolution of the xy-averaged L2-norm of the streamwise velocity)
      - v_z (time-evolution of the xy-averaged L2-norm of the wall-normal velocity)
    """
    def __init__(self, data_path) -> None:
        self._scalar_time_series_ids = ('T', 'L2U', 'L2u', 'L2v', 'L2w', 'D', 'max_KE', 'LF', 'RF', 'UlamDotU',
                                        'L2Ulam', 'L2UlamPlusU', 'DUlamPlusU')
        self._xy_aver_time_series_ids = ('ke_z', 'u_z', 'v_z')
        TimeIntegration.__init__(self, data_path)
        pass

    def _get_regexp_for_solution_filenames(self) -> str:
        return '^u(?P<num>\d+[.]?\d*).h5$'

    def upload_simulation_configuration(self) -> dict:
        _, attrs = self._read_field_and_attrs(0)
        return {'simulation_configuration': attrs}

    def upload_solution(self, t_i: SupportsFloat) -> Mapping[SupportsFloat, Field]:
        field, _ = self._read_field_and_attrs(t_i)
        return {t_i: field}

    def upload_data(self, data_id) -> dict:
        """
        Old version of channel flow has several version of summary data files. We check them all in row trying to find
        a fitting option.
        :return: dict
        """

        if data_id in self._scalar_time_series_ids:
            data = self._upload_scalar_time_series()
        elif data_id in self._xy_aver_time_series_ids:
            data_obj = self._upload_xy_aver_series(data_id)
            data = {data_id: data_obj}
        else:
            raise KeyError('Unsupported data ID passed: {}'.format(data_id))
        return data

    def _upload_scalar_time_series(self):
        all_cases_of_param_names = [
            ['T', 'L2U', 'L2u', 'L2v', 'L2w', 'D', 'max_KE', 'LF', 'RF', 'UlamDotU', 'L2Ulam', 'L2UlamPlusU',
             'DUlamPlusU'],
            ['T', 'L2U', 'L2u', 'L2v', 'L2w', 'D', 'max_KE', 'LF', 'RF', 'UlamDotU', 'L2Ulam', 'L2UlamPlusU'],
            ['T', 'L2U', 'L2u', 'L2v', 'L2w', 'D', 'max_KE', 'LF', 'RF'],
            ['T', 'L2U', 'D', 'max_KE', 'LF', 'RF'],
            ['T', 'L2U', 'D', 'max_KE'],
        ]
        scalar_time_series = None
        for param_names in all_cases_of_param_names:
            try:
                scalar_time_series = parse_datafile(os.path.join(self._data_path, 'summary.txt'), param_names,
                                                    [float for _ in range(len(param_names))])
                break
            except Exception as e_:
                pass
        if scalar_time_series is None:
            print('No supported version of summary file is found. Last exception thrown while parsing summary file: '
                  '{}'.format(e_))
        return scalar_time_series

    def _upload_xy_aver_series(self, data_id) -> Field:
        """
        Old version of channel flow permits several combinations of vector series files, namely, xy-averaged ||u||_2,
        xy-averaged ||v||_2, xy-averaged kinetic energy and spanwise component of the laminar flow.
        :return: Field
        """

        if data_id == 'ke_z':
            filename = 'avenergy.txt'
        elif data_id == 'u_z':
            filename = 'av_u.txt'
        elif data_id == 'v_z':
            filename = 'av_v.txt'
        else:
            raise KeyError('Unsupported data ID passed: {}'.format(data_id))
        t, q = parse_timed_numdatafile(os.path.join(self._data_path, filename))
        t, z = parse_timed_numdatafile(os.path.join(self._data_path, 'z.txt'))
        space = Space([np.array(t), np.array(z[0])])
        space.set_elements_names(['t', 'z'])
        field_ = Field([np.array(q)], space)
        field_.set_elements_names([data_id])
        return field_

    def _read_field_and_attrs(self, t_i: SupportsFloat):
        field_name = 'u{:.3f}.h5'.format(self.t[t_i])
        field, attrs = read_field(os.path.join(self._data_path, field_name))
        return field, attrs


class TimeIntegrationBuilder:
    """
    Class TimeIntegrationBuilder is a base builder class for TimeIntegration objects (see Builder pattern for details).
    Since TimeIntegration is a base class itself and cannot thus be created, one should pass a concrete TimeIntegration
    class to the constructor.

    One should derive its own class from this base one which will specify transform and DataAccessStrategy for
    real-valued series and DataAccessStrategy for solution fields.
    """
    def __init__(self, ti_class):
        self._ti_class = ti_class
        self._transform = None
        self._other_data_access_strategy = None
        self._solution_access_strategy = None

    def get_timeintegration(self, ti_path):
        ti_obj = self._ti_class(ti_path)
        ti_obj.other_data_access_strategy = self._other_data_access_strategy
        ti_obj.solution_access_strategy = self._solution_access_strategy
        ti_obj.transform = self._transform
        return ti_obj

    @abstractmethod
    def create_transform(self) -> None:
        raise NotImplementedError('Must be implemented')

    @abstractmethod
    def create_other_data_access_strategy(self) -> None:
        raise NotImplementedError('Must be implemented')

    @abstractmethod
    def create_solution_access_strategy(self) -> None:
        raise NotImplementedError('Must be implemented')


class NoBackupAccessBuilder(TimeIntegrationBuilder):
    """
    Class NoBackupAccessBuilder implements TimeIntegrationBuilder with such DataAccessStrategy for solution fields and
    other data that they never stored in TimeIntegration.
    """
    def __init__(self, ti_class):
        TimeIntegrationBuilder.__init__(self, ti_class)

    def create_transform(self) -> None:
        pass

    def create_other_data_access_strategy(self) -> None:
        self._other_data_access_strategy = free_data_after_access_strategy

    def create_solution_access_strategy(self) -> None:
        self._solution_access_strategy = free_data_after_access_strategy


class TimeIntegrationBuildDirector:
    """
    Class TimeIntegrationBuildDirector is a director in Builder pattern and is used for builder construction.
    A common use is that one create a builder, then create a director passing the builder to the constructor of the
    director and then call director's method construct(). After that, the builder can be used to produce TimeIntegration
    instances -- as many as one wants.
    """
    def __init__(self, builder):
        self.__builder = builder

    def construct(self):
        self.__builder.create_transform()
        self.__builder.create_other_data_access_strategy()
        self.__builder.create_solution_access_strategy()


def upload_and_transform_any(uploader: Callable[..., dict],
                             transformer: Optional[Callable[[dict], dict]],
                             *upload_func_args) -> dict:
    data = uploader(*upload_func_args)
    if transformer is not None:
        data = transformer(data)
    return data


def build_timeintegration_sequence(res, tasks, timeintegration_builder,
                                   regexp_pattern='^data-(?P<num>\d+[.]?\d*)$') -> Function:
    """
    Builds and returns a sequence of TimeIntegration instances each of which corresponds to a particular diretory in
    one of the given tasks

    :param res: Research object; it will be used to look for tasks
    :param tasks: a sequence of tasks (integer values); they will be treated as directories with many other
    subdirectories each of which is treated as time-integration data directory
    :param timeintegration_builder: TimeIntegrationBuilder object; it will be used to create TimeIntegration instances
    :param regexp_pattern: regular expression with group parameter num; it will be matched against dir names
    :return: a sequence of time-integrations as an instance of Function
    """
    data = []
    domain = []
    for task in tasks:
        root_path = res.get_task_path(task)
        print('\tReading ' + root_path)
        _, dirnames, _ = next(os.walk(root_path))
        for dirname in dirnames:
            params_dict = parse_by_named_regexp(regexp_pattern, dirname)
            if params_dict is None:
                print('\t Data directory "{}" in "{}" is invalid'.format(dirname, root_path))
                print('\t\t Skip it and go next')
            else:
                domain.append(float(params_dict['num']))
                data.append(timeintegration_builder.get_timeintegration(os.path.join(root_path, dirname)))
    return Function(data, domain)


'''
def get_baseflow_field(data_path, T, include_ubase=False):
    t, wbaset_list = parse_timed_numdatafile(os.path.join(data_path, 'wbase_t.txt'))
    wbase_list = np.array(wbaset_list)[t.index(T), :]
    ubase_list = np.array(parse_datafile(os.path.join(data_path, 'Ubase.asc'), ['U'], [float])['U'])
    initial_field, _ = read_field(os.path.join(data_path, 'u0.000.h5'))
    space = initial_field.space
    U = np.zeros_like(initial_field.u)
    V = np.zeros_like(initial_field.v)
    W = np.zeros_like(initial_field.w)
    for i in range(initial_field.u.shape[0]):
        for k in range(initial_field.u.shape[2]):
            if include_ubase:
                U[i, :, k] = ubase_list
            W[i, :, k] = wbase_list
    ubase_field = Field([U, V, W], space)
    ubase_field.set_uvw_naming()
    return ubase_field

def get_wbase(data_path):
    t, wbase = parse_timed_numdatafile(os.path.join(self._data_path, 'wbase_t.txt'))
    wbase_list = np.array(wbaset_list)
#    os.path.join(data_path, 'u0.000.h5')
    initial_field, _ = read_field(os.path.join(data_path, get_file_by_prefix(data_path, 'u')))
    space = Space([np.array(t), np.array(initial_field.space.y[::-1])]) # reverse order to coincide with a backward order from channelflow
    space.set_elements_names(['t', 'y'])
    wbase_field = Field([wbase_list], space)
    wbase_field.set_elements_names(['w'])
    return wbase_field

def get_xy_averaged_ke(data_path):
    xy_averaged_ke_file_cases = [os.path.join(data_path, 'avenergy.txt'), os.path.join(data_path, 'xyavg_energy', 'out.nc')]
    ke_field = None
    if os.path.exists(xy_averaged_ke_file_cases[0]):
        xy_averaged_ke_file = xy_averaged_ke_file_cases[0]
        t, ke_timed_list = parse_timed_numdatafile(xy_averaged_ke_file)
        _, z = parse_timed_numdatafile(data_path + '/z.txt')
        t = np.array(t)
        z = np.array(z[0])
        ke = np.array(ke_timed_list)
    elif os.path.exists(xy_averaged_ke_file_cases[1]):
        xy_averaged_ke_file = xy_averaged_ke_file_cases[1]
        f = netCDF4.Dataset(xy_averaged_ke_file, 'r', format='NETCDF4')
        t = np.array([float(os.path.splitext(filename)[0]) for filename in f.nco_input_file_list.split()])
        ke = np.array(f['Component_0'][:, :, 0, 0])
        ind = np.argsort(t) # find such indices that t is sorted
        t = np.take_along_axis(t, ind, axis=0) # sort t
        #ke = np.take_along_axis(ke, ind, axis=0) # sort ke along t-axis
        ke = np.take(ke, ind, axis=0) # sort ke along t-axis
        z = np.array(f['Z'])
    space = Space([t, z])
    space.set_elements_names(['t', 'z'])
    ke_field = Field([ke], space)
    ke_field.set_elements_names(['ke'])
    return ke_field

def get_xy_averaged_ke_from_fields(data_path):
    times = []
    for file_or_dir in os.listdir(data_path):
        basename, ext = os.path.splitext(file_or_dir)
        if ext in ['.nc', '.h5']:
            times.append(int(basename[1:]))
    times.sort()
    init_f, _ = read_field(os.path.join(data_path, 'u{}.nc'.format(times[0])))
    aver_ke_raw_field = np.zeros((len(times), len(init_f.space.z)))
    ke_raw_field = np.zeros_like(init_f.u)
    for t_i in range(len(times)):
        print('Reading t = {}'.format(times[t_i]))
        f, _ = read_field(os.path.join(data_path, 'u{}.nc'.format(times[t_i])))
        ke_raw_field = 0.5 * (np.power(f.u, 2) + np.power(f.v, 2) + np.power(f.w, 2))
        aver_ke_raw_field[t_i, :] = np.trapz(np.mean(ke_raw_field, axis=0), x=f.space.y, axis=0)

    space = Space([np.array(times, dtype=np.float64), np.array(init_f.space.z)])
    space.set_elements_names(['t', 'z'])
    ke_field = Field([aver_ke_raw_field], space)
    ke_field.set_elements_names(['ke'])
    return ke_field

def get_theory_wbase(A, omega, Re, ts, ys):
    theta = np.sqrt(omega * Re / 2.)
    Lambda = np.cos(2.*theta) + np.cosh(2.*theta)
    y_plus = theta*(1 + ys)
    y_minus = theta*(1 - ys)
    f = (np.cosh(y_plus)*np.cos(y_minus) + np.cosh(y_minus)*np.cos(y_plus)) / Lambda
    g = -(np.sinh(y_plus)*np.sin(y_minus) + np.sinh(y_minus)*np.sin(y_plus)) / Lambda
    W = np.zeros((len(ts), len(ys)))
    for t_i in range(len(ts)):
        W[t_i, :] = A * (f*np.sin(omega*ts[t_i]) + g*np.cos(omega*ts[t_i]))
    space = Space([np.array(ts), np.array(ys)])
    space.set_elements_names(['t', 'y'])
    wbase_field = Field([W], space)
    wbase_field.set_elements_names(['w'])
    return wbase_field

def get_theory_general_wbase(A_plus, A_minus, omega_plus_, omega_minus_, phi_plus_, phi_minus_, Re, ts, ys):
    theta_plus_ = np.sqrt(omega_plus_ * Re / 2.)
    A_over_lambda_plus_ = A_plus / (np.cosh(2*theta_plus_)**2 - np.cos(2*theta_plus_)**2)
    theta_minus_ = np.sqrt(omega_minus_ * Re / 2.)
    A_over_lambda_minus_ = A_minus / (np.cosh(2*theta_minus_)**2 - np.cos(2*theta_minus_)**2)

    def a(y):
        y_plus = theta_plus_*(y+1)
        y_minus = theta_plus_*(y-1)
        return np.sin(2*theta_plus_)*np.cosh(y_minus)*np.sin(y_plus) + np.sinh(2*theta_plus_)*np.sinh(y_plus)*np.cos(y_minus)

    def b(y):
        y_plus = theta_plus_*(y+1)
        y_minus = theta_plus_*(y-1)
        return np.sinh(2*theta_plus_)*np.cosh(y_plus)*np.sin(y_minus) - np.sin(2*theta_plus_)*np.sinh(y_minus)*np.cos(y_plus)

    def c(y):
        y_minus = theta_minus_*(y-1)
        return -np.sinh(2*theta_minus_)*np.cos(2*theta_minus_)*np.sinh(y_minus)*np.cos(y_minus) \
               -np.cosh(2*theta_minus_)*np.sin(2*theta_minus_)*np.cosh(y_minus)*np.sin(y_minus)

    def d(y):
        y_minus = theta_minus_*(y-1)
        return -np.sinh(2*theta_minus_)*np.cos(2*theta_minus_)*np.cosh(y_minus)*np.sin(y_minus) \
               +np.cosh(2*theta_minus_)*np.sin(2*theta_minus_)*np.sinh(y_minus)*np.cos(y_minus)

    W = np.zeros((len(ts), len(ys)))
    for t_i in range(len(ts)):
        W[t_i, :] = A_over_lambda_plus_*(a(ys)*np.sin(omega_plus_*ts[t_i] + 2*np.pi*phi_plus_) + b(ys)*np.cos(omega_plus_*ts[t_i] + 2*np.pi*phi_plus_)) \
                  + A_over_lambda_minus_*(c(ys)*np.sin(omega_minus_*ts[t_i] + 2*np.pi*phi_minus_) + d(ys)*np.cos(omega_minus_*ts[t_i] + 2*np.pi*phi_minus_))

#    def a(y, A, Omega):
#        return A / (np.cosh(2*Omega)**2 - np.cos(2*Omega)**2) * (np.sin(2*Omega)*np.cosh(Omega*(y-1))*np.sin(Omega*(y+1)) + np.sinh(2*Omega)*np.sinh(Omega*(y+1))*np.cos(Omega*(y-1)))
#    def b(y, A, Omega):
#        return A / (np.cosh(2*Omega)**2 - np.cos(2*Omega)**2) * (np.sinh(2*Omega)*np.cosh(Omega*(y+1))*np.sin(Omega*(y-1)) - np.sin(2*Omega)*np.sinh(Omega*(y-1))*np.cos(Omega*(y+1)))
#    def c(y, A, Omega):
#        return A / (np.cosh(2*Omega)**2 - np.cos(2*Omega)**2) * (-np.sinh(2*Omega)*np.cos(2*Omega)*np.sinh(Omega*(y-1))*np.cos(Omega*(y-1)) - np.cosh(2*Omega)*np.sin(2*Omega)*np.cosh(Omega*(y-1))*np.sin(Omega*(y-1)))
#    def d(y, A, Omega):
#        return A / (np.cosh(2*Omega)**2 - np.cos(2*Omega)**2) * (-np.sinh(2*Omega)*np.cos(2*Omega)*np.cosh(Omega*(y-1))*np.sin(Omega*(y-1)) + np.cosh(2*Omega)*np.sin(2*Omega)*np.sinh(Omega*(y-1))*np.cos(Omega*(y-1)))
#    def W_1(t, y, A, Omega, omega_, phi):
#        return A / (np.cosh(2*Omega)**2 - np.cos(2*Omega)**2) * (np.sin(2*Omega)*np.cosh(Omega*(y-1))*np.sin(Omega*(y+1)) + np.sinh(2*Omega)*np.sinh(Omega*(y+1))*np.cos(Omega*(y-1))) * np.sin(omega_*t + phi) \
#             + A / (np.cosh(2*Omega)**2 - np.cos(2*Omega)**2) * (np.sinh(2*Omega)*np.cosh(Omega*(y+1))*np.sin(Omega*(y-1)) - np.sin(2*Omega)*np.sinh(Omega*(y-1))*np.cos(Omega*(y+1))) * np.cos(omega_*t + phi)
#    def W_2(t, y, A, Omega, omega_, phi):
#        return A / (np.cosh(2*Omega)**2 - np.cos(2*Omega)**2) * (-np.sinh(2*Omega)*np.cos(2*Omega)*np.sinh(Omega*(y-1))*np.cos(Omega*(y-1)) - np.cosh(2*Omega)*np.sin(2*Omega)*np.cosh(Omega*(y-1))*np.sin(Omega*(y-1))) * np.sin(omega_*t + phi) \
#             + A / (np.cosh(2*Omega)**2 - np.cos(2*Omega)**2) * (-np.sinh(2*Omega)*np.cos(2*Omega)*np.cosh(Omega*(y-1))*np.sin(Omega*(y-1)) + np.cosh(2*Omega)*np.sin(2*Omega)*np.sinh(Omega*(y-1))*np.cos(Omega*(y-1))) * np.cos(omega_*t + phi)
#
#    W = np.zeros((len(ts), len(ys)))
#    for t_i in range(len(ts)):
#        W[t_i, :] = W_1(ts[t_i], ys, A_plus, theta_plus_, omega_plus_, phi_plus_) + W_2(ts[t_i], ys, A_minus, theta_minus_, omega_minus_, phi_minus_)

    space = Space([np.array(ts), np.array(ys)])
    space.set_elements_names(['t', 'y'])
    wbase_field = Field([W], space)
    wbase_field.set_elements_names(['w'])
    return wbase_field

def get_generated_dissipation(d_path):
    Ts = []
    Ds = []
    files = os.listdir(d_path)
    for file_ in files:
        Ts.append(float(file_[1:-4]))
        D_file = open(os.path.join(d_path, file_))
        Ds.append(float(D_file.read()))
    coupled_list = sorted(zip(Ts, Ds), key=itemgetter(0))
    Ts = [pair[0] for pair in coupled_list]
    Ds = [pair[1] for pair in coupled_list]
    return Ts, Ds
'''
