from typing_extensions import Literal
import functools
from abc import abstractmethod

from restools.timeintegration import TimeIntegrationChannelFlowV1, TimeIntegrationChannelFlowV2
from restools.data_access_strategies import free_data_after_access_strategy, hold_data_in_memory_after_access_strategy


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
        self._other_data_access_strategy = None
        self._solution_access_strategy = None

    def get_timeintegration(self, ti_path):
        ti_obj = self._ti_class(ti_path)
        ti_obj.other_data_access_strategy = self._other_data_access_strategy
        ti_obj.solution_access_strategy = self._solution_access_strategy
        return ti_obj

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

    def create_other_data_access_strategy(self) -> None:
        self._other_data_access_strategy = free_data_after_access_strategy

    def create_solution_access_strategy(self) -> None:
        self._solution_access_strategy = free_data_after_access_strategy


class CacheAllAccessBuilder(TimeIntegrationBuilder):
    """
    Class CacheAllAccessBuilder implements TimeIntegrationBuilder with such DataAccessStrategy for solution fields and
    other data that they immediately cached in TimeIntegration once they are accessed.
    """
    def __init__(self, ti_class):
        TimeIntegrationBuilder.__init__(self, ti_class)

    def create_other_data_access_strategy(self) -> None:
        self._other_data_access_strategy = hold_data_in_memory_after_access_strategy

    def create_solution_access_strategy(self) -> None:
        self._solution_access_strategy = hold_data_in_memory_after_access_strategy


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
        self.__builder.create_other_data_access_strategy()
        self.__builder.create_solution_access_strategy()


def get_ti_builder(cf_version: Literal['cfv1', 'cfv2'] = 'cfv1',
                   cache=False, upload_data_extension=None) -> TimeIntegrationBuilder:
    """
    Returns TimeIntegrationBuilder associated with a particular version of channelflow (cf_version), selected
    xy-averaged quantities, uploaded to vector_series, and able to either store or immediately free all the uploaded
    data (nobackup)

    :param cf_version: version of channelflow (can be either 'cfv1' or 'cfv2')
    :param cache: whether uploaded data should be cached after the use (nobackup=False) or not (nobackup=True)
    :param upload_data_extension: function with decorator ensure_data_id_supported loading additional data by Data ID
    :return: TimeIntegrationBuilder constructed by TimeIntegrationBuildDirector
    """
    ti_base_class = None
    if cf_version == 'cfv1':
        ti_base_class = TimeIntegrationChannelFlowV1
    elif cf_version == 'cfv2':
        ti_base_class = TimeIntegrationChannelFlowV2
    else:
        raise NotImplemented('The case cf_version={} must be implemented!'.format(cf_version))

    ti_class = ti_base_class
    if upload_data_extension is not None:
        ti_class = type('{}_{}'.format(ti_base_class.__name__, id(upload_data_extension)), (ti_base_class,), {})

        def _overridden_upload_data(obj, data_id):
            extra_data = upload_data_extension(obj, data_id)
            if extra_data is None:
                return super(type(obj), obj).upload_data(data_id)
            else:
                return extra_data
        ti_class.upload_data = _overridden_upload_data
    builder = None
    if cache:
        builder = CacheAllAccessBuilder(ti_class)
    else:
        builder = NoBackupAccessBuilder(ti_class)
    director = TimeIntegrationBuildDirector(builder)
    director.construct()
    return builder


def ensure_data_id_supported(func_=None, *, ids=()):
    """
    Decorator ensure_data_id_supported must be used when one wants to create a function ``upload_data_extension`` to
    pass it to function get_ti_builder. Argument ids specifies a list of Data IDs introduced by function
    ``upload_data_extension`` on top of what ``upload_data`` of the corresponding class, derived from TimeIntegration,
    provides.
    """
    def decorator_(func_):
        @functools.wraps(func_)
        def wrapper_(ti_obj, data_id, *args, **kwargs):
            if data_id in ids:
                return func_(ti_obj, data_id, *args, **kwargs)
            else:
                return None
        return wrapper_

    if func_ is None:
        return decorator_
    else:
        return decorator_(func_)