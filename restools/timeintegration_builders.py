from typing import Sequence
from typing_extensions import Literal

from restools.timeintegration import TimeIntegrationInOldChannelFlow, NoBackupAccessBuilder, \
    TimeIntegrationBuildDirector, TimeIntegrationBuilder


class RelaminarisationTimeBuilder(NoBackupAccessBuilder):
    """
    Class RelaminarisationTimeBuilder builds light-weight TimeIntegration instances which only contains the
    relaminarisation time associated with the simulation (it can be accessed via .scalar_series['t_relam']).
    """
    def __init__(self):
        NoBackupAccessBuilder.__init__(self, TimeIntegrationInOldChannelFlow)

    def create_transform_scalar_series(self) -> None:
        def transform_(d) -> dict:
            max_ke_eps = 0.2
            out = {'t_relam': None}
            for i in range(len(d['T'])):
                if d['max_KE'][i] < max_ke_eps:
                    out['t_relam'] = d['T'][i]
                    break
            return out
        self._transform_scalar_series = transform_


def get_ti_builder(cf_version: Literal['old', 'epfl'] = 'old',
                   xy_averaged_quantities: Sequence[str] = ('ke', 'u', 'v'),
                   nobackup=True) -> TimeIntegrationBuilder:
    """
    Returns TimeIntegrationBuilder associated with a particular version of channelflow (cf_version), selected
    xy-averaged quantities, uploaded to vector_series, and able to either store or immediately free all the uploaded
    data (nobackup)

    :param cf_version: version of channelflow (can be either 'old' or 'epfl')
    :param xy_averaged_quantities: list of xy-averaged quantities which should uploaded into vector_series. Can be any
    subset of ('ke', 'u', 'v')
    :param nobackup: whether uploaded data should be cached after the use (nobackup=False) or not (nobackup=True)
    :return:
    """
    if cf_version == 'old':
        ti_class = TimeIntegrationInOldChannelFlow
    else:
        raise NotImplemented('The case cf_version={} must be implemented!'.format(cf_version))

    if nobackup:
        builder = NoBackupAccessBuilder(ti_class)
    else:
        raise NotImplemented('The case nobackup={} must be implemented!'.format(nobackup))

    director = TimeIntegrationBuildDirector(builder)
    director.construct(xy_averaged_quantities)
    return builder
