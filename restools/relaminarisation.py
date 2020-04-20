from typing import Optional
from typing_extensions import TypedDict

import restools.timeintegration as ti
from restools.timeintegration_builders import ensure_data_id_supported


RelamTimeData = TypedDict('RelamTimeData', {'t_relam': Optional[float]})


@ensure_data_id_supported(ids=['t_relam'])
def upload_relaminarisation_time(ti_obj: ti.TimeIntegration, data_id) -> RelamTimeData:
    """
    Uploads relaminarisation time based on TimeIntegration object. Used to extend method upload_data implemented in
    the class derived from TimeIntegration.

    :param ti_obj: TimeIntegration instance
    :param data_id: Data ID associated with relaminarisation time (t_relam)
    :return: dict {'t_relam': float}
    """
    max_ke_eps = 0.2
    out = {'t_relam': None}
    for i in range(len(ti_obj.T)):
        if ti_obj.max_KE[i] < max_ke_eps:
            out['t_relam'] = ti_obj.T[i]
            break
    return out
