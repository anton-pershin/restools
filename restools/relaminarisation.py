from typing import Optional, Sequence
from typing_extensions import TypedDict

import numpy as np

import restools.timeintegration as ti
from restools.timeintegration_builders import ensure_data_id_supported


RelamTimeData = TypedDict('RelamTimeData', {'t_relam': Optional[float]})


@ensure_data_id_supported(ids=['t_relam'])
def upload_relaminarisation_time(ti_obj: ti.TimeIntegration, data_id, max_ke_eps=0.2) -> RelamTimeData:
    """
    Uploads relaminarisation time based on TimeIntegration object. Used to extend method upload_data implemented in
    the class derived from TimeIntegration.

    :param ti_obj: TimeIntegration instance
    :param data_id: Data ID associated with relaminarisation time (t_relam)
    :param max_ke_eps: threshold for the maximum pointwise kinetic energy (if the flow goes below this value, we count
                       this as a relaminarisation event
    :return: dict {'t_relam': float}
    """
    return {'t_relam': get_relaminarisation_time(ti_obj.T, ti_obj.max_ke, max_ke_eps)}


def expected_relam_time(ts: Sequence[float], Res: Sequence[float], delta_Re: float) -> np.ndarray:
    """
    Returns the expected relaminarisation time assuming that the real, observed Reynolds number is uniformly
    distributed in the interval [Re - delta_Re; Re + delta_Re]. That is, it is calculated as
    <t> = 1 / \delta Re \int_{Re - \delta Re}^{Re + \delta Re} t dRe

    :param ts: an array of relaminarisation times
    :param Res: an array of Reynolds numbers associated with ts
    :param delta_Re: half of the width of the interval of the uniform distribution used to calculate the expectation
    :return: the expected relaminarisation time as a function of Re
    """
    left_i = 0
    right_i = 0
    t_mean = np.zeros_like(ts)
    def _find_boundary(boundary_i, Res, wanted_Re):
        while boundary_i != len(Res) - 1:
            cur_boundary_Re = Res[boundary_i]
            next_boundary_Re = Res[boundary_i + 1]
            if next_boundary_Re < wanted_Re:
                boundary_i += 1
            else:
                if np.abs(next_boundary_Re - wanted_Re) < np.abs(cur_boundary_Re - wanted_Re):
                    boundary_i += 1
                break
        return boundary_i

    for i in range(len(ts)):
        Re = Res[i]
        left_i = _find_boundary(left_i, Res, Re - delta_Re)
        right_i = _find_boundary(right_i, Res, Re + delta_Re)
        t_mean[i] = 1./(Res[right_i] - Res[left_i]) * np.trapz(ts[left_i:right_i + 1], x=Res[left_i:right_i + 1])
    return t_mean


def is_relaminarised(max_ke: np.ndarray, max_ke_eps=0.2):
    return max_ke[-1] < max_ke_eps


def get_relaminarisation_time(t: np.ndarray, max_ke: np.ndarray, max_ke_eps=0.2) -> Optional[float]:
    for j in range(len(t)):
        if max_ke[j] < max_ke_eps:
            return t[j]
    return None
