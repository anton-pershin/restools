from typing import Tuple, List, Callable

import numpy as np

from restools.timeintegration import TimeIntegration
from restools.relaminarisation import is_relaminarised
from thequickmath.aux import index_for_almost_exact_coincidence, index_for_closest_element
from thequickmath.stats import EmpiricalDistribution


class Ensemble:
    """
    Class Ensemble represents a collection of simulations (as TimeIntegration objects) and allows to build empirical
    distributions with respect to any of time series provided by TimeIntegration.
    """
    def __init__(self, tis: List[TimeIntegration], initial_cutoff_time=100.,
                 turb_to_lam_transitional_time=400., minimal_simulation_duration=500., max_ke_eps=0.2):
        self.tis = tis
        self.initial_cutoff_time = initial_cutoff_time
        self.turb_to_lam_transitional_time = turb_to_lam_transitional_time
        self.minimal_simulation_duration = minimal_simulation_duration
        self.max_ke_eps = max_ke_eps

    def empirical_distribution(self, scalar_time_series_attr: str,
                               transform: Callable[[np.ndarray], np.ndarray] = lambda d: d) -> EmpiricalDistribution:
        """
        Returns empirical distribution built from the time series associated with attribute scalar_time_series_attr of
        TimeIntegration and then transformed via transform

        :exception: BadEnsemble if after filtering time series no data samples left
        """
        _, data_samples = self._data_samples_in_row(scalar_time_series_attr, transform)
        if len(data_samples) == 0:
            raise BadEnsemble('Given ensemble yields no data samples after appropriate selection. Selection settings: '
                              'initial cut-off time = {}, turbulent-to-laminar transition time = {}, '
                              'minimal simulation duration after '
                              'selection = {}'.format(self.initial_cutoff_time,
                                                      self.turb_to_lam_transitional_time,
                                                      self.minimal_simulation_duration))
        return EmpiricalDistribution(data_samples)

    def ke_distribution(self) -> EmpiricalDistribution:
        """
        Returns empirical distribution of kinetic energy

        :exception: BadEnsemble if after filtering time series no data samples left
        """
        return self.empirical_distribution('L2U', lambda d: 0.5*d**2)

    def dissipation_distribution(self) -> EmpiricalDistribution:
        """
        Returns empirical distribution of dissipation rate

        :exception: BadEnsemble if after filtering time series no data samples left
        """
        return self.empirical_distribution('DUlamPlusU')

    def block_extrema(self, scalar_time_series_attr: str, transform: Callable[[np.ndarray], np.ndarray] = lambda d: d,
                      block_size=500.) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns block minima and maxima found based on the time series associated with attribute scalar_time_series_attr
        of TimeIntegration and then transformed via transform. The whole time series is divided into block of size
        block_size and minima/maxima are found within each block.

        :return: a tuple of arrays of block minima and maxima
        :exception: BadEnsemble if after filtering time series no data samples left
        """
        times, data_samples = self._data_samples_in_row(scalar_time_series_attr, transform)
        blocks_num = int(times[-1] // block_size)
        block_minima = np.zeros((blocks_num,))
        block_maxima = np.zeros_like(block_minima)
        last_i = 0
        for block_i in range(blocks_num):
            next_i = index_for_almost_exact_coincidence(times, (block_i + 1) * block_size)
            block_minima[block_i] = np.min(data_samples[last_i: next_i])
            block_maxima[block_i] = np.max(data_samples[last_i: next_i])
            last_i = next_i
        return block_minima, block_maxima

    def _data_samples_in_row(self, scalar_time_series_attr: str,
                             transform: Callable[[np.ndarray], np.ndarray] = lambda d: d):
        data_samples = np.array([], dtype=float)
        times = np.array([], dtype=float)
        for ti in self.tis:
            t = ti.T
            max_ke = ti.max_ke
            t -= t[0]  # let's ensure that the time series starts from t = 0
            if t[-1] <= self.initial_cutoff_time + self.minimal_simulation_duration:
                continue
            t_i_begin = index_for_almost_exact_coincidence(t, self.initial_cutoff_time)
            t_i_relam = -1
            if is_relaminarised(max_ke, self.max_ke_eps):
                t_i_relam = index_for_closest_element(max_ke, self.max_ke_eps)

            if t[t_i_relam] - t[t_i_begin] < self.turb_to_lam_transitional_time + self.minimal_simulation_duration:
                continue
            else:
                t_i_end = index_for_almost_exact_coincidence(t, t[t_i_relam] - self.turb_to_lam_transitional_time)
            time_series = transform(getattr(ti, scalar_time_series_attr))
            data_samples = np.r_[data_samples, time_series[t_i_begin:t_i_end]]
            delta_t = t[t_i_begin + 1] - t[t_i_begin]
            time_shift = -t[t_i_begin] if len(times) == 0 else times[-1] + delta_t - t[t_i_begin]
            times = np.r_[times, t[t_i_begin:t_i_end] + time_shift]
        return times, data_samples


class BadEnsemble(Exception):
    pass
