from abc import ABC, abstractmethod
import os
import functools
from typing import Optional, Type, Sequence, Union, Tuple, Callable
from typing_extensions import Literal
from operator import itemgetter, add

import numpy as np
import scipy.stats
from scipy.optimize import root_scalar
from scipy.integrate import simps

from restools.relaminarisation import is_relaminarised
from restools.timeintegration import TimeIntegration, Perturbation
from restools.timeintegration_builders import TimeIntegration3DBuilder
import comsdk.comaux as comaux
from comsdk.research import Research
from thequickmath.aux import index_for_almost_exact_coincidence_unsorted
from thequickmath.stats import ScipyDistribution


Prior = Union[Callable[[Sequence[float], Sequence[float], int], Tuple[float, float]],
              Literal['uniform', 'jeffreys', 'haldane']]


class LaminarisationStudy:
    """
    Class LaminarisationStudy wraps up the data associated with a study of laminarisation probability. Such a study
    involves generating a large number of random perturbations (can be accessed via LaminarisationStudy.perturbations)
    having specific structure and kinetic energy (the values of the later can be accessed via
    LaminarisationStudy.energy_levels) and then time-integrating them (TimeIntegration instances can be accessed via
    LaminarisationStudy.timeintegrations). Further calculation of the laminarisation probability should be done using
    class LaminarisationProbabilityEstimation.

    Note that even though most of the data is loaded and created at the moment of object construction, we use lazy
    initialisation for TimeIntegration instances. Thus, if one has several thousands of simulations to analyse, calling
    LaminarisationStudy.timeintegrations() may be time-consuming.
    """
    def __init__(self, paths,
                 data_dir_naming: Type[comaux.StandardisedNaming],
                 ti_builder: TimeIntegration3DBuilder,
                 debug=False):
        """

        :param paths: task paths in which random perturbations and time-integration directories are located
        :param data_dir_naming: standardised naming of time-integration directories (must have ``energy_level``
                                and ``i`` as attributes)
        :param ti_builder: TimeIntegrationBuilder used to create instances of TimeIntegration
        """
        self._energy_levels = []
        self._paths = paths
        self._random_perturbations = []  # first index == energy_level_id, second index == path id,
        # self._random_perturbations[i][j] = Perturbation instance
        self._timeintegrations = []  # first index == energy_level_id,
        # second index == path id, third index coincides with the third index of self._random_perturbations
        # lazy initialisation for self._timeintegrations is implemented
        self._ti_builder = ti_builder
        self._data_dir_naming = data_dir_naming
        self._debug = debug

    @classmethod
    def from_paths(cls, paths, rp_naming: Type[comaux.StandardisedNaming],
                   data_dir_naming: Type[comaux.StandardisedNaming],
                   ti_builder: TimeIntegration3DBuilder,
                   energy_rtol=1e-05,
                   energy_atol=1e-08,
                   debug=False):
        study = LaminarisationStudy(paths, data_dir_naming, ti_builder, debug)
        study._upload(rp_naming, energy_rtol, energy_atol)
        return study

    @classmethod
    def from_tasks(cls, res: Research, tasks: Sequence[int], rp_naming: Type[comaux.StandardisedNaming],
                   data_dir_naming: Type[comaux.StandardisedNaming],
                   ti_builder: TimeIntegration3DBuilder,
                   energy_rtol=1e-05,
                   energy_atol=1e-08,
                   debug=False):
        return LaminarisationStudy.from_paths([res.get_task_path(t) for t in tasks], rp_naming, data_dir_naming,
                                              ti_builder, energy_rtol, energy_atol, debug)

    @property
    def energy_levels(self) -> Sequence[float]:
        return self._energy_levels

    def _upload(self, rp_naming, energy_rtol, energy_atol) -> None:
        """
        Walks through the task paths, in each of which random perturbations with standardised naming rp_naming and
        associated data directories of time-integrations with standardised naming data_dir_naming can be found and
        builds necessary inner objects.

        :param rp_naming: standardised naming of random perturbations (must have ``energy_level`` and ``i`` as
                          attributes)
        :param energy_rtol: rtol parameter used for assessing the accuracy of energy levels (see numpy.isclose docs)
        :param energy_atol: atol parameter used for assessing the accuracy of energy levels (see numpy.isclose docs)
        """

        for path_id, path in enumerate(self._paths):
            if self._debug:
                print('\tReading path {}'.format(path))
            found_data = comaux.find_all_files_by_standardised_naming(rp_naming, path)
            if found_data is None or found_data == []:
                raise ValueError("No random perturbations found in {}".format(path))
            for file_, data in found_data:
                energy = data['energy_level']
                try:
                    energy_level_id = index_for_almost_exact_coincidence_unsorted(self._energy_levels, energy,
                                                                                  rtol=energy_rtol,
                                                                                  atol=energy_atol)
                except ValueError:
                    self._energy_levels.append(energy)
                    self._random_perturbations.append([[] for _ in self._paths])
                    energy_level_id = len(self._energy_levels) - 1
                self._random_perturbations[energy_level_id][path_id].append(Perturbation(data))
        energy_level_and_rps_zipped = zip(self._energy_levels, self._random_perturbations)
        energy_level_and_rps_zipped_sorted = sorted(energy_level_and_rps_zipped, key=itemgetter(0))
        self._energy_levels = [e for e, rps in energy_level_and_rps_zipped_sorted]
        self._random_perturbations = [rps for e, rps in energy_level_and_rps_zipped_sorted]
        self._timeintegrations = [None for _ in self._energy_levels]  # first index == energy_level_id,
        if self._debug:
            for e_i in range(len(self._energy_levels)):
                print('Energy level {}'.format(self._energy_levels[e_i]))
                for path_i in range(len(self._paths)):
                    print('\t{} RPs in {}'.format(len(self._random_perturbations[e_i][path_i]), self._paths[path_i]))

    def perturbations(self, energy_level_id: Optional[int] = None) -> Sequence[Perturbation]:
        """
        Returns perturbations belonging to the energy level associated with the index ``energy_level_id``.

        :param energy_level_id: index (with respect to the list returned by LaminarisationStudy.energy_levels()) of the
        energy level at which perturbations should be gathered. If None, all available perturbations will be returned
        :return: list of perturbations as instances of Perturbation
        """
        if energy_level_id is None:
            return functools.reduce(add, functools.reduce(add, self._random_perturbations))
        else:
            return functools.reduce(add, self._random_perturbations[energy_level_id])

    def timeintegrations(self, energy_level_id: Optional[int] = None) -> Sequence[TimeIntegration]:
        """
        Returns TimeIntegration instances for the perturbations belonging to the energy level associated with the
        index ``energy_level_id``. They are sorted accordingly to the perturbations returned by
        LaminarisationStudy.perturbations

        :param energy_level_id: index (with respect to the list returned by LaminarisationStudy.energy_levels()) of the
        energy level at which time-integrations should be gathered. If None, all available time-integrations will be
        returned
        :return: list of TimeIntegration instances
        """
        tis = []
        if energy_level_id is None:
            for i in range(len(self._energy_levels)):
                if self._timeintegrations[i] is None:
                    self._timeintegrations[i] = self._get_ti_at_energy_level(i)
                tis += functools.reduce(add, self._timeintegrations[i])
        else:
            if self._timeintegrations[energy_level_id] is None:
                self._timeintegrations[energy_level_id] = self._get_ti_at_energy_level(energy_level_id)
            tis += functools.reduce(add, self._timeintegrations[energy_level_id])
        return tis

    def make_subsample(self, n: int):
        """
        Return a sub-sample of current LaminarisationStudy implying that at each energy level, there is now n
        perturbations/timeintegrations.

        :param n: number of perturbations/timeintegrations per each energy level
        :return: instance of LaminarisationStudy using a sub-sample of perturbations/simulations
        :rtype LaminarisationStudy

        """
        study = LaminarisationStudy(self._paths, self._data_dir_naming, self._ti_builder)
        study._energy_levels = self.energy_levels
        for e_i in range(len(study._energy_levels)):
            study._random_perturbations.append([[] for _ in study._paths])
            total_rps_per_energy_level = np.sum([len(rps) for rps in self._random_perturbations[e_i]])
            # probability of getting an i-th path
            pk = [len(rps) / float(total_rps_per_energy_level) for rps in self._random_perturbations[e_i]]
            pathrv = scipy.stats.rv_discrete(name='pathrv', values=(np.arange(len(pk)), pk))
            path_indices = pathrv.rvs(size=n)
            for path_i in path_indices:
                rp_i = np.random.randint(len(self._random_perturbations[e_i][path_i]))
                study._random_perturbations[e_i][path_i].append(self._random_perturbations[e_i][path_i][rp_i])
#                study._timeintegrations[e_i].append(self._timeintegrations[e_i][path_i][rp_i])
        study._timeintegrations = [None for _ in study._energy_levels]
        return study

    def _get_ti_at_energy_level(self, energy_level_id: int):
        tis = [[] for _ in self._paths]
        for j, path in enumerate(self._paths):
            if self._debug:
                print('\tReading path {}'.format(path))
            for rp in self._random_perturbations[energy_level_id][j]:
                data_dir = self._data_dir_naming.make_name(energy_level=self._energy_levels[energy_level_id], i=rp.i)
                ti = self._ti_builder.get_timeintegration(os.path.join(path, data_dir))
                tis[j].append(ti)
        return tis


class LaminarisationProbabilityEstimation:
    """
    Class LaminarisationProbabilityEstimation implements a generic algorithm estimating the laminarisation probability
    using the data presented by LaminarisationStudy. It makes use of the strategy pattern to specify a concrete
    estimation algorithm encapsulated in function _make_estimation which must be passed to the constructor.
    """
    def __init__(self, study: LaminarisationStudy,
                 make_estimation: Callable[[Sequence[float], Sequence[float]],
                                           Tuple[Sequence[float], Optional[Sequence[ScipyDistribution]]]]):
        self._study = study
        self._make_estimation = make_estimation

    @classmethod
    def from_bayesian_perspective(cls, study: LaminarisationStudy,
                                  priors: Optional[Callable[[Sequence[float], int], Tuple[float, float]]] = 'jeffreys'):
        return cls(study, functools.partial(make_bayesian_estimation, priors=priors))

    @classmethod
    def from_frequentist_perspective(cls, study: LaminarisationStudy):
        return cls(study, make_frequentist_estimation)

    def estimate(self, selector=lambda rp: rp) -> Tuple[Sequence[float], Optional[Sequence[ScipyDistribution]]]:
        """
        Returns a list of point estimates and a list of distributions (the latter is optional) for the laminarisation
        probability. The indices of the lists correspond to the indices of energy levels in LaminarisationStudy.

        :param selector: function verifying whether a particular perturbation goes to the sample used for
        the estimation (default selector allows for any perturbations). It must take perturbation as the only
        argument and return either the perturbation itself or None
        :return: a list of point estimates and a list of distributions (the latter is optional)
        """
        n_lam_array = []
        n_trans_array = []
        for i in range(len(self._study.energy_levels)):
            rps = self._study.perturbations(i)
            tis = self._study.timeintegrations(i)
            n = len(rps)
            n_lam = 0
            for rp, ti in zip(rps, tis):
                if selector(rp) is not None:
                    if is_relaminarised(ti.max_ke, max_ke_eps=0.2):
                        n_lam += 1
            n_trans = n - n_lam
            n_lam_array.append(n_lam)
            n_trans_array.append(n_trans)
        return self._make_estimation(n_lam_array, n_trans_array)

    @abstractmethod
    def _make_estimation(self, n_lam_array: Sequence[float], n_trans_array: Sequence[float]) \
            -> Tuple[Sequence[float], Optional[Sequence[ScipyDistribution]]]:
        raise NotImplementedError('Must be implemented. It must return an array of point estimates and an array of '
                                  'corresponding distributions (instances of ScipyDistribution) or None')


def make_bayesian_estimation(n_lam_array: Sequence[float], n_trans_array: Sequence[float],
                             priors: Prior = 'jeffreys') \
        -> Tuple[Sequence[float], Optional[Sequence[ScipyDistribution]]]:
    """
    Implements Bayesian estimation of the laminarisation probability. The laminarisation probability then has a
    posterior distribution of the form beta(a_0 + n_lam, b_0 + n_trans), where
    a_0 and b_0 are the prior distribution parameters (the prior is also a beta distribution) and n_lam(n_trans) are
    the number of laminarising (transitioning) RPs at a given energy level. Parameters of the prior distribution
    at i-th energy level are taken from function priors which takes the array of n_lam, the array of n_trans and i as
    inputs. Alternatively, one can specify one of the following standard priors: uniform prior (a_0 = 1, b_0 = 1),
    Jeffreys' prior (a_0 = 1/2, b_0 = 1/2) and Haldane (a_0 = 0, b_0 = 0).
    """
    distrs = []
    point_estimates = []
    for n_lam, n_trans in zip(n_lam_array, n_trans_array):
        if priors == 'uniform':
            a_0 = 1.
            b_0 = 1.
        elif priors == 'jeffreys':
            a_0 = 1./2.
            b_0 = 1./2.
        elif priors == 'haldane':
            a_0 = 0.
            b_0 = 0.
        else:
            a_0, b_0 = priors(n_lam, n_trans_array, len(point_estimates))
        a = a_0 + n_lam
        b = b_0 + n_trans
        distrs.append(ScipyDistribution(scipy.stats.beta, a, b))
        point_estimates.append(distrs[-1].mean())
    return np.array(point_estimates), distrs


def make_frequentist_estimation(n_lam_array: Sequence[float], n_trans_array: Sequence[float]) \
            -> Tuple[Sequence[float], Optional[Sequence[ScipyDistribution]]]:
    """
    Implements frequentist estimation of the laminarisation probability which is simply the number of laminarising
    RPs divided by the total number of RPs at a given energy level.
    """
    point_estimates = [float(n_lam) / (n_lam + n_trans) for n_lam, n_trans in zip(n_lam_array, n_trans_array)]
    distrs = None
    return point_estimates, distrs


class LaminarisationProbabilityFittingFunction(ABC):
    """
    Class LaminarisationProbabilityFittingFunction is an abstraction for a fitting function used to fit the
    laminarisation probability calculated at discrete energy levels. An instance of the derived class can be used with
    operator () to assess the value of the function. For example::

        fitting = SomeConcreteLaminarisationProbabilityFittingFunction(energy_levels, p_lam)
        print(fitting(0.003))
    """
    @abstractmethod
    def __call__(self, e: Union[float, Sequence[float]]):
        raise NotImplementedError('Must be implemented. It must return the function value at certain energy')

    def expected_probability(self, e_max=0.04, noise_distribution=lambda e, e_max: 1./e_max) -> float:
        """
        Returns the expected probability (the limits of the energy for integration: [0; e_max])

        :param e_max: maximum kinetic energy of perturbations
        :param noise_distribution: distribution of the energy of external disturbances
        :return: the expected probability
        """
        es = np.linspace(0., e_max, 201)
        return simps(noise_distribution(es, e_max)*self(es), es)

    def energy_with_99_lam_prob(self, bracket=[0., 0.02]) -> float:
        """
        Returns the energy at which the laminarisation probability is 0.99

        :param bracket: a list of two values: minimum and maximum kinetic energy used for search
        :return: the energy at which the laminarisation probability is 0.99
        """
        sol = root_scalar(lambda e: self(e) - 0.99, bracket=bracket, method='brentq')
        return sol.root

    @abstractmethod
    def energy_at_inflection_point(self) -> float:
        """
        Returns the energy at inflection point

        :return: the energy at inflection point
        """
        raise NotImplementedError('Must be implemented. It must return the energy at inflection point of the fitting '
                                  'function')

    @abstractmethod
    def energy_close_to_asymptote(self, eps=0.01, bracket=[0., 0.1]) -> float:
        """
        Returns the lowest energy above which the laminarisation probability is always within eps around its asymptote.

        :param eps: maximum distance from asymptote
        :param bracket: a list of two values: minimum and maximum kinetic energy used for search
        :return: the energy
        """
        if a + eps >= 1.:
            return 0.
        sol = root_scalar(lambda E: self(E) - a - eps, bracket=bracket, method='brentq')
        return sol.root


