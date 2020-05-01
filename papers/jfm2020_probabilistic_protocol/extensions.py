from typing import Optional, Sequence, Union, Callable

import numpy as np
from scipy.special import gammainc, gamma
from scipy.optimize import root_scalar, minimize

from restools.laminarisation_probability import LaminarisationProbabilityFittingFunction
import comsdk.comaux as comaux


class RandomPerturbationFilenameJFM2020(comaux.StandardisedNaming):
    """
    Class RandomPerturbationFilenameJFM2020 represents a standardised filename of random perturbations used in study
    Pershin, Beaume, Tobias, JFM, 2020.
    """

    @classmethod
    def parse(cls, name: str) -> Optional[dict]:
        res = super().parse(name)
        if res is None:
            return None
        for key in ('A', 'B', 'energy_level'):
            res[key] = float(res[key])
        res['i'] = int(res['i'])
        return res

    @classmethod
    def regexp_with_substitutions(cls, A=None, B=None, energy_level=None, i=None) -> str:
        # r'^LAM_PLUS_RAND_A_(?P<A>[+-]?\d*\.\d+)_B_(?P<B>[+-]?\d*\.\d+)_(?P<energy_level>\d*\.\d+)_(?P<i>\d+)\.h5'
        res = r'^LAM_PLUS_RAND_A_'
        res += comaux.take_value_if_not_none(A, default='(?P<A>[+-]?\d*\.\d+)')
        res += '_B_'
        res += comaux.take_value_if_not_none(B, default='(?P<B>[+-]?\d*\.\d+)')
        res += '_'
        res += comaux.take_value_if_not_none(energy_level, default='(?P<energy_level>\d*\.\d+)')
        res += '_'
        res += comaux.take_value_if_not_none(i, default='(?P<i>\d+)')
        res += '\.h5'
        return res

    @classmethod
    def make_name(cls, **kwargs):
        comaux.raise_exception_if_arguments_not_in_keywords_or_none(['A', 'B', 'energy_level', 'i'], kwargs)
        return 'LAM_PLUS_RAND_A_{}_B_{}_{}_{}.h5'.format(kwargs['A'], kwargs['B'], kwargs['energy_level'], kwargs['i'])


class DataDirectoryJFM2020AProbabilisticProtocol(comaux.StandardisedNaming):
    """
    Class DataDirectoryJFM2020AProbabilisticProtocol represents a standardised directory name of timeintegrations used
    in study Pershin, Beaume, Tobias, JFM, 2020.
    """

    @classmethod
    def parse(cls, name: str) -> Optional[dict]:
        res = super().parse(name)
        if res is None:
            return None
        res['energy_level'] = float(res['energy_level'])
        res['i'] = int(res['i'])
        return res

    @classmethod
    def regexp_with_substitutions(cls, energy_level=None, i=None) -> str:
        # r'^data-(?P<energy_level>\d*\.\d+)-(?P<i>\d+)'
        res = r'^data-'
        res += comaux.take_value_if_not_none(energy_level, default='(?P<energy_level>\d*\.\d+)')
        res += '-'
        res += comaux.take_value_if_not_none(i, default='(?P<i>\d+)')
        return res

    @classmethod
    def make_name(cls, **kwargs):
        comaux.raise_exception_if_arguments_not_in_keywords_or_none(['energy_level', 'i'], kwargs)
        return 'data-{}-{}'.format(kwargs['energy_level'], kwargs['i'])


class LaminarisationProbabilityFittingFunction2020JFM(LaminarisationProbabilityFittingFunction):
    """
    Laminarisation probability fitting function based on the gamma function and introduced in Pershin, Beaume, Tobias,
    JFM, 2020.
    """
    def __init__(self, alpha, beta, asymp):
        self.alpha = alpha
        self.beta = beta
        self.asymp = asymp

    @classmethod
    def from_data(cls, energy_levels, p_lam, x_0=np.array([0.07, 3., 500.]), lambda_reg=0.):
        def _fun(x, relam_prob, e):
            a = x[0]
            alpha = x[1]
            beta = x[2]
            return np.linalg.norm(gamma_fit(e, a, alpha, beta) - relam_prob)**2 + lambda_reg*alpha

        res_ = minimize(_fun, x_0, (p_lam, energy_levels))
        return LaminarisationProbabilityFittingFunction2020JFM(alpha=res_.x[1], beta=res_.x[2], asymp=res_.x[0])

    def __call__(self, e: Union[float, Sequence[float]]):
        return gamma_fit(e, self.asymp, self.alpha, self.beta)

    def energy_at_inflection_point(self):
        return (self.alpha - 1.) / self.beta if self.alpha >= 1. else None

    def energy_close_to_asymptote(self, eps=0.01, bracket=[0., 0.1]):
        if self.asymp + eps >= 1.:
            return 0.
        sol = root_scalar(lambda e: self(e) - self.asymp - eps, bracket=bracket, method='brentq')
        return sol.root


def gamma_fit(e, asymp, alpha, beta):
    return 1. - (1. - asymp)*gammainc(alpha, beta*e)


def dd_gamma_fit(e, a, alpha, beta):
    return (1. - a) / gamma(alpha) * (e**(alpha-2.) * np.exp(-beta*e) * beta**alpha * (beta*e - alpha + 1.))


def relative_probability_increase(fitting_noctrl: Callable[[np.ndarray], np.ndarray],
                                  fitting_ctrl: Callable[[np.ndarray], np.ndarray],
                                  e_max=0.04):
    """
    Return the relative probability increase under the action of control as introduced in Pershin, Beaume, Tobias,
    JFM, 2020.

    :param fitting_noctrl: laminarisation probability fitting function in the absence of control
    :param fitting_ctrl: laminarisation probability fitting function in the presence of control
    :param e_max: maximum kinetic energy used for integration
    :return: the relative probability increase
    """
    e = np.linspace(0., e_max, 200)
    return np.mean((fitting_ctrl(e) - fitting_noctrl(e)) / fitting_noctrl(e))
