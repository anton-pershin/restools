from typing import Optional, Sequence, Union, Callable
from functools import reduce

import numpy as np
from scipy.special import gammainc, gamma
from scipy.optimize import root_scalar, minimize

from restools.laminarisation_probability import LaminarisationProbabilityFittingFunction
from papers.jfm2020_probabilistic_protocol.data import Summary, SingleConfiguration
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
        while self(bracket[1]) - self.asymp - eps > 0.:
            bracket[1] *= 2
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


def plot_p_lam_from_conf(ax, summary: Summary, conf: SingleConfiguration, separate_bars_for_neg_and_pos_B=True,
                         color='blue', bar_width=0.0004, zorder=0, lower_deciles=None, upper_deciles=None, obj_to_rasterize=None,
                         bar_alpha=0.75):
    p_lam_neg_B = np.zeros_like(summary.energy_levels)
    p_lam_pos_B = np.zeros_like(summary.energy_levels)
    p_lam = np.zeros_like(summary.energy_levels)
    for e_i in range(len(summary.energy_levels)):
        def _add_next_rp_info(acc, rp_info):
            # acc = (N_total_neg_B, N_lam_pos_B, N_lam_neg_B)
            if rp_info.B < 0:
                return acc[0] + 1, acc[1], acc[2] + rp_info.is_laminarised
            else:
                return acc[0], acc[1] + rp_info.is_laminarised, acc[2]

        N_total = len(conf.rps_info[e_i])
        N_total_neg_B, N_lam_pos_B, N_lam_neg_B = reduce(_add_next_rp_info, conf.rps_info[e_i], (0, 0, 0))
        N_total_pos_B = N_total - N_total_neg_B
        p_lam_neg_B[e_i] = float(N_lam_neg_B) / N_total_neg_B
        p_lam_pos_B[e_i] = float(N_lam_pos_B) / N_total_pos_B
        p_lam[e_i] = float(N_lam_pos_B + N_lam_neg_B) / N_total

    adjusted_energy_levels = 0.5 * np.r_[[0.], np.array(summary.energy_levels) + np.array(summary.energy_deviations)]
    p_lam_neg_B = np.r_[[1.], p_lam_neg_B]
    p_lam_pos_B = np.r_[[1.], p_lam_pos_B]
    if separate_bars_for_neg_and_pos_B:
        obj = ax.bar(adjusted_energy_levels, p_lam_neg_B / 2., 2*bar_width, alpha=bar_alpha, color='magenta', label=r'$B < 0$')
        if obj_to_rasterize is not None:
            obj_to_rasterize.append(obj)
        obj = ax.bar(adjusted_energy_levels, p_lam_pos_B / 2., 2*bar_width, bottom=p_lam_neg_B / 2., alpha=bar_alpha, color='blue',
               label=r'$B \geq 0$')
        if obj_to_rasterize is not None:
            obj_to_rasterize.append(obj)
    else:
        plot_p_lam(ax, adjusted_energy_levels, np.r_[[1.], p_lam], color=color, bar_width=bar_width, zorder=zorder,
                   lower_deciles=lower_deciles, upper_deciles=upper_deciles, obj_to_rasterize=obj_to_rasterize, bar_alpha=bar_alpha)


def plot_p_lam(ax, energies, p_lam, color='blue', zorder=0, lower_deciles=None, upper_deciles=None, bar_width=0.0004,
               obj_to_rasterize=None, bar_alpha=0.75):
    if lower_deciles is None:
        obj = ax.bar(energies, p_lam, 2*bar_width, alpha=bar_alpha, color=color, zorder=zorder)
    else:
        obj = ax.bar(energies, p_lam, 2*bar_width,
                     yerr=np.transpose(np.c_[p_lam - lower_deciles, upper_deciles - p_lam]), alpha=bar_alpha,
                     color=color, zorder=zorder, capsize=3, ecolor='gray')
    if obj_to_rasterize is not None:
        obj_to_rasterize.append(obj)
