import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from restools.laminarisation_probability import make_bayesian_estimation
from restools.plotting import label_axes, rasterise_and_save, reduce_eps_size
from papers.jfm2020_nonlinear_robustness.data import Summary
from papers.jfm2020_nonlinear_robustness.extensions import DistributionSummary, \
    find_lam_event_number_by_random_sampling, plot_distribution_summary
from papers.jfm2020_probabilistic_protocol.data import Summary as SummaryProbProto
from papers.jfm2020_probabilistic_protocol.extensions import LaminarisationProbabilityFittingFunction2020JFM
from thequickmath.stats import EmpiricalDistribution
from thequickmath.misc import relative_error
from comsdk.comaux import load_from_json


if __name__ == '__main__':
    plt.style.use('resources/default.mplstyle')

    summary = load_from_json(Summary)
    summary_prob_proto = load_from_json(SummaryProbProto)
    conf_unctrl = summary_prob_proto.confs[1]
    conf_ctrl = summary_prob_proto.confs[3]
    energies = 0.5 * np.r_[[0.], summary_prob_proto.energy_levels]

    # PLOT DEPENDENCE OF THE ERROR ON THE NUMBER OF RPS

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    obj_to_rasterize = []
    n_per_energy_level = summary.minimum_sample_size_per_energy_level
    seed = summary.seed_for_bayesian_example
    for ax_row, conf in zip((axes[0, :], axes[1, :]), (conf_unctrl, conf_ctrl)):
        print('Processing configuration "{}"'.format(conf.description))
        p_lam_error_distr = DistributionSummary()
        e_a_error_distr = DistributionSummary()
        e_flex_error_distr = DistributionSummary()
        exact_fitting = LaminarisationProbabilityFittingFunction2020JFM.from_data(energies,
                                                                                  np.r_[[1.], conf.p_lam])
        e_p_exact = exact_fitting.expected_probability()
        e_flex_exact = exact_fitting.energy_at_inflection_point()
        e_a_exact = exact_fitting.energy_close_to_asymptote()
        #rp_numbers = np.arange(1, 40) * n_per_energy_level
        rp_numbers = np.arange(1, 5) * n_per_energy_level
        for rp_num in rp_numbers:
            print('\tCalculating error distribution for N = {} RPs'.format(rp_num))
            samples_num = summary.default_sample_number
            e_p_errors = np.zeros((samples_num,))
            e_a_errors = np.zeros((samples_num,))
            e_flex_errors = np.zeros((samples_num,))
            n_lam = find_lam_event_number_by_random_sampling(conf.rps_info, samples_num, rp_num,
                                                             seed)
            for s_i in range(n_lam.shape[0]):
                p_lam_means, _ = make_bayesian_estimation(n_lam[s_i], rp_num - n_lam[s_i])
                fitting = LaminarisationProbabilityFittingFunction2020JFM.from_data(energies, np.r_[[1.], p_lam_means])
                if fitting.asymp < 0 or fitting.alpha < 0 or fitting.beta < 0:
                    for e in (e_p_errors, e_a_errors, e_flex_errors):
                        e[s_i] = e[s_i - 1]  # todo: need to fill with another sample!
                    continue
                e_p_errors[s_i] = relative_error(fitting.expected_probability(), e_p_exact)
                e_a_errors[s_i] = relative_error(fitting.energy_close_to_asymptote(), e_a_exact)
                if fitting.alpha < 1.:
                    e_flex_errors[s_i] = e_flex_errors[s_i - 1]  # todo: need to fill with another sample!
                    continue
                e_flex_errors[s_i] = relative_error(fitting.energy_at_inflection_point(), e_flex_exact)
            for errors, distr_summary in zip((e_p_errors, e_a_errors, e_flex_errors),
                                             (p_lam_error_distr, e_a_error_distr, e_flex_error_distr)):
                empirical_distr = EmpiricalDistribution(errors)
                quantiles = empirical_distr.ppf([0.1, 0.25, 0.75, 0.9])
                distr_summary.append(mean=empirical_distr.mean(), lower_decile=quantiles[0],
                                     lower_quartile=quantiles[1], upper_quartile=quantiles[2],
                                     upper_decile=quantiles[3])
        for ax, distr_summary in zip(ax_row, (p_lam_error_distr, e_a_error_distr, e_flex_error_distr)):
            plot_distribution_summary(ax, distr_summary, rp_numbers, obj_to_rasterize, means_line_style='x-')
            ax.grid()
    for ax, label in zip(axes[0, :], ('(a) ' + r'$e_S$', '(b) ' + r'$e_a$', '(c) ' + r'$e_{flex}$')):
        label_axes(ax, label=label, loc=(0.5, 1.06), fontsize=16)
    for ax in axes[1, :]:
        ax.set_xlabel(r'$N$')
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.25, wspace=0.2)
    fname = 'errors.eps'
    rasterise_and_save(fname, rasterise_list=obj_to_rasterize, fig=fig, dpi=300)
    reduce_eps_size(fname)
    plt.show()
