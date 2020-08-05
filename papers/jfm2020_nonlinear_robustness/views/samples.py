import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from restools.timeintegration_builders import get_ti_builder
from restools.plotting import label_axes, rasterise_and_save, reduce_eps_size
from restools.laminarisation_probability import make_bayesian_estimation, make_frequentist_estimation
from papers.jfm2020_nonlinear_robustness.data import Summary
from papers.jfm2020_nonlinear_robustness.extensions import DistributionSummary, \
    find_lam_event_number_by_random_sampling, plot_distribution_summary
from papers.jfm2020_probabilistic_protocol.data import Summary as SummaryProbProto
from papers.jfm2020_probabilistic_protocol.extensions import LaminarisationProbabilityFittingFunction2020JFM, \
    plot_p_lam_from_conf, plot_p_lam
from thequickmath.stats import EmpiricalDistribution
from comsdk.comaux import load_from_json
from comsdk.research import Research


if __name__ == '__main__':
    plt.style.use('resources/default.mplstyle')

    summary = load_from_json(Summary)
    summary_prob_proto = load_from_json(SummaryProbProto)
    ti_builder = get_ti_builder()
    conf_unctrl = summary_prob_proto.confs[1]
    conf_ctrl = summary_prob_proto.confs[3]
    res_unctrl = Research.open(conf_unctrl.res_id)
    res_ctrl = Research.open(conf_ctrl.res_id)
    energies = 0.5 * np.r_[[0.], summary_prob_proto.energy_levels]
    energies_for_plotting = np.linspace(0., np.max(energies), 200)
    fitting_distrs = []

    # PLOT P_LAM FOR UNCONTROLLED SYSTEM

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    obj_to_rasterize = []
    for ax, conf, label in zip((axes[0, 1], axes[1, 1]), (conf_unctrl, conf_ctrl), (r'(b)', r'(d)')):
        print('Processing configuration "{}"'.format(conf.description))
        fitting_values = np.zeros((summary.default_sample_number, len(energies_for_plotting)))
#        fitting_mean = LaminarisationProbabilityFittingFunction2020JFM.from_data(energies, np.r_[[1.], conf.p_lam])
#        fitting_mean_values = fitting_mean(energies_for_plotting)
#        ax.plot(energies_for_plotting, fitting_mean_values, linewidth=2, color='blue')

        print('\tSampling {} RPs'.format(summary.default_sample_number))
        n_lam = find_lam_event_number_by_random_sampling(conf.rps_info, summary.default_sample_number,
                                                         summary.sample_size_per_energy_level, summary.seed)
        print('\tMaking P_lam estimations')
        p_lam_all_means = np.zeros_like(n_lam)
        for s_i in range(summary.default_sample_number):
            p_lam_means, _ = make_frequentist_estimation(n_lam[s_i], summary.sample_size_per_energy_level - n_lam[s_i])
            p_lam_all_means[s_i, :] = p_lam_means
            p_lam = np.r_[[1.], p_lam_means]
            fitting = LaminarisationProbabilityFittingFunction2020JFM.from_data(energies, p_lam)
            if fitting.asymp < 0 or fitting.alpha < 0 or fitting.beta < 0:
                fitting_values[s_i, :] = fitting_values[s_i - 1, :]  # todo: need to fill with another sample!
                continue
            fitting_values[s_i, :] = fitting(energies_for_plotting)
#            ax.plot(energies_for_plotting, fitting_values[s_i, :], linewidth=0.7, color='blue', alpha=0.5)
        print('\tConstructing confidence bands')
        fitting_distr = DistributionSummary()
        for e_i in range(len(energies_for_plotting)):
            distr_at_energy_level = EmpiricalDistribution(fitting_values[:, e_i])
            quantiles = distr_at_energy_level.ppf([0.1, 0.25, 0.75, 0.9])
            fitting_distr.append(mean=distr_at_energy_level.mean(), lower_decile=quantiles[0],
                                 lower_quartile=quantiles[1], upper_quartile=quantiles[2], upper_decile=quantiles[3])

        p_lam_lower_decile = np.zeros((p_lam_all_means.shape[1],))
        p_lam_upper_decile = np.zeros((p_lam_all_means.shape[1],))
        for e_i in range(p_lam_all_means.shape[1]):
            distr_at_energy_level = EmpiricalDistribution(p_lam_all_means[:, e_i])
            deciles = distr_at_energy_level.ppf([0.1, 0.9])
            p_lam_lower_decile[e_i] = deciles[0]
            p_lam_upper_decile[e_i] = deciles[1]

        fitting_distrs.append(fitting_distr)
        plot_p_lam_from_conf(ax, summary_prob_proto, conf, separate_bars_for_neg_and_pos_B=False, color='lightpink',
                             zorder=0, lower_deciles=np.r_[[1.], p_lam_lower_decile],
                             upper_deciles=np.r_[[1.], p_lam_upper_decile], obj_to_rasterize=obj_to_rasterize)
        plot_distribution_summary(ax, fitting_distr, energies_for_plotting, obj_to_rasterize)
        ax.grid()
        label_axes(ax, label=label, loc=(0.5, 1.05), fontsize=16)

    # PLOT ONE RANDOM SAMPLE AND RESULTING CONFIDENCE BAND

    n_per_energy_level = summary.sample_size_per_energy_level
    seed = summary.seed_for_bayesian_example
    for ax_sample, conf, label in zip((axes[0, 0], axes[1, 0]), (conf_unctrl, conf_ctrl), (r'(a)', r'(c)')):
        print('Processing configuration "{}"'.format(conf.description))
        n_lam = find_lam_event_number_by_random_sampling(conf.rps_info, 2, n_per_energy_level, seed)[0]
        p_lam_means, p_lam_distrs = make_bayesian_estimation(n_lam, n_per_energy_level - n_lam)
        p_lam = np.r_[[1.], p_lam_means]
        p_lam_lower_decile = np.r_[[1.], [d.ppf(0.1) for d in p_lam_distrs]]
        p_lam_upper_decile = np.r_[[1.], [d.ppf(0.9) for d in p_lam_distrs]]
        plot_p_lam(ax_sample, energies, p_lam, color='lightpink', zorder=0, lower_deciles=p_lam_lower_decile,
                   upper_deciles=p_lam_upper_decile, obj_to_rasterize=obj_to_rasterize)
        for data, color in zip((p_lam_lower_decile, p_lam_upper_decile), ('red', 'red')):
            data_fitting = LaminarisationProbabilityFittingFunction2020JFM.from_data(energies, data)
            ax_sample.plot(energies_for_plotting, data_fitting(energies_for_plotting), color=color)
        print('\tConstructing confidence bands for a single sample based on Bayesian estimation')
        p_lam_samples = np.array([d.rvs(size=summary.default_sample_number) for d in p_lam_distrs])
        fitting_values = np.zeros((summary.default_sample_number, len(energies_for_plotting)))
        for s_i in range(p_lam_samples.shape[1]):
            p_lam_sample = p_lam_samples[:, s_i]
            p_lam = np.r_[[1.], p_lam_sample]
            fitting = LaminarisationProbabilityFittingFunction2020JFM.from_data(energies, p_lam)
            if fitting.asymp < 0 or fitting.alpha < 0 or fitting.beta < 0:
                fitting_values[s_i, :] = fitting_values[s_i - 1, :]  # todo: need to fill with another sample!
                continue
            fitting_values[s_i, :] = fitting(energies_for_plotting)
        fitting_distr = DistributionSummary()
        for e_i in range(len(energies_for_plotting)):
            distr_at_energy_level = EmpiricalDistribution(fitting_values[:, e_i])
            quantiles = distr_at_energy_level.ppf([0.1, 0.25, 0.75, 0.9])
            fitting_distr.append(mean=distr_at_energy_level.mean(), lower_decile=quantiles[0],
                                 lower_quartile=quantiles[1], upper_quartile=quantiles[2], upper_decile=quantiles[3])
        plot_distribution_summary(ax_sample, fitting_distr, energies_for_plotting, obj_to_rasterize)
        ax_sample.grid()
        label_axes(ax_sample, label=label, loc=(0.5, 1.05), fontsize=16)
    for ax in axes[:, 0]:
        ax.set_ylabel(r'$P_{lam}$')
    for ax in axes[1, :]:
        ax.set_xlabel(r'$E$')
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.2)
    fname = 'p_lam_subsampling.eps'
    plt.savefig(fname)
    rasterise_and_save(fname, rasterise_list=obj_to_rasterize, fig=fig, dpi=300)
    reduce_eps_size(fname)
    plt.show()
