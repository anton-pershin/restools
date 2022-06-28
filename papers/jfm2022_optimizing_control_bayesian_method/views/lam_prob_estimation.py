import os
import sys
import argparse
sys.path.append(os.getcwd())

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from restools.timeintegration_builders import get_ti_builder
from restools.plotting import label_axes
from restools.laminarisation_probability import LaminarisationStudy, LaminarisationProbabilityEstimation
from papers.jfm2022_optimizing_control_bayesian_method.data import Summary
from papers.jfm2022_optimizing_control_bayesian_method.extensions import DistributionSummary, exponential_noise_distribution
from papers.jfm2020_probabilistic_protocol.data import Summary as SummaryProbProto
from papers.jfm2020_probabilistic_protocol.extensions import RandomPerturbationFilenameJFM2020, \
    DataDirectoryJFM2020AProbabilisticProtocol, LaminarisationProbabilityFittingFunction2020JFM
from thequickmath.stats import EmpiricalDistribution
from comsdk.comaux import load_from_json, dump_to_json
from comsdk.research import Research


def _load_from_tasks(tasks, amplitude, frequency, s_distr_summary, s_exp_distr_summary, e_a_distr_summary, e_flex_distr_summary):
    if amplitude == 0.3 and frequency == 0.0625:
        res = res_exception
    else:
        res = res_default
    if isinstance(tasks, int):
        tasks = [tasks]
    lam_study = LaminarisationStudy.from_tasks(res, tasks,
                                               RandomPerturbationFilenameJFM2020,
                                               DataDirectoryJFM2020AProbabilisticProtocol,
                                               ti_builder)
    p_lam_means, p_lam_distrs = LaminarisationProbabilityEstimation.from_bayesian_perspective(lam_study).estimate()
    p_lam_samples = np.array([d.rvs(size=summary.default_sample_number) for d in p_lam_distrs])
    s_values = np.zeros((summary.default_sample_number,))
    s_exp_values = np.zeros((summary.default_sample_number,))
    e_a_values = np.zeros((summary.default_sample_number,))
    e_flex_values = np.zeros((summary.default_sample_number,))
    print('\tGenerating samples...')
    for s_i in range(summary.default_sample_number):
        p_lam_sample = p_lam_samples[:, s_i]
        p_lam = np.r_[[1.], p_lam_sample]
        fitting = LaminarisationProbabilityFittingFunction2020JFM.from_data(energies, p_lam)
        if fitting.asymp < 0 or fitting.alpha < 1 or fitting.beta < 0:
            s_values[s_i] = s_values[s_i - 1]  # todo: need to fill with another sample!
            s_exp_values[s_i] = s_exp_values[s_i - 1]  # todo: need to fill with another sample!
            e_a_values[s_i] = e_a_values[s_i - 1]  # todo: need to fill with another sample!
            e_flex_values[s_i] = e_flex_values[s_i - 1]  # todo: need to fill with another sample!
            print('\tBad sample')
            continue
        s_values[s_i] = fitting.expected_probability()
        s_exp_values[s_i] = fitting.expected_probability(noise_distribution=exponential_noise_distribution)
        e_a_values[s_i] = fitting.energy_close_to_asymptote()
        e_flex_values[s_i] = fitting.energy_at_inflection_point()
    for d_summary, sampled_values in zip((s_distr_summary, s_exp_distr_summary, e_a_distr_summary, e_flex_distr_summary),
                                         (s_values, s_exp_values, e_a_values, e_flex_values)):
        quantity_distr = EmpiricalDistribution(sampled_values)
        deciles = quantity_distr.ppf([0.1, 0.9])
        d_summary.append(mean=quantity_distr.mean(), lower_decile=deciles[0], upper_decile=deciles[1])


def _load_from_dump(summary, a_i, freq_j, s_distr_summary, s_exp_distr_summary, e_a_distr_summary, e_flex_distr_summary):
    for d_summary, triple_stats in zip((s_distr_summary, s_exp_distr_summary, e_a_distr_summary, e_flex_distr_summary),
                                       (summary.p_lam_info.s[a_i][freq_j], summary.p_lam_info.s_exp[a_i][freq_j],
                                        summary.p_lam_info.e_a[a_i][freq_j], summary.p_lam_info.e_flex[a_i][freq_j])):
        d_summary.append(mean=triple_stats[1], lower_decile=triple_stats[0], upper_decile=triple_stats[2])


if __name__ == '__main__':
    plt.style.use('resources/default.mplstyle')
    parser = argparse.ArgumentParser(description='Plots laminarisation probability obtained by Bayesian estimation.')
    parser.add_argument('mode', metavar='MODE', nargs='?', choices=['loadfromdump', 'dump'], default='loadfromdump',
                        help='decides where the data should be taken from and whether the data is dumped '
                             '(must be either loadfromdump or dump)')
    args = parser.parse_args()
    summary = load_from_json(Summary)
    summary_prob_proto = load_from_json(SummaryProbProto)
    ti_builder = get_ti_builder()
    res_default = Research.open(summary.p_lam_info.res_id)
    res_exception = Research.open(summary.p_lam_info.res_id_exception_for_A_03_omega_1_16)
    energies = 0.5 * np.r_[[0.], summary_prob_proto.energy_levels]

    if args.mode == 'dump':
        summary.p_lam_info.s = []
        summary.p_lam_info.s_exp = []
        summary.p_lam_info.e_a = []
        summary.p_lam_info.e_flex = []

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax_s = axes[0][0]
    ax_s_exp = axes[0][1]
    ax_e_a = axes[1][0]
    ax_e_flex = axes[1][1]

    print(summary_prob_proto.confs[1].description)
    fitting_no_ctrl = LaminarisationProbabilityFittingFunction2020JFM.from_data(
        0.5 * np.array([0.] + summary_prob_proto.energy_levels), np.array([1.] + summary_prob_proto.confs[1].p_lam))
    lam_score_no_ctrl = fitting_no_ctrl.expected_probability()
    lam_score_exp_no_ctrl = fitting_no_ctrl.expected_probability(noise_distribution=exponential_noise_distribution)
    e_a_no_ctrl = fitting_no_ctrl.energy_close_to_asymptote()
    e_flex_no_ctrl = fitting_no_ctrl.energy_at_inflection_point()
    for ax, q in zip((ax_s, ax_s_exp, ax_e_a, ax_e_flex),
                     (lam_score_no_ctrl, lam_score_exp_no_ctrl, e_a_no_ctrl, e_flex_no_ctrl)):
        ax.plot(summary.p_lam_info.frequencies[::-1], len(summary.p_lam_info.frequencies)*[q],
                'k-', label=r'$W_{osc} = 0$')

    # PLOT ONE RANDOM SAMPLE AND RESULTING CONFIDENCE BAND

    n_per_energy_level = summary.sample_size_per_energy_level
    seed = summary.seed
    for i, amplitude in enumerate(summary.p_lam_info.amplitudes):
        s_exp_distr_summary = DistributionSummary()
        s_distr_summary = DistributionSummary()
        e_a_distr_summary = DistributionSummary()
        e_flex_distr_summary = DistributionSummary()
        for j, frequency in enumerate(summary.p_lam_info.frequencies):
            print('Processing A = {}, omega = {}'.format(amplitude, frequency))
            if args.mode == 'dump':
                _load_from_tasks(summary.p_lam_info.tasks[i][j], amplitude, frequency, s_distr_summary,
                                 s_exp_distr_summary, e_a_distr_summary, e_flex_distr_summary)
            elif args.mode == 'loadfromdump':
                _load_from_dump(summary, i, j, s_distr_summary, s_exp_distr_summary, e_a_distr_summary,
                                e_flex_distr_summary)
        for ax, d_summary in zip((ax_s, ax_s_exp, ax_e_a, ax_e_flex),
                                 (s_distr_summary, s_exp_distr_summary, e_a_distr_summary, e_flex_distr_summary)):
            means = np.array(d_summary.means)
            lower_deciles = np.array(d_summary.lower_deciles)
            upper_deciles = np.array(d_summary.upper_deciles)
            ax.errorbar(summary.p_lam_info.frequencies[::-1], means, fmt='o-',
                        yerr=np.transpose(np.c_[means - lower_deciles, upper_deciles - means]), capsize=3,
                        label=r'$W_{osc} = ' + str(amplitude) + r'$')

        if args.mode == 'dump':
            for d_summary, storage in zip((s_distr_summary, s_exp_distr_summary,
                                           e_a_distr_summary, e_flex_distr_summary),
                                          (summary.p_lam_info.s, summary.p_lam_info.s_exp,
                                           summary.p_lam_info.e_a, summary.p_lam_info.e_flex)):
                storage.append([[lower_decile, mean, upper_decile]
                                for lower_decile, mean, upper_decile in
                                zip(d_summary.lower_deciles, d_summary.means, d_summary.upper_deciles)])
    if args.mode == 'dump':
        dump_to_json(summary)
    for ax, ylabel, title in zip((ax_s, ax_s_exp, ax_e_a, ax_e_flex), (r'$S$', r'$S$', r'$E_a$', r'$E_{flex}$'),
                                 (r'(a)', r'(b)', r'(c)', r'(d)')):
        ax.grid()
        ax.set_xlabel(r'$\omega$')
        ax.set_ylabel(ylabel)
        ax.set_xscale('log', basex=2)
        ax.set_xticks(summary.p_lam_info.frequencies)
        ax.set_xticklabels([r'$2^{' + str(int(np.log2(summary.p_lam_info.frequencies[i]))) + r'}$'
                            for i in range(len(summary.p_lam_info.frequencies))])
        label_axes(ax, label=title, loc=(0.47, 1.05), fontsize=16)
    ax_s.set_ylim((0., 1.))
    ax_s_exp.set_ylim((0.4, 1.))
#    ax_s_exp.legend(bbox_to_anchor=(0.1, 1.05, 0.8, 0.1), loc='upper center',
#                    ncol=5, fancybox=True, fontsize=12)
    ax_s_exp.legend(bbox_to_anchor=(0.75, 0.0, 0.9, 1.02), loc='upper center',
                    ncol=1, fancybox=True)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, right=0.86, wspace=0.26)
    fname = 'estimation.eps'
    plt.savefig(fname)
#    rasterise_and_save(fname, rasterise_list=obj_to_rasterize, fig=fig, dpi=300)
#    reduce_eps_size(fname)
    plt.show()

    # PLOT ONLY LAMINARIZATION SCORE FOR THE EXPONENTIAL LAW FOR POSTER

    if args.mode == 'dump':
        summary.p_lam_info.s_exp = []

    fig, ax = plt.subplots(1, figsize=(7, 5))
    print(summary_prob_proto.confs[1].description)
    fitting_no_ctrl = LaminarisationProbabilityFittingFunction2020JFM.from_data(
        0.5 * np.array([0.] + summary_prob_proto.energy_levels), np.array([1.] + summary_prob_proto.confs[1].p_lam))
    lam_score_exp_no_ctrl = fitting_no_ctrl.expected_probability(noise_distribution=exponential_noise_distribution)
    ax.plot(summary.p_lam_info.frequencies[::-1], len(summary.p_lam_info.frequencies)*[lam_score_exp_no_ctrl],
           'k-', label=r'$W_{osc} = 0$')

    n_per_energy_level = summary.sample_size_per_energy_level
    seed = summary.seed
    for i, amplitude in enumerate(summary.p_lam_info.amplitudes):
        s_exp_distr_summary = DistributionSummary()
        s_distr_summary = DistributionSummary()
        e_a_distr_summary = DistributionSummary()
        e_flex_distr_summary = DistributionSummary()
        for j, frequency in enumerate(summary.p_lam_info.frequencies):
            print('Processing A = {}, omega = {}'.format(amplitude, frequency))
            if args.mode == 'dump':
                _load_from_tasks(summary.p_lam_info.tasks[i][j], amplitude, frequency, s_distr_summary,
                                 s_exp_distr_summary, e_a_distr_summary, e_flex_distr_summary)
            elif args.mode == 'loadfromdump':
                _load_from_dump(summary, i, j, s_distr_summary, s_exp_distr_summary, e_a_distr_summary,
                                e_flex_distr_summary)
        means = np.array(s_exp_distr_summary.means)
        lower_deciles = np.array(s_exp_distr_summary.lower_deciles)
        upper_deciles = np.array(s_exp_distr_summary.upper_deciles)
        ax.errorbar(summary.p_lam_info.frequencies[::-1], means, fmt='o-',
                    yerr=np.transpose(np.c_[means - lower_deciles, upper_deciles - means]), capsize=3,
                    label=r'$W_{osc} = ' + str(amplitude) + r'$')

        if args.mode == 'dump':
            for d_summary, storage in zip((s_distr_summary, s_exp_distr_summary,
                                           e_a_distr_summary, e_flex_distr_summary),
                                          (summary.p_lam_info.s, summary.p_lam_info.s_exp,
                                           summary.p_lam_info.e_a, summary.p_lam_info.e_flex)):
                storage.append([[lower_decile, mean, upper_decile]
                                for lower_decile, mean, upper_decile in
                                zip(d_summary.lower_deciles, d_summary.means, d_summary.upper_deciles)])
    if args.mode == 'dump':
        dump_to_json(summary)
    ax.grid()
    ax.set_xlabel(r'$\omega$')
    ax.set_ylabel(r'$S$')
    ax.set_xscale('log', basex=2)
    ax.set_xticks(summary.p_lam_info.frequencies)
    ax.set_xticklabels([r'$2^{' + str(int(np.log2(summary.p_lam_info.frequencies[i]))) + r'}$'
                        for i in range(len(summary.p_lam_info.frequencies))])
    ax.set_ylim((0.4, 1.))
#    ax_s_exp.legend(bbox_to_anchor=(0.1, 1.05, 0.8, 0.1), loc='upper center',
#                    ncol=5, fancybox=True, fontsize=12)
    ax.legend(bbox_to_anchor=(0.8, 0.0, 0.9, 1.02), loc='upper center',
              ncol=1, fancybox=True)
#    ax.legend()
    plt.tight_layout(rect=[0, 0, 1.05, 1])
    fname = 'lam_score_exp_poster.eps'
    plt.savefig(fname)
#    rasterise_and_save(fname, rasterise_list=obj_to_rasterize, fig=fig, dpi=300)
#    reduce_eps_size(fname)
    plt.show()

    # PLOT ONLY LAMINARIZATION SCORE FOR THE UNIFORM AND EXPONENTIAL LAWS FOR PRESENTATION

    if args.mode == 'dump':
        summary.p_lam_info.s = []
        summary.p_lam_info.s_exp = []

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    print(summary_prob_proto.confs[1].description)
    fitting_no_ctrl = LaminarisationProbabilityFittingFunction2020JFM.from_data(
        0.5 * np.array([0.] + summary_prob_proto.energy_levels), np.array([1.] + summary_prob_proto.confs[1].p_lam))
    lam_score_exp_no_ctrl = fitting_no_ctrl.expected_probability(noise_distribution=exponential_noise_distribution)
    lam_score_uniform_no_ctrl = fitting_no_ctrl.expected_probability()
    for ax, lam_score in zip(axes, (lam_score_uniform_no_ctrl, lam_score_exp_no_ctrl)):
        ax.plot(summary.p_lam_info.frequencies[::-1], len(summary.p_lam_info.frequencies)*[lam_score],
               'k-', label=r'$W_{osc} = 0$')
    n_per_energy_level = summary.sample_size_per_energy_level
    seed = summary.seed
    for i, amplitude in enumerate(summary.p_lam_info.amplitudes):
        s_exp_distr_summary = DistributionSummary()
        s_distr_summary = DistributionSummary()
        e_a_distr_summary = DistributionSummary()
        e_flex_distr_summary = DistributionSummary()
        for j, frequency in enumerate(summary.p_lam_info.frequencies):
            print('Processing A = {}, omega = {}'.format(amplitude, frequency))
            if args.mode == 'dump':
                _load_from_tasks(summary.p_lam_info.tasks[i][j], amplitude, frequency, s_distr_summary,
                                 s_exp_distr_summary, e_a_distr_summary, e_flex_distr_summary)
            elif args.mode == 'loadfromdump':
                _load_from_dump(summary, i, j, s_distr_summary, s_exp_distr_summary, e_a_distr_summary,
                                e_flex_distr_summary)
        means = np.array(s_distr_summary.means)
        lower_deciles = np.array(s_distr_summary.lower_deciles)
        upper_deciles = np.array(s_distr_summary.upper_deciles)
        axes[0].errorbar(summary.p_lam_info.frequencies[::-1], means, fmt='o-',
                    yerr=np.transpose(np.c_[means - lower_deciles, upper_deciles - means]), capsize=3,
                    label=r'$W_{osc} = ' + str(amplitude) + r'$')
        means = np.array(s_exp_distr_summary.means)
        lower_deciles = np.array(s_exp_distr_summary.lower_deciles)
        upper_deciles = np.array(s_exp_distr_summary.upper_deciles)
        axes[1].errorbar(summary.p_lam_info.frequencies[::-1], means, fmt='o-',
                    yerr=np.transpose(np.c_[means - lower_deciles, upper_deciles - means]), capsize=3,
                    label=r'$W_{osc} = ' + str(amplitude) + r'$')

        if args.mode == 'dump':
            for d_summary, storage in zip((s_distr_summary, s_exp_distr_summary,
                                           e_a_distr_summary, e_flex_distr_summary),
                                          (summary.p_lam_info.s, summary.p_lam_info.s_exp,
                                           summary.p_lam_info.e_a, summary.p_lam_info.e_flex)):
                storage.append([[lower_decile, mean, upper_decile]
                                for lower_decile, mean, upper_decile in
                                zip(d_summary.lower_deciles, d_summary.means, d_summary.upper_deciles)])
    if args.mode == 'dump':
        dump_to_json(summary)
    for ax in axes:
        ax.grid()
        ax.set_xlabel(r'$\omega$')
        ax.set_xscale('log', basex=2)
        ax.set_xticks(summary.p_lam_info.frequencies)
        ax.set_xticklabels([r'$2^{' + str(int(np.log2(summary.p_lam_info.frequencies[i]))) + r'}$'
                            for i in range(len(summary.p_lam_info.frequencies))])
    axes[0].set_title(r'Uniform energy distribution')
    axes[1].set_title(r'Exponential energy distribution')
    axes[0].set_ylabel(r'$S$')
    axes[0].set_ylim((0., 1.))
    axes[1].set_ylim((0., 1.))
#    ax_s_exp.legend(bbox_to_anchor=(0.1, 1.05, 0.8, 0.1), loc='upper center',
#                    ncol=5, fancybox=True, fontsize=12)
    ax.legend(bbox_to_anchor=(0.75, 0.0, 0.9, 1.02), loc='upper center',
              ncol=1, fancybox=True)
#    ax.legend()
    plt.tight_layout(rect=[0, 0, 1., 1])
    fname = 'lam_score_uni_and_exp_pres.eps'
    plt.savefig(fname)
#    rasterise_and_save(fname, rasterise_list=obj_to_rasterize, fig=fig, dpi=300)
#    reduce_eps_size(fname)
    plt.show()

