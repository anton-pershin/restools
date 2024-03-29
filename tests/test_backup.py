"""
Do not forget to refactor this as a good package.
Now let's code an example of building TimeIntegration instance
"""

import sys
import os
import operator
from functools import partial, reduce
sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt

import restools.timeintegration as ti
from restools.timeintegration_builders import get_ti_builder
from restools.relaminarisation import upload_relaminarisation_time
import restools.laminarisation_probability as lp
from papers.jfm2020_probabilistic_protocol.data import Summary, RPInfo
from comsdk.misc import print_pretty_dict, load_from_json, dump_to_json
from comsdk.research import Research

d = load_from_json(Summary)
#for conf in d.confs:
#    res = Research.open(conf.res_id)
#    lam_study = lp.LaminarisationStudy.from_tasks(res, tasks=conf.tasks,
#                                                  rp_naming=lp.RandomPerturbationFilenameJFM2020,
#                                                  data_dir_naming=lp.DataDirectoryJFM2020AProbabilisticProtocol,
#                                                  ti_builder=get_ti_builder(),
#                                                  debug=True)
#    d.energy_levels = lam_study.energy_levels
#    conf.rps_info = []
#    for e_i in range(len(lam_study.energy_levels)):
#        print('Energy level {}, id = {}'.format(lam_study.energy_levels[e_i], e_i))
#        conf.rps_info.append([])
#        for rp, ti_ in zip(lam_study.perturbations(e_i), lam_study.timeintegrations(e_i)):
#            is_relam = lp.is_relaminarised(ti_.max_ke)
##            print('A = {}, B = {}, is relam = {}'.format(rp.A, rp.B, is_relam))
#            conf.rps_info[e_i].append(RPInfo(rp.A, rp.B, int(is_relam)))
#dump_json_serializable_object(d)
#raise Exception

#for conf in d.confs:
#    p_lam = []
#    for e_i in range(len(d.energy_levels)):
#        N_total = len(conf.rps_info[e_i])
#        N_lam = reduce(lambda acc, rp_info: acc + rp_info.is_laminarised, conf.rps_info[e_i], 0)
#        p_lam.append(float(N_lam) / N_total)
#    conf.p_lam = p_lam
#dump_json_serializable_object(d)
#raise Exception

# Reading a single simulation from a specific research (WIDE_SIMS_IN_PHASE), specific task #100 and specific directory
# within the task (data-244.3125)
res = Research.open('WIDE_SIMS_IN_PHASE')
builder = get_ti_builder()
ti_obj = builder.get_timeintegration(os.path.join(res.get_task_path(100), 'data-244.3125'))
# Taking a solution at time t = 100
f = ti_obj.solution(100)
print('Print of Field object:')
print(f)
print('\nShape of u-component at t = 100: {}'.format(f.u.shape))
# Taking a simulation configuration (e.g., numerical resolution)
sim_conf = ti_obj.simulation_configuration
print('\nSimulation configuration:')
print_pretty_dict(sim_conf)
# Taking some extra data provided by the solver (e.g., time or xy-averaged kinetic energy)
print('\nSummary data, time:')
print(ti_obj.T)
print('\nXY-averaged kinetic energy (dimensions):')
print(ti_obj.ke_z.shape)

# Reading relaminarisation times based on simulations associated with tasks 100, 101 and 102
builder = get_ti_builder(cache=True, upload_data_extension=partial(upload_relaminarisation_time, max_ke_eps=0.2001))
ti_function = ti.build_timeintegration_sequence(res, [100, 101, 102], builder)
for Re in ti_function.domain:
    print('Relaminarisation time at Re = {} is {}'.format(Re, ti_function.at(Re).t_relam))

# Reading and estimating laminarisation probability (p_lam) based on simulations associated with tasks 100, 101 and 102
# in research called P_LAM_ESTIMATION

res = Research.open('P_LAM_EST  IMATION')
lam_study = lp.LaminarisationStudy.from_tasks(res, tasks=[1, 2, 3],
                                              rp_naming=lp.RandomPerturbationFilenameJFM2020,
                                              data_dir_naming=lp.DataDirectoryJFM2020AProbabilisticProtocol,
                                              ti_builder=get_ti_builder(cache=False))
# Estimating p_lam and taking p_lam distributions at each energy level
est = lp.LaminarisationProbabilityBayesianEstimation(lam_study)
point_estimates, distrs = est.estimate()
p_lam = np.r_[[1.], point_estimates]
# Calculating the first and last deciles based on p_lam distributions
p_lam_lower_deciles = np.r_[[0.], [d.ppf(0.1) for d in distrs]]
p_lam_upper_deciles = np.r_[[0.], [d.ppf(0.9) for d in distrs]]
energy_levels = 0.5 * np.r_[[0.], lam_study.energy_levels]
# Building the fitting function based on p_lam estimates
fitting = lp.LaminarisationProbabilityFittingFunction2020JFM.from_data(energy_levels, p_lam)
es = np.linspace(energy_levels[0], energy_levels[-1], 200)
# Plotting the result
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
bar_width = 0.0003
ax.bar(energy_levels, p_lam, 2*bar_width,
       yerr=np.transpose(np.c_[p_lam - p_lam_lower_deciles, p_lam_upper_deciles - p_lam]),
       alpha=0.3,
       color='red',
       capsize=3)
ax.plot(es, fitting(es), linewidth=2)
ax.grid()
plt.show()

# Subsampling of laminarisation study
lam_substudy = lam_study.make_subsample(5)
for e_i in range(len(lam_substudy.energy_levels)):
    print(len(lam_substudy.perturbations(e_i)), len(lam_substudy.timeintegrations(e_i)))
