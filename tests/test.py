"""
Do not forget to refactor this as a good package.
Now let's code an example of building TimeIntegration instance
"""

import sys
import os
sys.path.append(os.getcwd())

import restools.timeintegration as ti
import restools.timeintegration_builders as ti_builders
from comsdk.comaux import print_pretty_dict
from comsdk.research import Research


builder = ti_builders.get_ti_builder(xy_averaged_quantities=['ke'])
ti_obj = builder.get_timeintegration('/media/tony/WD_My_Passport_2TB/Leeds/PhD/research/'
                                     '2018-11-13_Simulations_for_in-phase_oscillations_in_wide_domain/'
                                     '100-S13_ContinuingTimeIntegration_R_244.3125_246.6875_0.125_A_0.001_omega_0.0625/'
                                     'data-244.3125')
f = ti_obj.solution(100)
print('Shape of u-component at t = 100: {}'.format(f.u.shape))
sim_conf = ti_obj.simulation_configuration
print('\nSimulation configuration:')
print_pretty_dict(sim_conf)
print('\nSummary data:')
print(ti_obj.scalar_series['T'])
print('\nKE data:')
print(ti_obj.vector_series.ke.shape)

builder = ti_builders.RelaminarisationTimeBuilder()
director = ti.TimeIntegrationBuildDirector(builder)
director.construct([])
res = Research.open('WIDE_SIMS_IN_PHASE')
ti_function = ti.build_timeintegration_sequence(res, [100, 101, 102], builder)
for Re in ti_function.domain:
    print('Relaminarisation time at Re = {} is {}'.format(Re, ti_function.at(Re).scalar_series['t_relam']))

res = Research.create('SOME_TEST', 'Some test research for fun')

#print_pretty_dict(ti_obj.summary_data)
