"""
Do not forget to refactor this as a good package.
Now let's code an example of building TimeIntegration instance
"""

import sys
import os
sys.path.append(os.getcwd())

import restools.timeintegration as ti
from restools.timeintegration_builders import get_ti_builder
from restools.relaminarisation import upload_relaminarisation_time
from comsdk.comaux import print_pretty_dict
from comsdk.research import Research


res = Research.open('WIDE_SIMS_IN_PHASE')
builder = get_ti_builder()
ti_obj = builder.get_timeintegration(os.path.join(res.get_task_path(100), 'data-244.3125'))
f = ti_obj.solution(100)
print('Print of Field object:')
print(f)
print('Shape of u-component at t = 100: {}'.format(f.u.shape))
sim_conf = ti_obj.simulation_configuration
print('\nSimulation configuration:')
print_pretty_dict(sim_conf)
print('\nSummary data, time:')
print(ti_obj.T)
print('\nKE data:')
print(ti_obj.ke_z.shape)


builder = get_ti_builder(cache=True, upload_data_extension=upload_relaminarisation_time)
ti_function = ti.build_timeintegration_sequence(res, [100, 101, 102], builder)
for Re in ti_function.domain:
    print('Relaminarisation time at Re = {} is {}'.format(Re, ti_function.at(Re).t_relam))

res = Research.create('SOME_TEST', 'Some test research for fun')

#print_pretty_dict(ti_obj.summary_data)
