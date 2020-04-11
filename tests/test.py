"""
Do not forget to refactor this as a good package.
Now let's code an example of building TimeIntegration instance
"""

import sys
import os
sys.path.append(os.getcwd())

import restools.timeintegration as ti
from comsdk.comaux import print_pretty_dict
from comsdk.research import Research


class RelaminarisationTimeBuilder(ti.NoBackupAccessBuilder):
    def __init__(self):
        ti.NoBackupAccessBuilder.__init__(self, ti.TimeIntegrationInOldChannelFlow)

    def create_transform(self) -> None:
        def transform_(d) -> dict:
            max_ke_eps = 0.2
            out = {'t_relam': None}
            for i in range(len(d['T'])):
                if d['max_KE'][i] < max_ke_eps:
                    out['t_relam'] = d['T'][i]
                    break
            return out
        self._transform = transform_


builder = ti.NoBackupAccessBuilder(ti.TimeIntegrationInOldChannelFlow)
director = ti.TimeIntegrationBuildDirector(builder)
director.construct()
ti_obj = builder.get_timeintegration('/path/to/data')
f = ti_obj.solution(100)
print('Shape of u-component at t = 100: {}'.format(f.u.shape))
sim_conf = ti_obj.simulation_configuration
print('\nSimulation configuration:')
print_pretty_dict(sim_conf)
print('\nSummary data:')
print(ti_obj.summary_data['T'])

#builder = RelaminarisationTimeBuilder()
#director = ti.TimeIntegrationBuildDirector(builder)
#director.construct()
#res = Research.create_from_config('WIDE_SIMS_IN_PHASE')
#ti_function = ti.build_timeintegration_sequence(res, [100, 101, 102], builder)
#for Re in ti_function.domain:
#    print('Relaminarisation time at Re = {} is {}'.format(Re, ti_function.at(Re).summary_data['t_relam']))

#print_pretty_dict(ti_obj.summary_data)
