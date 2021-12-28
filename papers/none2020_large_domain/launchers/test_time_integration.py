import os
import sys
sys.path.append(os.getcwd())

from restools.graphs.timeintegration import RemoteIntegrationGraph, SimulateflowChannelflowV2
from papers.none2020_large_domain.extensions import localise_random_field
from comsdk.communication import LocalCommunication, SshCommunication
from comsdk.research import Research


if __name__ == '__main__':
    local_comm = LocalCommunication.create_from_config()
    ssh_comm = SshCommunication.create_from_config('arc4_epfl_cf')
    res = Research.open('LARGE_DOMAIN', ssh_comm)

    a_x = 30.
    b_x = 10.
    a_z = 10.
    b_z = 10.
    c = 0.3
    ic = localise_random_field(os.path.join(res.local_research_path, 'RandomFieldX64piY2Z32pi.nc'),
                               os.path.join(res.local_research_path, 'LocalisedRandomFieldX64piY2Z32pi_Ax_{}_Bx_{}_'
                                                                     'Az_{}_Bz_{}_C_{}.nc'.format(int(a_x), int(b_x),
                                                                                                  int(a_z), int(b_z),
                                                                                                  c)),
                               a_x=a_x, b_x=b_x, a_z=a_z, b_z=b_z, c=c)
    Re = 500
    data = {
        'R': Re,
        'T0': 0,
        'T': 2000,
        'dt': 1./Re,
        'dT': 0.5,  # dT between saved data
        's': 40,  # s dT between saved fields
        'vdt': 'false',
        'e': 0.0000005,
        'res': res,
        'u': ssh_comm.username,
        'time_required': '24:00:00',
        'cores_required': 12,
        'initial_condition': ic,
        'i': 0,
    }
    graph = RemoteIntegrationGraph(res, local_comm, ssh_comm,
                                   SimulateflowChannelflowV2(ic_filename_key='initial_condition'),
                                   #spanning_key='initial_condition',
                                   #spanning_key='R',
                                   task_prefix='TestLocalisedSim_'
                                               'Ax_{}_Bx_{}_Az_{}_Bz_{}_C_{}'.format(int(a_x), int(b_x),
                                                                                     int(a_z), int(b_z),
                                                                                     c))
    graph.initialize_data_for_start(data)
    print(data)
    okay = graph.run(data)
    if not okay:
        print(data['__EXCEPTION__'])
    print(data)
