import os
import sys
sys.path.append(os.getcwd())
import time
import json

import numpy as np

from restools.standardised_programs import StandardisedProgramEdge, StandardisedIntegrator, MoehlisModelIntegrator, EsnIntegrator, EsnTrainer, nohup_command_start_and_end
from restools.timeintegration import TimeIntegrationLowDimensional
from papers.none2021_predicting_transition_using_reservoir_computing.extensions import LocalPythonTimeIntegrationGraph,\
    RemotePythonTimeIntegrationGraph
from comsdk.communication import LocalCommunication, SshCommunication
from comsdk.research import Research, CreateTaskGraph
from comsdk.graph import Graph, State, Func
from comsdk.edge import Edge, dummy_predicate, make_dump, make_composite_func, make_cd
import comsdk.comaux as aux
#from comsdk.comaux import ProxyDict


job_finished_predicate = Func(func=lambda d: check_job_finished(d))
job_unfinished_predicate = Func(func=lambda d: not check_job_finished(d))
def check_job_finished(data):
    cd_for_check(data, 'optimal_esn_filename')
    job_finished = False
    if os.path.exists(os.path.join(data['__WORKING_DIR__'], data['finished'])):
        job_finished = True
    else:
        time.sleep(2)
    cd_for_check(data, '..')
    return job_finished

def cd_for_check(d, key_path):
        if key_path == '..':
            d['__WORKING_DIR__'] = os.path.dirname(d['__WORKING_DIR__'])
        else:
            subdir = aux.recursive_get(d, key_path)
            d['__WORKING_DIR__'] = os.path.join(d['__WORKING_DIR__'], subdir)

def make_esn_dirs(data):
    def _make_esn_dirs(data):
        esn_dir = os.path.join(data['__WORKING_DIR__'], f"{data['optimal_esn_filename']}")
        os.mkdir(esn_dir)
    return _make_esn_dirs
   

def make_esn_path(data, input_filename):
    def _make_esn_path(data):
#        if not 'esn_path' in data:
        data['esn_path'] = os.path.join(data['__WORKING_DIR__'], data['optimal_esn_filename'],data['optimal_esn_filename'])
    return _make_esn_path

def make_optimal_esn_filename(data1, data2):
 #   def _make_optimal_esn_filename():
    data1['optimal_esn_filename'] = data2['optimal_esn_filename']
#    return _make_optimal_esn_filename

def make_little_dump(input_filename, omit=None, chande_dir=True):
    def _little_dump(d):
        if omit is None:
            dumped_d = d
        else:
            if (isinstance(d, aux.ProxyDict)):
                dumped_d = {key: val for key, val in d._data.items() if not key in omit}
            else:
                dumped_d = {key: val for key, val in d.items() if not key in omit}
     #   if (esn_name):
      #      make_optimal_esn_filename(dumped_d, d)
        dumped_d['optimal_esn_filename'] = d['optimal_esn_filename']
        dumped_d['output_filenames'] = d['output_filenames']
        dumped_d['finished'] = d['finished']
        dumped_d['started'] = d['started']
        if (chande_dir):
            cd_for_check(dumped_d, 'optimal_esn_filename')
        dump_path = os.path.join(dumped_d['__WORKING_DIR__'], d[input_filename])
        with open(dump_path, 'w') as f:
            json.dump(dumped_d, f)
        if (chande_dir):
            cd_for_check(dumped_d, '..')
    return _little_dump

class ESNTrainAndIntergateGraph(Graph):
    def __init__(self, res, comm, input_filename, task_prefix=''):
        def task_name_maker(d):
            task_name = task_prefix
            task_name += '_R_{}'.format(d['re'])
            task_name += '_T_{}'.format(d['final_time'])
            task_name += '_ens_{}'.format(len(d['initial_conditions']))
            return task_name
    # (task_start) -> (task_end) -пустое ребро-> (tt_init) -train-> (tt_term) -fin-> (ti_init) -integrate-> (ti_term)
    #                                                               (not fin)^
    #                                          ^тут начинается subgraph
#        state_for_keys_mapping = State('START Making implicit parallelization', array_keys_mapping={'input_filename':'input_filenames', 'optimal_esn_filename':'optimal_esn_filenames'})
        #!   #, array_keys_mapping={'input_filename':'input_filenames', 'optimal_esn_filename':'optimal_esn_filenames', 'output_filenames':'output_filenames', 'finished':'finished', 'started':'started', 'pipes_index':'pipes_index'})
        state_for_keys_mapping = State('START Making implicit parallelization', array_keys_mapping={'input_filename':'input_filenames', 'optimal_esn_filename':'optimal_esn_filenames', 'pipes_index':'pipes_index'})
        state_for_cd_esn_dir = State('START Create dir for esn')
        state_for_optimal_esn_filename = State('START Making names for esn files')
        
        tt_init, tt_term, train_edge = LocalPythonTimeIntegrationGraph.create_branch(comm, EsnTrainer(input_filename_key='input_filename', nohup=True, pipes_index_key='pipes_index'), edge_need=True)#, array_keys_mapping={'input_filename':'input_filenames', 'optimal_esn_filename':'optimal_esn_filenames'})
        ti_init, ti_term, integrate_edge = LocalPythonTimeIntegrationGraph.create_branch(comm, EsnIntegrator(input_filename_key='input_filename', nohup=True, pipes_index_key='pipes_index'), edge_need=True)
    
        dummy_edge = Edge(dummy_predicate, Func())
        edge_esn_dir = Edge(dummy_predicate, Func())
        edge_esn_name = Edge(dummy_predicate, Func())
        
        qstat_edge = Edge(job_unfinished_predicate, Func())
        done_edge = Edge(job_finished_predicate, Func())
        
        state_for_keys_mapping.connect_to(state_for_optimal_esn_filename, edge=dummy_edge)
        state_for_optimal_esn_filename.connect_to(state_for_cd_esn_dir, edge=edge_esn_name)
        state_for_cd_esn_dir.connect_to(tt_init, edge=edge_esn_dir)

     ###!   
      ###  state_for_cd_esn_dir.connect_to(state_for_optimal_esn_filename, edge=edge_esn_dir)
      ###  state_for_optimal_esn_filename.connect_to(tt_init, edge=edge_esn_name)
        
        
        tt_term.connect_to(tt_term, edge=qstat_edge)
        tt_term.connect_to(ti_init, edge=done_edge)
        
        
     #!
        edge_esn_dir.use_proxy_data = True
        edge_esn_name.use_proxy_data = True
        done_edge.use_proxy_data = True

      ###  edge_esn_name.preprocess = make_composite_func(make_cd('optimal_esn_filename'))
        #!
        edge_esn_dir.preprocess = make_composite_func(make_esn_dirs(data), make_little_dump('input_filename', omit=['res']))#, make_cd('optimal_esn_filename'))#, make_little_dump('input_filename', omit=['res']))
        edge_esn_name.postprocess = make_composite_func(make_little_dump('input_filename', omit=['res'], chande_dir=False))#, make_esn_dirs(data))
     
        train_edge.use_proxy_data = True
        integrate_edge.use_proxy_data = True
        train_edge.preprocess=make_cd('optimal_esn_filename')
        train_edge.postprocess=make_cd('..')
        integrate_edge.preprocess=make_cd('optimal_esn_filename')
        integrate_edge.postprocess=make_cd('..')
     ##   edge_esn_dir.postprocess = make_composite_func(make_cd('..'))#, make_little_dump('input_filename', omit=['res']))
        
     ##   done_edge.preprocess = make_composite_func(make_cd('..'), make_cd('optimal_esn_filename'))
        done_edge.postprocess = make_composite_func(make_esn_path(data, input_filename), make_little_dump('input_filename', omit=['res']))# make_dump(input_filename, omit=['res'], method='json'))  
        
        subgraph = Graph(state_for_keys_mapping, ti_term)
        
    # (task_start) -> (task_end) -пустое ребро->  (subgraph_state)   
        task_start, task_end = CreateTaskGraph.create_branch(res, task_name_maker=task_name_maker)#, array_keys_mapping={'input_filename':'input_filename'})
        subgraph_state = State('START Working with ESN')
        dumping_edge = Edge(dummy_predicate, Func())
        task_end.connect_to(subgraph_state, edge=dumping_edge)
        dumping_edge.postprocess = make_composite_func(make_dump(input_filename, omit=['res'], method='json'))

        subgraph_state.replace_with_graph(subgraph)
        super().__init__(task_start, ti_term)


if __name__ == '__main__':
    local_comm = LocalCommunication.create_from_config()
#    ssh_comm = SshCommunication.create_from_config('atmlxint2')
#    res = Research.open('RC_MOEHLIS', ssh_comm)
    res = Research.open('RC_MOEHLIS')

   # ics = [[0,1,2,3,4,5,6,7]]        #initial condition for all ESNs
    ics = []
    source_task = 45
    for ic_i in range(1, 2):
        ti = TimeIntegrationLowDimensional(res.get_task_path(source_task), ic_i)
        #q = ti.timeseries - np.array([1., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=np.float64)
        #ti = TimeIntegrationLowDimensional(res.get_task_path(source_task), 1)
        ics.append(ti.timeseries[:10].tolist())
        #ics.append(ti.timeseries[begin_time-10:begin_time].tolist())
    n_ics = len(ics)
    source_task = 45
    #begin_time = 13940 + 100 + 100 + 100
    n_ESNs = 3     #number of ESN 
   
    re = 275
    esn_name = f'esn_re_{re}'
    #esn_name = 'esn_trained_wo_lam_event'
    l_x = 1.75
    l_z = 1.2
    n_steps = 20000

    task = res._get_next_task_number()
    res._tasks_number -= 1
    task_prefix=f'ESNEnsemble'
    
    data = {
        'res': res,
        'pipes_index': [str(i) for i in range(1, n_ESNs + 1)],
        'n_ESNs': n_ESNs,
        
        'training_timeseries_path': os.path.join(res.local_research_path, f'training_timeseries_re_{re}_new_pert.pickle'),
        'test_timeseries_path': os.path.join(res.local_research_path, f'test_timeseries_re_{re}_pert.pickle'),
        'synchronization_len': 10,
        'test_chunk_timeseries_len': 300,
        'spectral_radius_values': [0.5],
        'sparsity_values': [0.1, 0.5, 0.9],
        'reservoir_dimension': 1500,
        'optimal_esn_filenames': [f'{esn_name}_{i}' for i in range(1, n_ESNs + 1)],
        'optimal_esn_filename': f'{esn_name}',
        'finished': f'__finished__',
        'started': f'__started__',
#        'finished': [f'__finished_{i}__' for i in range(1, n_ESNs + 1)],
#        'started': [f'__started_{i}__' for i in range(1, n_ESNs + 1)],


        'dt_as_defined_by_esn': 1,
        'n_steps': n_steps,
        'final_time': n_steps,
        're': re,
        'l_x': l_x,
        'l_z': l_z,
        'initial_conditions': ics,
        'input_filename': 'inputs.json',
        'input_filenames': [ f'inputs_{i}.json' for i in range(1, n_ESNs + 1)],
        'output_filenames': str(1),
     #   'output_filenames': [str(i) for i in range(1, n_ESNs + 1)],
        'description': f'Predictions of trained ESN ensemble with one initial condition for all ESNs. '
                       f'Noise is disabled while predicting'
    }

#    graph = RemotePythonTimeIntegrationGraph(res, local_comm, ssh_comm,
#                                             EsnIntegrator(input_filename_key='input_filename', nohup=True),
#                                             input_filename=data['input_filename'],
#                                             task_prefix='ESNEnsemble')
    

    graph = ESNTrainAndIntergateGraph(res, local_comm,
                                           input_filename=data['input_filename'],
                                            task_prefix=task_prefix)
 #   print("graph")
    okay = graph.run(data)
    if not okay:
        print(data['__EXCEPTION__'])
 #   print(data)