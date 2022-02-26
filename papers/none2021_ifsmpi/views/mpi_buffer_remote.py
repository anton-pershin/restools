import os
import sys
sys.path.append(os.getcwd())
import time

import numpy as np

from restools.standardised_programs import StandardisedProgram, StandardisedProgramEdge, SimplePythonProgram
from restools.timeintegration import TimeIntegrationLowDimensional
from papers.none2021_predicting_transition_using_reservoir_computing.extensions import RemotePythonTimeIntegrationGraph
from comsdk.communication import LocalCommunication, SshCommunication
from comsdk.research import Research, CreateTaskGraph
from comsdk.graph import Graph, State, Func
from comsdk.edge import Edge, InOutMapping, UploadOnRemoteEdge, DownloadFromRemoteEdge, dummy_predicate, make_dump


class RemotePlottingGraph(Graph):
    def __init__(self, local_comm, remote_comm, plotting_prog, input_filename,
                 output_filenames_key='output_filenames'):
        p_start, p_end = RemotePlottingGraph.create_branch(local_comm, remote_comm, plotting_prog,
                                                           output_filenames_key=output_filenames_key)
        
        def set_up_working_dir(d):
            d['__REMOTE_WORKING_DIR__'] = remote_comm.host.programs[plotting_prog.name]

        s_init = State('INIT')
        s_finished = State('FINISHED')
        dumping_edge = Edge(dummy_predicate, Func(func=set_up_working_dir))
        dumping_edge.postprocess = make_dump(input_filename, method='json')
        s_init.connect_to(p_start, edge=dumping_edge)
        p_end.connect_to(s_finished, edge=Edge(dummy_predicate, Func(func=lambda d: remote_comm.rm('/'.join([d['__REMOTE_WORKING_DIR__'], '__finished__'])))))
        super().__init__(s_init, s_finished)

    @staticmethod
    def create_branch(local_comm, remote_comm, plotting_prog: StandardisedProgram,
                      relative_keys=(), keys_mapping={}, array_keys_mapping=None,
                      init_field_at_remote_key=None, output_filenames_key='figure_filename'):
        def task_finished(d):
            time.sleep(2)
            return '__finished__' in remote_comm.listdir(d['__REMOTE_WORKING_DIR__'])

        task_finished_predicate = Func(func=task_finished)
        task_not_finished_predicate = Func(func=lambda d: not task_finished(d))
        io_mapping = InOutMapping(relative_keys=relative_keys, keys_mapping=keys_mapping)
        upload_edge = UploadOnRemoteEdge(remote_comm,
                                         local_paths_keys=plotting_prog.trailing_args_keys,
                                         already_remote_path_key=init_field_at_remote_key)
        integrator_edge = StandardisedProgramEdge(plotting_prog, remote_comm,
                                                 io_mapping=io_mapping, remote=True)
        download_edge = DownloadFromRemoteEdge(remote_comm,
                                               predicate=task_finished_predicate,
                                               io_mapping=io_mapping,
                                               remote_paths_keys=(output_filenames_key,),
                                               update_paths=False)
        s_ready = State('READY_FOR_PLOTTING', array_keys_mapping=array_keys_mapping)
        s_uploaded_input_files = State('UPLOADED_INPUT_FILES')
        s_integrated = State('PLOTTED')
        s_downloaded_output_files = State('DOWNLOADED_OUTPUT_FILES')
        s_ready.connect_to(s_uploaded_input_files, edge=upload_edge)
        s_uploaded_input_files.connect_to(s_integrated, edge=integrator_edge)
        s_integrated.connect_to(s_downloaded_output_files, edge=download_edge)
        s_integrated.connect_to(s_integrated, edge=Edge(task_not_finished_predicate, Func(func=lambda d: time.sleep(5))))
        return s_ready, s_downloaded_output_files


if __name__ == '__main__':
    local_comm = LocalCommunication.create_from_config()
    ssh_comm = SshCommunication.create_from_config('atmlxint1')
    #res = Research.open('RC_MOEHLIS', ssh_comm)
    data = {
    #    'res': res,
        'input_filename': 'inputs.json',
        'figure_filename': 'plot.eps',
        'description': f'Test res plotting.',
        'index': 42,
    }

    graph = RemotePlottingGraph(local_comm, ssh_comm,
                                SimplePythonProgram(program_name='plot_mpi_buffer.py',
                                                    input_filename_key='input_filename',
                                                    nohup=True,
                                                    pipes_index_key='index'),
                                input_filename=data['input_filename'],
                                output_filenames_key='figure_filename')
    okay = graph.run(data)
    if not okay:
        print(data['__EXCEPTION__'])
    print(data)
