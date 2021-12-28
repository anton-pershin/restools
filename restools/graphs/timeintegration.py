import os
import sys
sys.path.append(os.getcwd())
import random
from functools import partial
from typing import Type

import numpy as np

from restools.standardised_programs import StandardisedIntegrator, StandardisedProgramEdge, RandomfieldChannelflowV1, \
    AddfieldsChannelflowV1
from restools.timeintegration_builders import CacheAllAccess3DBuilder
from restools.relaminarisation import is_relaminarised
from restools.helpers import unlist_if_necessary
from comsdk.research import CreateTaskGraph, CreateTaskEdge
from comsdk.edge import Edge, ExecutableProgramEdge, QsubScriptEdge, UploadOnRemoteEdge, DownloadFromRemoteEdge, \
    job_unfinished_predicate, job_finished_predicate, dummy_edge, dummy_predicate, dummy_morphism, make_dump, \
    InOutMapping, connect_branches
from comsdk.graph import Graph, State, Func
import comsdk.comaux as comaux
from thequickmath.aux import is_sequence


class RemoteIntegrationGraph(Graph):
    def __init__(self, res, local_comm, remote_comm, integrator_prog: StandardisedIntegrator, spanning_key=None,
                 task_prefix=''):
        def task_name_maker(d):
            task_name = task_prefix
            if self.spanning_key is not None:
                if self.spanning_key != 'R':
                    task_name += '_R_{}'.format(d['R'])
                if self.spanning_key == 'initial_condition':
                    task_name += '_ics_{}'.format(len(d['initial_condition']))
                else:
                    task_name += '_{}_{}_{}_{}'.format(self.spanning_key, d[self.spanning_key][0],
                                                       d[self.spanning_key][-1],
                                                       d[self.spanning_key][1] - d[self.spanning_key][0])
            else:
                task_name += '_R_{}'.format(d['R'])
            if 'A' in d:
                task_name += '_A_{}_omega_{}'.format(d['A'], d['omega'])
            return task_name

        self.spanning_key = spanning_key
        if self.spanning_key is None:
            array_keys_mapping = None
        else:
            array_keys_mapping = {k: k for k in [self.spanning_key, 'T0', 'o', 'qsub_script_name', 'qsub_script',
                                                 'job_ID', 'i', 'data_subdir']}
            if self.spanning_key == 'R':
                array_keys_mapping['dt'] = 'dt'
            if self.spanning_key != 'initial_condition':
                array_keys_mapping['initial_condition'] = 'initial_condition'

        def update_integration_params(d):
            d['o'] = d['data_subdir']

        all_ti_finished = State('ALL_INTEGRATION_FINISHED')
        task_start, task_end = CreateTaskGraph.create_branch(res, task_name_maker=task_name_maker, remote=True)
        task_end.connect_to(all_ti_finished, edge=dummy_edge)

        ti_start, ti_end = RemoteIntegrationGraph.create_branch(local_comm, remote_comm, integrator_prog)
        ti_block_start = State('INTEGRATION_BLOCK_STARTED', array_keys_mapping=array_keys_mapping)
        ti_block_start.connect_to(ti_start, edge=Edge(dummy_predicate, Func(func=update_integration_params)))
        task_end.replace_with_graph(Graph(ti_block_start, ti_end))
        task_end.connect_to(all_ti_finished, edge=dummy_edge)
        super().__init__(task_start, all_ti_finished)

    @staticmethod
    def create_branch(local_comm, remote_comm, integrator_prog: StandardisedIntegrator,
                      relative_keys=(), keys_mapping={}, array_keys_mapping=None,
                      init_field_at_remote_key=None):
        io_mapping = InOutMapping(relative_keys=relative_keys, keys_mapping=keys_mapping)
        make_up_qsub_script_edge = QsubScriptEdge(integrator_prog.name, local_comm, remote_comm,
                                                  io_mapping=io_mapping,
                                                  keyword_names=integrator_prog.keyword_names,
                                                  trailing_args_keys=integrator_prog.trailing_args_keys)
        upload_a_edge = UploadOnRemoteEdge(remote_comm,
                                           local_paths_keys=integrator_prog.trailing_args_keys,
                                           already_remote_path_key=init_field_at_remote_key)
        upload_qsub_script_edge = UploadOnRemoteEdge(remote_comm,
                                                     io_mapping=io_mapping,
                                                     local_paths_keys=('qsub_script',))
        qsub_edge = ExecutableProgramEdge('qsub', remote_comm,
                                          io_mapping=io_mapping,
                                          trailing_args_keys=('qsub_script',),
                                          output_dict={'job_finished': False},
                                          stdout_processor=remote_comm.host.set_job_id,
                                          remote=True)
        qstat_edge = ExecutableProgramEdge('qstat', remote_comm,
                                           predicate=job_unfinished_predicate,
                                           io_mapping=io_mapping,
                                           keyword_names=('u',),
                                           remote=True,
                                           stdout_processor=remote_comm.host.check_task_finished)
        download_edge = DownloadFromRemoteEdge(remote_comm,
                                               predicate=dummy_predicate,
                                               io_mapping=io_mapping,
                                               remote_paths_keys=('o',),
                                               update_paths=False)
        s_ready = State('READY_FOR_TIME_INTEGRATION', array_keys_mapping=array_keys_mapping)
        s_uploaded_input_files = State('UPLOADED_INPUT_FILES')
        s_made_up_qsub_script = State('MADE_UP_QSUB_SCRIPT')
        s_uploaded_qsub_script = State('UPLOADED_QSUB_SCRIPT')
        s_sent_job = State('SENT_JOB')
        s_postprocessed_integration = State('POSTPROCESSED_INTEGRATION')
        s_downloaded_output_files = State('DOWNLOADED_OUTPUT_FILES')
        s_ready.connect_to(s_uploaded_input_files, edge=upload_a_edge)
        s_uploaded_input_files.connect_to(s_made_up_qsub_script, edge=make_up_qsub_script_edge)
        s_made_up_qsub_script.connect_to(s_uploaded_qsub_script, edge=upload_qsub_script_edge)
        s_uploaded_qsub_script.connect_to(s_sent_job, edge=qsub_edge)
        s_sent_job.connect_to(s_sent_job, edge=qstat_edge)
        s_sent_job.connect_to(s_postprocessed_integration,
                              edge=integrator_prog.postprocessor_edge(remote_comm, predicate=job_finished_predicate,
                                                                      io_mapping=io_mapping))
        s_postprocessed_integration.connect_to(s_downloaded_output_files, edge=download_edge)
        return s_ready, s_downloaded_output_files

    def initialize_data_for_start(self, data):
        unlist = False
        if self.spanning_key is None:
            unlist = True
            key_id = 'ic'
            spanning_values = [0]
        elif self.spanning_key == 'initial_condition':
            spanning_values = range(len(data['initial_condition']))
            key_id = 'ic'
        else:
            spanning_values = data[self.spanning_key]
            data['initial_condition'] = [data['initial_condition'] for _ in spanning_values]
            key_id = self.spanning_key
        if self.spanning_key == 'R':
            data['R'] = np.array(data['R'])
            data['dt'] = 1./data['R']
        data['T0'] = unlist_if_necessary([0 for _ in spanning_values], unlist)
        data['o'] = unlist_if_necessary([None for _ in spanning_values], unlist)
        data['data_subdir'] = unlist_if_necessary(['data-{}'.format(val) for val in spanning_values], unlist)
        data['qsub_script_name'] = unlist_if_necessary(['ti_{}_{}.sh'.format(key_id, val) for val in spanning_values],
                                                       unlist)
        data['qsub_script'] = unlist_if_necessary([None for _ in spanning_values], unlist)
        data['job_ID'] = unlist_if_necessary([None for _ in spanning_values], unlist)
        data['i'] = unlist_if_necessary([0 for _ in spanning_values], unlist)
        if 'job_finished' in data:
            del data['job_finished']


class LocalIntegrationGraph(Graph):
    def __init__(self, res, local_comm, integrator_prog: StandardisedIntegrator, task_prefix=''):
        def task_name_maker(d):
            if 'A' in d:
                return '{}_Re_{}_A_{}_omega_{}'.format(task_prefix, d['R'], d['A'], d['omega'])
            else:
                return '{}_Re_{}_A_0'.format(task_prefix, d['R'])

        def is_finished(d): return d['i'] == len(d['initial_condition'])

        def update_integration_params(d):
            d['initial_condition_path'] = d['initial_condition'][d['i']]
            d['o'] = os.path.join(d['__WORKING_DIR__'], 'data-{}'.format(d['i']))
            if 'A' in d:
                d['phi'] = random.uniform(0., 1.)
            d['i'] += 1

        all_ti_finished = State('ALL_INTEGRATION_FINISHED')
        task_start, task_end = CreateTaskGraph.create_branch(res, task_name_maker=task_name_maker)
        ti_start, ti_end = LocalIntegrationGraph.create_branch(local_comm, integrator_prog)
        task_end.connect_to(ti_start, edge=Edge(dummy_predicate, Func(func=update_integration_params)))
        ti_end.connect_to(ti_start, edge=Edge(Func(func=lambda d: not is_finished(d)),
                                              Func(func=update_integration_params)))
        ti_end.connect_to(all_ti_finished, edge=Edge(Func(func=is_finished), dummy_morphism))
        super().__init__(task_start, all_ti_finished)

    @staticmethod
    def create_branch(local_comm, integrator_prog: StandardisedIntegrator, relative_keys=(), keys_mapping={},
                      array_keys_mapping=None):
        s_init = State('READY_FOR_LOCAL_INTEGRATION', array_keys_mapping=array_keys_mapping)
        s_integrated = State('INTEGRATION_FINISHED')
        s_postprocessed = State('INTEGRATION_POSTPROCESSED')
        io_mapping = InOutMapping(relative_keys=relative_keys, keys_mapping=keys_mapping)
        ti_edge = ExecutableProgramEdge(integrator_prog.name, local_comm,
                                        io_mapping=io_mapping,
                                        keyword_names=integrator_prog.keyword_names,
                                        trailing_args_keys=integrator_prog.trailing_args_keys)
        s_init.connect_to(s_integrated, edge=ti_edge)
        s_integrated.connect_to(s_postprocessed,
                                edge=integrator_prog.postprocessor_edge(local_comm, io_mapping=io_mapping))
        return s_init, s_integrated


class MultipliedRandomisedIntegrationGraph(Graph):
    def __init__(self, res, local_comm, ssh_comm, integrator_prog_type: Type[StandardisedIntegrator], array_keys_mapping,
                 random_field_postprocess=None):
        s_start, s_end = MultipliedRandomisedIntegrationGraph.create_branch(res, local_comm, ssh_comm,
                                                                            integrator_prog_type,
                                                                            array_keys_mapping,
                                                                            random_field_postprocess=random_field_postprocess)
        super().__init__(s_start, s_end)

    @staticmethod
    def create_branch(res, local_comm, ssh_comm, integrator_prog_type: Type[StandardisedIntegrator], array_keys_mapping,
                      random_field_postprocess=None):
        task_start, task_end = CreateTaskGraph.create_branch(res,
                                              task_name_maker=lambda d: 'MRTI_Re_{}'.format(d['timeintegration']['R']),
                                              remote=True)
        gen_rp_start, gen_rp_end = RandomFieldGraph.create_branch(local_comm, 'random_perturbation_path',
                                                                  relative_keys=(('random_perturbation',),),
                                                                  random_field_postprocess=random_field_postprocess)
        addfields_start, addfields_end = AddFieldsGraph.create_branch(local_comm, 'addfields_params',
                                                                      'initial_condition_path',
                                                                      relative_keys=(('initial_conditions',),),
                                                                      array_keys_mapping=array_keys_mapping)
        ti_begin, ti_end = RemoteIntegrationGraph.create_branch(local_comm, ssh_comm,
                                                     integrator_prog_type('initial_condition_path'),
                                                     relative_keys=(('initial_conditions',), ('timeintegration',)))

        connect_branches(((task_start, task_end),
                          (gen_rp_start, gen_rp_end),
                          (addfields_start, addfields_end)))
        ti_subgraph = Graph(ti_begin, ti_end)
        addfields_end.replace_with_graph(ti_subgraph)
        return task_start, ti_end


class MultipliedInitialisedIntegrationGraph(Graph):
    def __init__(self, res, local_comm, ssh_comm, integrator_prog_type: Type[StandardisedIntegrator], array_keys_mapping):
        s_start, s_end = MultipliedInitialisedIntegrationGraph.create_branch(res, local_comm, ssh_comm,
                                                                             integrator_prog_type,
                                                                             array_keys_mapping)
        super().__init__(s_start, s_end)

    @staticmethod
    def create_branch(res, local_comm, ssh_comm, integrator_prog_type: Type[StandardisedIntegrator], array_keys_mapping):
        task_start, task_end = CreateTaskGraph.create_branch(res,
                              task_name_maker=lambda d: 'MITI_R_{}_A_{}_omega_{}'.format(d['timeintegration']['R'],
                                                                                         d['timeintegration']['A'],
                                                                                         d['timeintegration']['omega']),
                                                             remote=True)
        addfields_start, addfields_end = AddFieldsGraph.create_branch(local_comm, 'addfields_params',
                                                                      'initial_condition_path',
                                                                      relative_keys=(('initial_conditions',),),
                                                                      array_keys_mapping=array_keys_mapping)
        ti_begin, ti_end = RemoteIntegrationGraph.create_branch(local_comm, ssh_comm,
                                                     integrator_prog_type('initial_condition_path'),
                                                     relative_keys=(('initial_conditions',), ('timeintegration',)))
        connect_branches(((task_start, task_end),
                          (addfields_start, addfields_end)))
        ti_subgraph = Graph(ti_begin, ti_end)
        addfields_end.replace_with_graph(ti_subgraph)
        return task_start, ti_end


class ContinuingIntegrationGraph(Graph):
    def __init__(self, res, local_comm, ssh_comm, integrator_prog_type: Type[StandardisedIntegrator],
                 spanning_key='R', task_prefix='', initial_condition_preprocessor=None):
        def task_name_maker(d):
            if (not is_sequence(d[spanning_key])) or (len(d[spanning_key]) == 1):
                task_name = '{}_ContinuingTimeIntegration_{}_{}_'.format(task_prefix, spanning_key, d[spanning_key][0])
            else:
                task_name = '{}_ContinuingTimeIntegration_{}_{}_{}_{}_'.format(task_prefix, spanning_key, d[spanning_key][0], d[spanning_key][-1], d[spanning_key][1] - d[spanning_key][0])
            selected_params = ['R', 'A', 'omega', 'phi']
            if spanning_key in selected_params:
                selected_params.remove(spanning_key)
            task_name += '_'.join(['{}_{}'.format(key, d[key]) for key in selected_params if key in d.keys()])
            if not 'A' in d.keys():
                task_name += 'A_0'
            return task_name

        def is_beginning(d):
            return d['i'] == 0 or d['i'][0] == 0

        s_ready = State('READY_FOR_CONTINUING_INTEGRATION_GRAPH')
        s_task_loaded = State('TASK_LOADED')

        s_cont_ti_subgraph_begin, s_cont_ti_subgraph_end = \
            ContinuingIntegrationGraph.create_branch(local_comm, ssh_comm, integrator_prog_type,
                                                     spanning_key=spanning_key,
                                                     initial_condition_preprocessor=initial_condition_preprocessor)
        create_task_edge = CreateTaskEdge(res, task_name_maker=task_name_maker, predicate=Func(func=is_beginning),
                                          remote=True)
        create_task_edge.postprocess = make_dump(r'data_wo_res_init.obj', omit='res')
        s_ready.connect_to(s_task_loaded, edge=create_task_edge)
        s_ready.connect_to(s_task_loaded, edge=Edge(Func(func=lambda d: not is_beginning(d)), dummy_morphism))
        s_task_loaded.connect_to(s_cont_ti_subgraph_begin, edge=dummy_edge)
        super().__init__(s_ready, s_cont_ti_subgraph_end)

    @staticmethod
    def create_branch(local_comm, ssh_comm, integrator_prog_type: Type[StandardisedIntegrator], spanning_key='R',
                      initial_condition_preprocessor=None):
        s_ready = State('READY_FOR_CONTINUING_INTEGRATION_SUBGRAPH')
        s_cont_ti_started = State('CONTINUING_INTEGRATION_STARTED')
        s_cont_ti_finished = State('CONTINUING_INTEGRATION_FINISHED')

        array_keys_mapping = {k: k for k in [spanning_key, 'T0', 'o', 'qsub_script_name', 'qsub_script', 'job_ID', 'i',
                                             'data_subdir', 'initial_condition_path']}
        if spanning_key == 'R':
            array_keys_mapping['dt'] = 'dt'
        s_ti_started_subgraph_init = State('INTEGRATION_STARTED_SUBGRAPH_INIT', array_keys_mapping=array_keys_mapping)
        s_ti_started_subgraph_next = State('INTEGRATION_STARTED_SUBGRAPH_NEXT')
        s_checked_ti_not_finished = State('CHECKED_INTEGRATION_NOT_FINISHED')
        s_ti_finished_subgraph = State('INTEGRATION_FINISHED_SUBGRAPH')

        def is_finished(d):
            data_path = os.path.join(d['__WORKING_DIR__'],
                                     '{}-{}'.format(d['data_subdir'], d['i']))

            if not os.path.exists(data_path):
                return False
            ti = CacheAllAccess3DBuilder(integrator_prog_type.ti_class).get_timeintegration(data_path)
            if ti.T[-1] >= d['T1']:
                return True
            return is_relaminarised(ti.max_ke)

        def get_latest_velocity_field_filename(data_path):
            field_file = None
            field_time = 0
            filenames_and_params = \
                comaux.find_all_files_by_standardised_naming(integrator_prog_type.ti_class.solution_standardised_filename,
                                                             data_path)
            for filename, params in filenames_and_params:
                if params['t'] > field_time:
                    field_time = params['t']
                    field_file = filename
            return field_file, field_time

        def update_integration_params(d):
            if d['i'] != 0:
                all_data_path = d['__WORKING_DIR__']
                current_data_dir = '{}-{}'.format(d['data_subdir'], d['i'])
                current_data_path = os.path.join(all_data_path, current_data_dir)
                print('\t {}'.format(current_data_path))
                latest_u_field_file, u_field_time = get_latest_velocity_field_filename(current_data_path)
                d['initial_condition_path'] = os.path.join(d['__REMOTE_WORKING_DIR__'], current_data_dir,
                                                           latest_u_field_file)
                d['init_field_at_remote'] = True
                d['T0'] = u_field_time
            d['i'] += 1
            d['o'] = '{}-{}'.format(d['data_subdir'], d['i'])

        s_ti_begin, s_ti_end = RemoteIntegrationGraph.create_branch(local_comm, ssh_comm,
                                                         integrator_prog_type('initial_condition_path'),
                                                         init_field_at_remote_key='init_field_at_remote')
        ic_preprocessor_edge = Edge(dummy_predicate, initial_condition_preprocessor) \
            if initial_condition_preprocessor is not None else dummy_edge
        s_ready.connect_to(s_cont_ti_started, edge=ic_preprocessor_edge)
        s_cont_ti_started.connect_to(s_cont_ti_finished, edge=dummy_edge)
        s_ti_started_subgraph_init.connect_to(s_ti_started_subgraph_next, edge=dummy_edge)
        update_integration_params_edge = Edge(Func(func=lambda d: not is_finished(d)),
                                              Func(func=update_integration_params))
        s_ti_started_subgraph_next.connect_to(s_checked_ti_not_finished, edge=update_integration_params_edge)
        concat_integration = partial(integrator_prog_type.concatenate_integration_piece,
                                     input_datas_key=None, output_data_key='data_subdir')
        s_ti_started_subgraph_next.connect_to(s_ti_finished_subgraph,
                                              edge=Edge(Func(func=is_finished), Func(func=concat_integration)))
        s_checked_ti_not_finished.connect_to(s_ti_begin, edge=dummy_edge)
        after_integration_edge = dummy_edge
        s_ti_end.connect_to(s_ti_started_subgraph_next, edge=after_integration_edge)
        s_cont_ti_started.replace_with_graph(Graph(s_ti_started_subgraph_init, s_ti_finished_subgraph))
        return s_ready, s_cont_ti_finished

    @staticmethod
    def initialize_data_for_start(data, spanning_key, initial_condition):
        spanning_values = data[spanning_key]
        data['T0'] = [0 for _ in spanning_values]
        data['o'] = [None for _ in spanning_values]
        data['data_subdir'] = ['data-{}'.format(val) for val in spanning_values]
        data['qsub_script_name'] = ['ti_{}_{}.sh'.format(spanning_key, val) for val in spanning_values]
        data['qsub_script'] = [None for _ in spanning_values]
        data['job_ID'] = [None for _ in spanning_values]
        data['i'] = [0 for _ in spanning_values]
        data['initial_condition_path'] = [initial_condition for _ in spanning_values]
        if 'job_finished' in data:
            del data['job_finished']

    @staticmethod
    def initialize_data_for_restart(data, res, Re, i):
        data['res'] = res
        data['R'] = Re
        data['dt'] = 1./Re
        data['T0'] = [0 for _ in Re]
        data['o'] = [None for _ in Re]
        data['data_subdir'] = ['data-{}'.format(Re_) for Re_ in Re]
        data['qsub_script_name'] = ['ti_R_{}.sh'.format(Re_) for Re_ in Re]
        data['qsub_script'] = [None for _ in Re]
        data['job_ID'] = [None for _ in Re]
        data['i'] = i
        data['initial_condition_path'] = [data['initial_condition_path'][0] for _ in Re]
        data['init_field_at_remote'] = False


class AddFieldsGraph(Graph):
    def __init__(self, comm, params_key, output_file_key, relative_keys=(), keys_mapping={}, array_keys_mapping=None):
        s_init, s_term = self.create_branch(comm, params_key, output_file_key, relative_keys=relative_keys,
                                            keys_mapping=keys_mapping, array_keys_mapping=array_keys_mapping)
        super().__init__(s_init, s_term)

    @staticmethod
    def create_branch(comm, params_key, output_file_key, relative_keys=(), keys_mapping={}, array_keys_mapping=None):
        s_init = State('READY_FOR_ADD_FIELDS', array_keys_mapping=array_keys_mapping)
        s_term = State('ADD_FIELDS_FINISHED')
        addfields_edge = StandardisedProgramEdge(AddfieldsChannelflowV1(params_key=params_key,
                                                                        output_filename_key=output_file_key),
                                                 comm, relative_keys=relative_keys, keys_mapping=keys_mapping)
        s_init.connect_to(s_term, edge=addfields_edge)
        return s_init, s_term


class RandomFieldGraph(Graph):
    def __init__(self, comm, output_filename_key, relative_keys=(), keys_mapping={}, array_keys_mapping=None,
                 random_field_postprocess=None):
        s_init, s_term = self.create_branch(comm, output_filename_key, relative_keys=relative_keys,
                                            keys_mapping=keys_mapping, array_keys_mapping=array_keys_mapping,
                                            random_field_postprocess=random_field_postprocess)
        super().__init__(s_init, s_term)

    @staticmethod
    def create_branch(comm, output_filename_key, relative_keys=(), keys_mapping={}, array_keys_mapping=None,
                      random_field_postprocess=None):
        s_init = State('READY_FOR_RANDOM_FIELD_GENERATION', array_keys_mapping=array_keys_mapping)
        s_term = State('RANDOM_FIELD_GENERATED')
        random_field_edge = StandardisedProgramEdge(RandomfieldChannelflowV1(output_filename_key), comm,
                                                    relative_keys=relative_keys, keys_mapping=keys_mapping)
        s_init.connect_to(s_term, edge=random_field_edge)
        if random_field_postprocess is not None:
            random_field_edge.postprocess = random_field_postprocess
        return s_init, s_term
