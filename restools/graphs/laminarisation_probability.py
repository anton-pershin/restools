### TODO: need to rewrite and adapt to the style I have used in launchers.timeintegration

def create_probabilistic_study_graph(res, local_comm, ssh_comm, code_version='openmp'):
    s_ready = State('READY_FOR_CONTINUING_INTEGRATION')
    s_task_loaded = State('TASK_LOADED')
    s_integration_started = State('INTEGRATION_STARTED')
    s_all_finished = State('ALL_FINISHED')

    main_array_keys_mapping = {k: k for k in ['o', 'qsub_script_name', 'qsub_script', 'job_ID', 'i', 'initial_condition_path', 'rand_pert_path', 'energy_level', 'sd', 'addfields_params']}
    main_array_keys_mapping['sd'] = ('random_perturbation', 'sd')
    s_random_integration_started_subgraph_init = State('RANDOM_INTEGRATION_STARTED_SUBGRAPH_INIT',
                                                       array_keys_mapping=main_array_keys_mapping)
    s_random_integration_started_subgraph_next = State('RANDOM_INTEGRATION_STARTED_SUBGRAPH_NEXT')
    s_random_initial_condition_generated = State('RANDOM_INTIAL_CONDITION_GENERATED')
    s_checked_integration_not_finished = State('CHECKED_INTEGRATION_NOT_FINISHED')
    s_all_initial_conditions_time_integrated = State('ALL_INITIAL_CONDITIONS_TIME_INTEGRATED')

    is_beginning = lambda d: d['i'] == 0 or d['i'][0] == 0

    def is_finished(d):
        return d['i'] >= d['total_points_at_each_energy_level']

    def update_integration_params(d):
        d['i'] += 1
        d['sd'] = random.getrandbits(32)
        d['rand_pert_path'] = os.path.join(d['__WORKING_DIR__'],
                                           d['rand_pert_format'].format(d['energy_level'], d['i']))
        if d['energy_type'] == 'relative':
            # generate random A and B to guarantee that the initial condition is on the isoenergy ellipse
            # we also ensure that the resulting perturbation is small enough compared to the laminar flow (in percent by epsilon_pert)
            # (see meething notes from 2019/02/08)
            while True:
                t = random.uniform(0., 2.*np.pi)
                A = np.sqrt(d['energy_level']) * np.cos(t)
                B = np.sqrt(d['energy_level'] / d['U_lam_energy']) * np.sin(t) - 1.
                if A**2 + (B**2 - d['epsilon_pert']) * d['U_lam_energy'] <= 0:
                    break
        elif d['energy_type'] == 'departure':
            # here we ensure that B is distributed uniformly
            t = random.uniform(0., 2.*np.pi)
            B_max = np.sqrt(d['energy_level'] / d['U_lam_energy'])
            if 'B_sign' in d:
                B = random.uniform(0, B_max) if d['B_sign'] > 0 else random.uniform(-B_max, 0)
            else:
                B = random.uniform(-B_max, B_max)
            A = np.sqrt(d['energy_level'] - B**2 * d['U_lam_energy']) # the sign of A can be ignored
            #A = np.sqrt(d['energy_level']) * np.cos(t)
            #B = np.sqrt(d['energy_level'] / d['U_lam_energy']) * np.sin(t)
        else:
            raise Exception('Unknown energy type: {}'.format(d['energy_type']))
        if 'A' in d:
            d['phi'] = random.uniform(0., 1.)

        d['initial_condition_path'] = os.path.join(d['__WORKING_DIR__'],
                                                   d['initial_condition_format'].format(A, B, d['energy_level'], d['i']))
        d['addfields_params'] = [A, d['rand_pert_path'], B, d['laminar_flow_field_path']]
        d['o'] = d['o_format'].format(d['energy_level'], d['i'])

    def task_name_maker(d):
        prefix = 'ProbabiliticStudy'
        Re_dependence = 'Re_{}'.format(d['R'])
        energies_and_RP_number = 'elevels_{}_pointsatlevel_{}'.format(len(d['energy_level']), d['total_points_at_each_energy_level'])
        if 'B_sign' in d:
            B_sign = 'B_plus' if d['B_sign'] > 0 else 'B_minus'
        else:
            B_sign = None
        control_dependence = 'A_{}_omega_{}'.format(d['A'], d['omega']) if 'A' in d else 'A_0'
        return '_'.join((piece for piece in (prefix, Re_dependence, energies_and_RP_number, B_sign, control_dependence) if piece is not None))

    create_task_edge = comsdk.research.CreateTaskEdge(res, task_name_maker=task_name_maker, predicate=is_beginning)
    create_task_edge.postprocess = make_dump(r'data_wo_res_init.obj', omit='res')
    update_integration_params_edge = Edge(lambda d: not is_finished(d), update_integration_params)
    generate_random_perturbation_edge = create_random_perturbation_edge(local_comm, 'rand_pert_path',
                                                                        relative_keys=(('random_perturbation',),))
    add_laminar_field_edge = create_add_fields_edge(local_comm,
                                                    params_key='addfields_params',
                                                    output_file_key='initial_condition_path')
    #update_integration_params_edge.preprocess = make_dump(r'data_wo_res_{}.obj', format_keys=['i',], omit='res')

#    update_integration_params_edge.postprocess = make_cd('integration_subdir')

    s_ti_begin, s_ti_end = integration_branch_factory(local_comm, ssh_comm,
                                                      init_field_path_key='initial_condition_path',
                                                      code_version=code_version)
    s_ready.connect_to(s_task_loaded, edge=create_task_edge)
    #s_ready.connect_to(s_task_loaded, edge=Edge(lambda d: not is_beginning(d), update_integration_params_on_restart))
    s_ready.connect_to(s_task_loaded, edge=Edge(lambda d: not is_beginning(d), dummy_edge))

    s_task_loaded.connect_to(s_integration_started, edge=DummyEdge())
    s_integration_started.connect_to(s_all_finished, edge=DummyEdge())

    s_random_integration_started_subgraph_init.connect_to(s_random_integration_started_subgraph_next, edge=DummyEdge())
    s_random_integration_started_subgraph_next.connect_to(s_checked_integration_not_finished, edge=update_integration_params_edge)
    s_random_integration_started_subgraph_next.connect_to(s_all_initial_conditions_time_integrated,
                                                          edge=Edge(is_finished, dummy_edge))
    s_checked_integration_not_finished.connect_to(s_random_initial_condition_generated, edge=generate_random_perturbation_edge)
    s_random_initial_condition_generated.connect_to(s_ti_begin, edge=add_laminar_field_edge)
    s_ti_end.connect_to(s_random_integration_started_subgraph_next, edge=DummyEdge())

    s_integration_started.replace_with_graph(Graph(s_random_integration_started_subgraph_init, s_all_initial_conditions_time_integrated))

    return Graph(s_ready, s_all_finished)
