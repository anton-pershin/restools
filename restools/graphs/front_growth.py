### TODO: need to rewrite and adapt to the style I have used in launchers.timeintegration

from chflow.ke_tools import average_turbulent_fraction_growth_rate

def create_onset_search_monotonic_front_growth_rate_graph(res, local_comm, ssh_comm, task_prefix, code_version='openmp'):
    s_ready_for_iteration = State('READY_FOR_ONSET_SEARCH_ITERATION')
    s_task_loaded = State('TASK_LOADED')
    s_ti_init, s_ti_term = create_continuing_integration_branch(res, local_comm, ssh_comm, spanning_key='R', code_version=code_version)
    s_onset_search_finished = State('ONSET_SEARCH_FINISHED')

    def onset_search_finished_predicate(d): return True if d['Re_turb'] - d['Re_lam'] <= 2 else False
    def reset_onset_search_iteration(d):
        print('\tResetting onset search iteration...')
        # find laminarising R, find transitioning R, then find the middle
#        Re_lam_path = os.path.join(d['__WORKING_DIR__'], 'data-{}'.format(d['Re_lam']))
#        Re_turb_path = os.path.join(d['__WORKING_DIR__'], 'data-{}'.format(d['Re_turb']))
        cur_Re = d['R'][0]
        Re_cur_path = os.path.join(d['__WORKING_DIR__'], 'data-{}'.format(cur_Re))
        c_cur = average_turbulent_fraction_growth_rate(Re_cur_path)
        if c_cur < 0:
            d['Re_lam'] = cur_Re
        else:
            d['Re_turb'] = cur_Re
        initialize_data_for_restart_of_continuing_integration_graph(d, d['res'], Re=np.array([int((d['Re_lam'] + d['Re_turb']) // 2)]),
                                                                                 i=[0])
        d['search_i'] += 1
        print('\tNew Re = {}'.format(d['R'][0]))
#        if c_left * c_right > 0:
#            raise Exception('Bisection is impossible: both averaged growth rates, c = {} at Re = {} and c = {} at Re = {}, are positive/negative'.format(c_left, d['R'][0], c_right, d['R'][1]))
#        if c_left > c_right:
#            raise Exception('Bisection is impossible: averaged growth rate at Re = {} (c = {}) is larger than at Re = {} (c = {})'.format(d['R'][0], c_left, d['R'][1], c_right))

    def task_name_maker(d):
        return '{}_R4_onset_search_A_{}_omega_{}'.format(task_prefix, d['A'], d['omega'])

    def current_R_timeintegrated(d):
        Re_cur_path = os.path.join(d['__WORKING_DIR__'], 'data-{}'.format(d['R'][0]))
        print('\tChecking current Re has already been time-integrated: {}'.format(os.path.exists(Re_cur_path)))
        return os.path.exists(Re_cur_path)

    is_beginning = lambda d: d['search_i'] == 0
    create_task_edge = comsdk.research.CreateTaskEdge(res, task_name_maker=task_name_maker, predicate=is_beginning)
    def search_iter_incr(d): d['search_i'] += 1
    create_task_edge.postprocess = make_composite_func(make_dump(r'data_wo_res_{}.obj', format_keys=['search_i'], omit='res'),
                                                       search_iter_incr)
    reset_onset_search_iteration_edge = Edge(lambda data: not onset_search_finished_predicate(data), reset_onset_search_iteration)
    reset_onset_search_iteration_edge.postprocess = make_dump(r'data_wo_res_{}.obj', format_keys=['search_i'], omit='res')
    reset_onset_search_iteration_after_task_loaded_edge = Edge(current_R_timeintegrated, reset_onset_search_iteration)
    reset_onset_search_iteration_after_task_loaded_edge.postprocess = make_dump(r'data_wo_res_{}.obj', format_keys=['search_i'], omit='res')

    s_ready_for_iteration.connect_to(s_task_loaded, edge=create_task_edge)
    s_ready_for_iteration.connect_to(s_task_loaded, edge=Edge(lambda d: not is_beginning(d), dummy_edge))
#    cur_R_timeintegrated_and_search_not_finished = make_composite_predicate(onset_search_finished_predicate, current_R_timeintegrated)
    s_task_loaded.connect_to(s_ti_init, edge=Edge(lambda d: not current_R_timeintegrated(d), dummy_edge))
    s_task_loaded.connect_to(s_ti_init, edge=reset_onset_search_iteration_after_task_loaded_edge)
    s_ti_term.connect_to(s_ti_init, edge=reset_onset_search_iteration_edge)
    s_ti_term.connect_to(s_onset_search_finished, edge=Edge(onset_search_finished_predicate, dummy_edge))
    return Graph(s_ready_for_iteration, s_onset_search_finished)

def create_onset_search_nonmonotonic_front_growth_rate_graph(res, local_comm, ssh_comm, initial_condition, task_prefix, code_version='openmp'):
    s_ready_for_iteration = State('READY_FOR_ONSET_SEARCH_ITERATION')
    s_task_loaded = State('TASK_LOADED')
    s_search_reset = State('SEARCH_RESET')
    s_ti_init, s_ti_term = create_continuing_integration_branch(res, local_comm, ssh_comm, spanning_key='R', code_version=code_version)
    s_onset_search_finished = State('ONSET_SEARCH_FINISHED')

    def onset_search_finished_predicate(d): return True if d['delta_Re'] < 2 else False
    def reset_onset_search_iteration(d):
        def update_ti_params(d):
            d['R'] = np.array(range(d['Re_lam'] + d['delta_Re'], d['Re_turb'], d['delta_Re']))
            d['dt'] = 1./d['R']
            initialize_data_for_continuing_integration_graph(d, 'R', initial_condition=initial_condition)
            d['search_i'] += 1
            print_pretty_dict(d)

        print('\tResetting onset search iteration...')

        # Check whether all time-integrations finished
        Res = range(d['Re_lam'] + d['delta_Re'], d['Re_turb'], d['delta_Re'])
        Res_paths = [os.path.join(d['__WORKING_DIR__'], 'data-{}'.format(Re)) for Re in Res]
        if d['search_i'] == 1: # the very beginning (we have not yet integrated)
            update_ti_params(d)
        elif False in [os.path.exists(Re_path) for Re_path in Res_paths]: # there are unfinished ti => need to continue ti at the same search iteration
            i_by_Re = {}
            Res_temp = list(Res)
            for file_or_dir in os.listdir(d['__WORKING_DIR__']):
                if file_or_dir.startswith('data-'):
                    dirname_parts = file_or_dir.split('-')
                    Re = int(dirname_parts[1])
                    if len(dirname_parts) == 3:
                        i = int(dirname_parts[2])
                        if not Re in i_by_Re:
                            i_by_Re[Re] = i
                            Res_temp.remove(Re)
                        else:
                            if i > i_by_Re[Re]:
                                i_by_Re[Re] = i
                    elif len(dirname_parts) == 2:
                        if Re in Res_temp:
                            Res_temp.remove(Re)
            for Re in Res_temp: # go for the remaining Re which were not ti-d at all
                i_by_Re[Re] = 0
            sorted_i_by_Re = sorted(list(i_by_Re.items()), key=itemgetter(0))
            Res_updated = [Re for Re, i in sorted_i_by_Re]
            i_updated = [i for Re, i in sorted_i_by_Re]
            for Re, i in zip(Res_updated, i_updated):
                print('\tFound unfinished TI: Re = {}, i = {}'.format(Re, i))
#            d['initial_condition_path'] = ['/media/tony/WD_My_Passport_2TB/Leeds/PhD/results/cont_32pi/saddle-nodes/S5_Re175.36015_D1.234223.h5']
            initialize_data_for_restart_of_continuing_integration_graph(d, d['res'], Re=np.array(Res_updated), i=i_updated)
        else: # current iteration finished (no unfinished ti) => go to the next iteration
            turb_traj_found = False
            for i in range(len(Res_paths)):
                c = average_turbulent_fraction_growth_rate(Res_paths[i])
                if c > 0:
                    d['Re_turb'] = Res[i]
                    if i != 0:
                        d['Re_lam'] = Res[i - 1]
                    turb_traj_found = True
                    break
            if not turb_traj_found: # d['Re_turb'] remains the closest turb trajectory. Need to pick max Re for laminar trajectory
                d['Re_lam'] = Res[-1]
            d['delta_Re'] = int(d['delta_Re'] / 2)
            if d['delta_Re'] == 0:
                raise Exception('Delta Re is zero. Terminate the onset search.')
            update_ti_params(d)
            print('\tUpdated estimations for the onset: Re_lam = {}, Re_turb = {}. \
Starting next search iteration with delta_Re = {} between Re = {} and Re = {}'.format(d['Re_lam'],
                                                                                      d['Re_turb'],
                                                                                      d['delta_Re'],
                                                                                      d['R'][0],
                                                                                      d['R'][-1]))

    def task_name_maker(d):
        return '{}_R4_onset_search_A_{}_omega_{}'.format(task_prefix, d['A'], d['omega'])

    is_beginning = lambda d: d['search_i'] == 0
    create_task_edge = comsdk.research.CreateTaskEdge(res, task_name_maker=task_name_maker, predicate=is_beginning)
    def search_iter_incr(d): d['search_i'] += 1
    create_task_edge.postprocess = make_composite_func(make_dump(r'data_wo_res_{}.obj', format_keys=['search_i'], omit='res'),
                                                       search_iter_incr)
    reset_onset_search_iteration_edge = Edge(lambda data: not onset_search_finished_predicate(data), reset_onset_search_iteration)
    reset_onset_search_iteration_edge.postprocess = make_dump(r'data_wo_res_{}.obj', format_keys=['search_i'], omit='res')
    reset_onset_search_iteration_after_task_loaded_edge = Edge(dummy_predicate, reset_onset_search_iteration)
    reset_onset_search_iteration_after_task_loaded_edge.postprocess = make_dump(r'data_wo_res_{}.obj', format_keys=['search_i'], omit='res')

    s_ready_for_iteration.connect_to(s_task_loaded, edge=create_task_edge)
    s_ready_for_iteration.connect_to(s_task_loaded, edge=Edge(lambda d: not is_beginning(d), dummy_edge))
#    cur_R_timeintegrated_and_search_not_finished = make_composite_predicate(onset_search_finished_predicate, current_R_timeintegrated)
    s_task_loaded.connect_to(s_search_reset, edge=reset_onset_search_iteration_after_task_loaded_edge)
    s_search_reset.connect_to(s_ti_init, edge=DummyEdge())
    s_ti_term.connect_to(s_search_reset, edge=reset_onset_search_iteration_edge)
    s_ti_term.connect_to(s_onset_search_finished, edge=Edge(onset_search_finished_predicate, dummy_edge))
    return Graph(s_ready_for_iteration, s_onset_search_finished)
