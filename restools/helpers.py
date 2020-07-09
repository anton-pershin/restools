def set_job_id_arc(data, stdout_lines):
    print(stdout_lines)
    return {'job_ID': int(stdout_lines[0].split()[2])} # example: 'Your job 664989 ("fe_170.310.sh") has been submitted'


def set_job_id_cirrus(data, stdout_lines):
    print(stdout_lines)
    return {'job_ID': int(stdout_lines[0].split('.')[0])} # example: '1068318.indy2-login0'


def check_task_finished_arc(data, stdout_lines):
    '''
    Example:
    job-ID  prior   name       user         state submit/start at     queue                          slots ja-task-ID
    -----------------------------------------------------------------------------------------------------------------
    663565 0.00053 RT700-tran scegr        r     09/19/2018 23:51:22 24core-128G.q@dc2s2b1a.arc3.le    24
    663566 0.00053 RT800-tran scegr        r     09/19/2018 23:51:22 24core-128G.q@dc3s5b1a.arc3.le    24
    663567 0.00053 RT900-tran scegr        r     09/20/2018 00:00:22 24core-128G.q@dc4s2b1b.arc3.le    24
    663569 0.00053 RT1000-tra scegr        r     09/20/2018 00:05:07 24core-128G.q@dc1s1b3d.arc3.le    24
    '''
    job_finished = True
    for line in stdout_lines[2:]:
        items = line.split()
        if int(items[0]) == data['job_ID']:
            job_finished = False
    return {'job_finished': job_finished}


def check_task_finished_cirrus(data, stdout_lines):
    '''
    Example:

    indy2-login0:
                                                                Req'd  Req'd   Elap
    Job ID          Username Queue    Jobname    SessID NDS TSK Memory Time  S Time
    --------------- -------- -------- ---------- ------ --- --- ------ ----- - -----
    1068318.indy2-l apershin workq    ti_R_180.s    --    1   8    --  12:00 Q   --
    '''
    job_finished = True
    if stdout_lines != []:
        first_line = 5 if stdout_lines[0].strip() == '' else 4
        for line in stdout_lines[first_line:]:
            items = line.split()
            if int(items[0].split('.')[0]) == data['job_ID']:
                job_finished = False
    return {'job_finished': job_finished}


def unlist_if_necessary(seq, unlist=True):
    if unlist:
        return seq[0]
    else:
        return seq
