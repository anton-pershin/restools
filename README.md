# restools
A set of pre- and post-processing tools for computational research in transitional turbulence, echo state networks and approximate computing.

# Getting started

To start working with restools, create your own `config_research.json`:
```bash
cp pycomsdk/config_research.json.example config_research.json
```
First of all, one needs to fill the list of local roots as absolute paths where research directories to be located: `LOCAL_HOST.research_roots`. Then, one needs to fill `RESEARCH` field to inform about available research. Below is the list of currently used research IDs and their names. Research ID is used as a short ID everywhere in API. Research description is the same as the name of research directory which needs to be located in one of the local roots defined in `LOCAL_HOST.research_roots`.

> :warning: If you plan to use existing research IDs and download tasks from the cloud, don't forget to create research directories in one of the local roots with the names as defined below. 

| Research ID | Research description (name of research directory) |
|-------------|---------------------------------------------------|
| EQ | 2017-03-02-eq-continuation
| EQ_STAB | 2018-09-11-eq-stability
| EQ_SUB_HOPF | 2018-08-23-eq-subcritical-hopf-bifurcation
| TW | 2017-03-02-tw-continuation
| TW_STAB | 2018-10-04-tw-stability
| PO5 | 2017-10-04-po5
| EQ_LOW | 2018-04-03-eq-lower-branch-analysis
| EDGE_SMALL | 2018-06-20-edge-tracking-in-small-domain
| EDGE_IN_PHASE | 2018-10-19-edge-tracking-for-in-phase-oscillations
| WIDE_SIMS_IN_PHASE | 2018-11-13-simulations-for-in-phase-oscillations-in-wide-domain
| WIDE_SIMS_ANTIPHASE | 2019-04-20-simulations-for-antiphase-oscillations-in-wide-domain
| RAND_STUDY | 2019-02-21-randomised-study-in-small-domain
| RAND_STUDY_UNIFORM_B | 2019-04-19-randomised-study-with-uniform-pdf-for-B-in-small-domain
| RAND_STUDY_UNIFORM_B_OSC | 2019-05-06-randomised-study-for-in-phase-oscillations-and-uniform-pdf-for-B-in-small-domain
| P_LAM_ESTIMATION | 2019-11-13-laminarisation-probability-estimation
| P_LAM_RE | 2020-01-20-laminarisation-probability-dependence-on-Re
| TEST_EPFL_CF | 2020-03-26-test-simulations-for-EPFL-version-of-channelflow
| SOME_TEST | 2020-04-20-some-test-research-for-fun
| LARGE_DOMAIN | 2020-07-07-large-domain-simulations
| RC_MOEHLIS | 2021-04-30-predicting-transition-to-turbulence-using-esn
| ECRAD | 2021-05-18-ecrad-reduced-precision
| IFSMPI | 2021-07-08-ifs-reduced-precision-in-mpi-communications

If you plan to interact with the cloud (e.g., download existing research tasks, upload new ones etc.), you need to [install AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html). All the interaction should be done using a management script `restools/manage.py`: see [docs](/doc/manage.md) for details.

If you want to create new research, use a management script `restools/manage.py`: see [docs](/doc/manage.md) for details.

If you want to run some external programs locally as a part of a task launch, they need to be added to `LOCAL_HOST.custom_programs` (here, key is the directory where external programs are located and value is a list of executables/python scripts to be run). Sometimes, it is useful to execute a command-line-like expression as a part of task launch -- such aliases need to be added to `LOCAL_HOST.custom_commands` (here, key is the alias and value is a command-line expression).

If you want to run task remotely, remote host data need to be added to `REMOTE_HOSTS` (here, key is the host name, used as ID further, and value is the host data). Host data includes the following fields:
| Field | Description |
|-------|-------------|
| ssh_host | Host address |
| max_cores | Maximum number of cores you can use on this machine |
| username | SSH username |
| password | SSH password (can be null if pkey is provided) |
| pkey | SSH public key (can be omitted if password is provided) |
| execute_after_connection | Command-line expression which needs to be executed right after SSH connection |
| research_root | Research root on the remote machine (all the research directories will be there) |
| env_programs | A list of programs available from the command line due to environment settings |
| custom_programs | Same as custom_programs for local host |
| sge_template_name | Name of the template used to generate an SGE script for job scheduling (can be omitted if not needed). Template files must be located in the directory defined in `TEMPLATES_PATH` in the config file | 
| job_setter | An importable python function used to submit job on the remote machine with job scheduling (can be omitted if not needed). See `restools.helpers.set_job_id_cirrus` for an example |
| job_finished_checker | an importable python function used to check whether a job on the remote machine has finished (can be omitted if not needed). See `restools.helpers.check_task_finished_cirrus` for an example |

# Repository structure

* `/doc`: documentation
* `/jsons`: data storage for paper data 
* `/papers`: codes used to reproduce all the results and figures found in papers
* `/pycomsdk`: tools for creating research, tasks and graph-based algorithms
* `/reducedmodels`:  (TODO: obsolete)
* `/resources`: files used to set up the library
* `/restools`: a collection of research tools used for transition-to-turbulence, echo-state-network and approximate-computing research; the most general, useful and well tested tools from `/papers` migrate to this directory 
* `/templates`: any templates proceeded by mako
* `/tests`: tests (TODO)
* `/thequickmath`: simple numerical methods and algorithms

# Documentation

1. [Management script](/doc/manage.md): creating research, uploading and downloading tasks from the cloud and remote machines
2. [Research organizer pycomsdk](/doc/pycomsdk.md): general ideas behind the hierachy of research and tasks, interaction with remote hosts, graph-based scenario of running programs and python function and many other things
3. [Papers](/doc/papers.md): codes used to reproduce all the results and figures found in our papers, both published and yet unpublished
4. [Time-integration tools](doc/time_integr_tools.md): a collection of classes and functions helpful for dealing with time-integration data and time series (particularly used for transition to turbulence research)
