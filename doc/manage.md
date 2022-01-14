# Management script: restools/manage.py

The usage of script is as follows:
```bash
python restools/manage.py <command_group> <command> <command_details>
```
where `<command_group>` can be either `research`, `meeting`, `grabresults` or `cloud`. Details are below

## Manipulating research data: <command_group> == research

This group of commands allows us to create a research catalog:
```
python restools/manage.py research create
```
It will ask for research description (max 60 symbols) and short ID which will be used latter in other commands.

## Creating a meeting presentation: <command_group> == meeting

This group of commands allows us to create a meeting presentation in a directory set in `MEETINGS_PATH` in `config_research.json`:
```
python restools/manage.py meeting create
```
It will ask for year, month and day.

## Grabbing results from remote hosts: <command_group> == grabresults

This group of commands allows us to download research data (namely, tasks) from a remote host to local storage defined by the first entry in `LOCAL_HOST.research_roots` in `config_research.json`:
```
python restools/manage.py grabresults --res <RES_ID> --remote <REMOTE_HOST_ID> <task_numbers_separated_by_whitespaces>
```
where 
* `<RES_ID>` is the short research ID used as a key `RESEARCH` in `config_research.json`
* `<REMOTE_HOST_ID>` is the remote host ID used as a key in `REMOTE_HOSTS` in `config_research.json`
* `<task_numbers_separated_by_whitespaces>` is a list of task numbers associated with the given research which should be downloaded

## Manipulating research data on cloud: <command_group> == cloud

This group of commands allows us to upload or download research data to/from a cloud. Below `RES_ID` is the short research ID as a key `RESEARCH` in `config_research.json`

### Listing research tasks available on the cloud:
```
python restools/manage.py cloud list --res <RES_ID>
```
This command will output task archives stored in the cloud.

### Uploading tasks to the cloud:
```
python restools/manage.py cloud upload --res <RES_ID> --tasks <task_numbers_separated_by_whitespaces>
```
This command will compress tasks and upload archives to the cloud using aws commands.

### Downloading tasks from the cloud
```
python restools/manage.py cloud download --res <RES_ID> --tasks <task_numbers_separated_by_whitespaces>
```
This command will download task archives from the cloud and extract them to the local research catalog.
