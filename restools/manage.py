import os
import sys
sys.path.append(os.getcwd())
import argparse
from abc import ABC, abstractmethod
import subprocess

import restools
from comsdk.research import Research, split_task_dir
from comsdk.communication import LocalCommunication, SshCommunication
from restools.report import QuickPresentation


class CustomSubparser(ABC):
    def __init__(self, name):
        self.subparser = None
        self.name = name

    def add_subparser(self, subparsers):
        self.subparser = subparsers.add_parser(self.name)
        self.subparser.set_defaults(func=self.handler)

    def add_standard_arguments(self):
        self.subparser.add_argument('action', metavar='ACTION', nargs='?', choices=['create',], default='create',
                                    help='action applied to object (must be only create for now)')

    @abstractmethod
    def add_arguments(self):
        raise NotImplementedError('Must implement addition of arguments for the subparser')

    @abstractmethod
    def handler(self, args):
        raise NotImplementedError('Must implement handler for the subparser')


class ResearchSubparser(CustomSubparser):
    def __init__(self):
        super().__init__('research')

    def add_arguments(self):
        self.add_standard_arguments()

    def handler(self, args):
        if args.action == 'create':
            new_research_descr = input('Enter new research name: ')
            new_research_id = input('Enter short id: ')
            res = Research.create(new_research_id, new_research_descr)
        else:
            raise ValueError('Program action "{}" is unknown'.format(args.action))


class MeetingSubparser(CustomSubparser):
    def __init__(self):
        super().__init__('meeting')

    def add_arguments(self):
        self.add_standard_arguments()

    def handler(self, args):
        if args.action == 'create':
            year = input('Enter year: ')
            month = input('Enter month: ')
            day = input('Enter day: ')
            qp = QuickPresentation.from_config('Meeting notes', [int(year), int(month), int(day)])
            qp.print_out()
        else:
            raise ValueError('Program action "{}" is unknown'.format(args.action))


class GrabResultsSubparser(CustomSubparser):
    def __init__(self):
        super().__init__('grabresults')

    def add_arguments(self):
        self.subparser.add_argument('--res', metavar='RES', help='research ID')
        self.subparser.add_argument('--remote', metavar='REMOTE', help='remote machine ID')
        self.subparser.add_argument('tasks', metavar='TASKS', nargs='+', type=int, help='a list of tasks to download')

    def handler(self, args):
        local_comm = LocalCommunication.create_from_config()
        ssh_comm = SshCommunication.create_from_config(args.remote)
        res = Research.open(args.res, ssh_comm)
        for t in args.tasks:
            res.grab_task_results(t)


class CloudSubparser(CustomSubparser):
    def __init__(self):
        super().__init__('cloud')

    def add_arguments(self):
        self.subparser.add_argument('action', metavar='ACTION', nargs='?', choices=['download', 'upload', 'list'], 
                                    default='download',
                                    help='action applied to tasks')
        self.subparser.add_argument('--res', metavar='RES', help='research ID')
        self.subparser.add_argument('--tasks', metavar='TASKS', nargs='*', type=int, help='a list of tasks to process')

    def handler(self, args):
        if args.action == 'list':
            res = Research.open(args.res)
            print(self._list_tasks_in_aws_format(res))
        elif args.action == 'download':
            res = Research.open(args.res)
            task_list = self._from_aws_format_to_python_list(self._list_tasks_in_aws_format(res))
            required_task_numbers = args.tasks
            task_archives_to_download = []
            for task_name in task_list:
                task_number = split_task_dir(task_name)[0]
                if task_number in required_task_numbers:
                    task_archives_to_download.append(task_name)
            if len(task_archives_to_download) == 0:
                raise Exception(f'No appropriate tasks are found in the cloud. Please check that '
                                f'research name is correct: "{res.research_dir}" '
                                f'and tasks "{required_task_numbers}" do exist')
            for task_archive in task_archives_to_download:
                self._download_archive_from_aws_cloud(res, task_archive)
        elif args.action == 'upload':
            res = Research.open(args.res)
            for task_number in args.tasks:
                self._upload_task_to_aws_cloud_as_archive(res, task_number)

    def _list_tasks_in_aws_format(self, res):
        command_line = 'aws --endpoint-url=https://storage.yandexcloud.net s3 ls --recursive s3://' + res.research_dir
        result = subprocess.run(command_line, stdout=subprocess.PIPE, shell=True)
        return result.stdout.decode('utf-8')

    def _download_archive_from_aws_cloud(self, res, archive_name):
        local_archive_path = os.path.join(res.local_research_path, archive_name)
        download_command_line = 'aws --endpoint-url=https://storage.yandexcloud.net s3 ' \
                               f'cp s3://{res.research_dir}/{archive_name} {local_archive_path}'
        subprocess.run(download_command_line, shell=True)
        extract_command_line = f'tar -zxvf {local_archive_path} --directory {res.local_research_path}'
        subprocess.run(extract_command_line, shell=True)
        remove_archive_command_line = f'rm {local_archive_path}'
        subprocess.run(remove_archive_command_line, shell=True)

    def _upload_task_to_aws_cloud_as_archive(self, res, task):
        task_path = res.get_task_path(task)
        task_name = os.path.basename(task_path)
        archive_name = f'{task_name}.tar.gz'
        local_archive_path = os.path.join(res.local_research_path, archive_name)
        compress_command_line = f'cd {res.local_research_path}; tar -czvf {archive_name} {task_name}'  # doing cd is more robust than using --directory
        subprocess.run(compress_command_line, shell=True)
        upload_command_line = 'aws --endpoint-url=https://storage.yandexcloud.net s3 ' \
                               f'cp {local_archive_path} s3://{res.research_dir}/{archive_name}'
        subprocess.run(upload_command_line, shell=True)
        remove_archive_command_line = f'rm {local_archive_path}'
        subprocess.run(remove_archive_command_line, shell=True)

    def _from_aws_format_to_python_list(self, list_in_aws_format):
        task_list = []
        rows = list_in_aws_format.split('\n')
        for row in rows:
            entries = row.split()
            if len(entries) == 4:
                task_list.append(entries[3])
        return task_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Manages different objects related to research.')
    subparsers = parser.add_subparsers(title='objects to manage')
    for cls in (ResearchSubparser, MeetingSubparser, GrabResultsSubparser, CloudSubparser):
        subparser = cls()
        subparser.add_subparser(subparsers)
        subparser.add_arguments()
    args = parser.parse_args()
    args.func(args)
