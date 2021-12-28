import os
import sys
sys.path.append(os.getcwd())
import argparse
from abc import ABC, abstractmethod

import restools
from comsdk.research import Research
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Manages different objects related to research.')
    subparsers = parser.add_subparsers(title='objects to manage')
    for cls in (ResearchSubparser, MeetingSubparser, GrabResultsSubparser):
        subparser = cls()
        subparser.add_subparser(subparsers)
        subparser.add_arguments()
    args = parser.parse_args()
    args.func(args)
