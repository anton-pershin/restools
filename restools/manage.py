import os
import sys
sys.path.append(os.getcwd())
import argparse

import restools
from comsdk.research import Research
from restools.report import QuickPresentation


def handle_research(args):
    if args.action == 'create':
        new_research_descr = input('Enter new research name: ')
        new_research_id = input('Enter short id: ')
        res = Research.create(new_research_id, new_research_descr)
    else:
        raise ValueError('Program action "{}" is unknown'.format(args.action))


def handle_meeting(args):
    if args.action == 'create':
        year = input('Enter year: ')
        month = input('Enter month: ')
        day = input('Enter day: ')
        qp = QuickPresentation.from_config('Meeting notes', [int(year), int(month), int(day)])
        qp.print_out()
    else:
        raise ValueError('Program action "{}" is unknown'.format(args.action))


def add_standard_options(parser):
    parser.add_argument('action', metavar='ACTION', nargs='?', choices=['create',], default='create',
                        help='action applied to object (must be only create for now)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Manages different objects related to research.')
    subparsers = parser.add_subparsers(title='objects to manage')
    res_parser = subparsers.add_parser('research')
    meeting_parser = subparsers.add_parser('meeting')
    for p, handler in zip((res_parser, meeting_parser), (handle_research, handle_meeting)):
        add_standard_options(p)
        p.set_defaults(func=handler)
    args = parser.parse_args()
    args.func(args)
