import os
import sys
sys.path.append(os.getcwd())
import argparse

import restools
from comsdk.research import Research


def handle_research(args):
    if args.action == 'create':
        new_research_descr = input('Enter new research name: ')
        new_research_id = input('Enter short id: ')
        res = Research.create(new_research_id, new_research_descr)
    else:
        raise ValueError('Program action "{}" is unknown'.format(args.action))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Manages different objects related to research.')
    subparsers = parser.add_subparsers(title='objects to manage')
    res_parser = subparsers.add_parser('research')
    res_parser.add_argument('action', metavar='ACTION', nargs='?', choices=['create',], default='create',
                            help='action applied to research object (must be only create for now)')
    res_parser.set_defaults(func=handle_research)
    args = parser.parse_args()
    args.func(args)
