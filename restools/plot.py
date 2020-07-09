import os
import sys
sys.path.append(os.getcwd())
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from restools.timeintegration_builders import get_ti_builder
from restools.plotting import put_fields_on_axes
from comsdk.research import Research
from thequickmath.field import read_field


def make_path_root_out_of_path_related_arguments(args):
    path = ''
    if args.research is not None:
        res = Research.open(args.research)
        if args.task is not None:
            path = os.path.join(path, res.get_task_path(args.task))
            if args.param is not None:
                path = os.path.join(path, 'data-{}'.format(args.param))
        else:
            path = os.path.join(path, res.local_research_path)
    return path


def handle_field(args):
    path = make_path_root_out_of_path_related_arguments(args)
    # Augment the path with a filename
    if args.path is not None:
        path = os.path.join(path, args.path)
        f, _ = read_field(path)
    elif args.time is not None:
        f = get_ti_builder(args.program).get_timeintegration(path).solution(args.time)
    else:
        raise ValueError('Neither --path nor --time are specified: cannot build the filename')
    # Plot field according to the domain size
    if args.domain == 'large':
        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(nrows=1, ncols=6, figure=fig)
        ax_zy = fig.add_subplot(gs[0, 0])
        ax_zx = fig.add_subplot(gs[0, 1:])
        put_fields_on_axes(f, ax_zx=ax_zx, ax_zy=ax_zy, enable_quiver=False, vertical=True)
        ax_zx.set_xlabel('x', fontsize=16)
        ax_zy.set_xlabel('y', fontsize=16)
        ax_zy.set_ylabel('z', fontsize=16)
    elif args.domain == 'wide':
        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(nrows=3, ncols=1, figure=fig)
        ax_zy = fig.add_subplot(gs[-1, 0])
        ax_zx = fig.add_subplot(gs[0:-1, 0])
        put_fields_on_axes(f, ax_zx=ax_zx, ax_zy=ax_zy, enable_quiver=False, vertical=False)
        ax_zx.set_ylabel('x', fontsize=16)
        ax_zy.set_xlabel('z', fontsize=16)
        ax_zy.set_ylabel('y', fontsize=16)
    elif args.domain == 'small':
        fig, axes = plt.subplots(2, 1, figsize=(5, 6))
        ax_zx = axes[0]
        ax_zy = axes[1]
        put_fields_on_axes(f, ax_zx=ax_zx, ax_zy=ax_zy, enable_quiver=False, vertical=False)
        ax_zx.set_ylabel('x', fontsize=16)
        ax_zy.set_xlabel('z', fontsize=16)
        ax_zy.set_ylabel('y', fontsize=16)
    else:
        raise ValueError('Unknown domain: {}'.format(args.domain))
    plt.tight_layout()
    if args.save is not None:
        plt.savefig(args.save, dpi=200)
    plt.show()


def handle_spacetime(args):
    path = make_path_root_out_of_path_related_arguments(args)
    # Augment the path with an additional bit from args.path
    if args.path is not None:
        path = os.path.join(path, args.path)
    ti = get_ti_builder(args.program).get_timeintegration(path)
    st_data = getattr(ti, args.dataid)
    # Plot contours
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    cvals = 200
    cmap = matplotlib.cm.jet
    X_, Y_ = np.meshgrid(st_data.space.elements[0], st_data.space.elements[1], indexing='ij')
    cont = ax.contourf(X_, Y_, st_data.elements[0], cvals, cmap=cmap)
    ax.set_xlabel('${}$'.format(st_data.space.elements_names[0]), fontsize=16)
    ax.set_ylabel('${}$'.format(st_data.space.elements_names[1]), fontsize=16)
    plt.tight_layout()
    if args.save is not None:
        plt.savefig(args.save, dpi=200)
    plt.show()


def add_path_related_arguments(p: argparse.ArgumentParser):
    p.add_argument('-r', '--research', metavar='RESEARCH',
                   help='research id')
    p.add_argument('-t', '--task', metavar='TASK', type=int,
                   help='task number')
    p.add_argument('-p', '--param', metavar='PARAM',
                   help='parameter value (it is to be used when the task folder contains sub-folders of the '
                        'format "data-PARAMVALUE")')
    p.add_argument('--path', metavar='PATH',
                   help='path to the object (if research and task are supplied, then the path is '
                        'relative with respect to them)')


def add_save_related_arguments(p: argparse.ArgumentParser):
    p.add_argument('-s', '--save', metavar='SAVE',
                   help='file to which figure should be saved (extension to be included)')


def add_program_related_arguments(p: argparse.ArgumentParser):
    p.add_argument('-pr', '--program', metavar='PARAM', choices=['cfv1', 'cfv2'], default='cfv2',
                   help='identification of the program generating the field')


def add_field_parser(subparsers):
    field_parser = subparsers.add_parser('field')
    add_path_related_arguments(field_parser)
    add_save_related_arguments(field_parser)
    add_program_related_arguments(field_parser)
    field_parser.add_argument('-d', '--domain', metavar='PARAM', choices=['large', 'wide', 'small'], default='large',
                              help='size of the domain, will determine the scheme of representation')
    field_parser.add_argument('-T', '--time', metavar='TIME', type=float,
                              help='time to which the field corresponds')
    field_parser.set_defaults(func=handle_field)


def add_spacetime_parser(subparsers):
    st_parser = subparsers.add_parser('spacetime')
    add_path_related_arguments(st_parser)
    add_save_related_arguments(st_parser)
    add_program_related_arguments(st_parser)
    st_parser.add_argument('-di', '--dataid', metavar='DATAID', default='ke_z',
                              help='data id (in the context of TimeIntegration) corresponding to space-time data')
    st_parser.set_defaults(func=handle_spacetime)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plots different objects related to research.')
    subparsers = parser.add_subparsers(title='objects to plot')
    add_field_parser(subparsers)
    add_spacetime_parser(subparsers)
    args = parser.parse_args()
    args.func(args)
