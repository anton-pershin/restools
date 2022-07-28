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
from comsdk.misc import take_value_by_index
from thequickmath.field import read_field


def create_list_action(transform=None):
    class make_list_class(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            values_as_list = values.split(',')
            if transform is not None:
                values_as_list = [transform(v) for v in values_as_list]
            setattr(args, self.dest, values_as_list)
    return make_list_class


def append_lists_to_achieve_equal_length(*args):
    max_len = max([len(l_) for l_ in args if l_ is not None])
    for l_ in args:
        if l_ is not None:
            l_ += [l_[-1]] * (max_len - len(l_))
    return max_len


def make_path_root_out_of_path_related_arguments(research, task, param):
    path = ''
    if research is not None:
        res = Research.open(research)
        if task is not None:
            path = os.path.join(path, res.get_task_path(task))
            if param is not None:
                path = os.path.join(path, 'data-{}'.format(param))
        else:
            path = os.path.join(path, res.local_research_path)
    return path


def plot_field(ax_zx, ax_zy, f, domain):
    if domain == 'large':
        put_fields_on_axes(f, ax_zx=ax_zx, ax_zy=ax_zy, enable_quiver=False, vertical=True)
        ax_zx.set_xlabel('x', fontsize=16)
        ax_zy.set_xlabel('y', fontsize=16)
        ax_zy.set_ylabel('z', fontsize=16)
    elif domain in ['wide', 'small']:
        put_fields_on_axes(f, ax_zx=ax_zx, ax_zy=ax_zy, enable_quiver=False, vertical=False)
        ax_zx.set_ylabel('x', fontsize=16)
        ax_zy.set_xlabel('z', fontsize=16)
        ax_zy.set_ylabel('y', fontsize=16)
    else:
        raise ValueError('Unknown domain: {}'.format(args.domain))


def plot_spacetime(ax, st_data):
    cvals = 200
    cmap = matplotlib.cm.jet
    X_, Y_ = np.meshgrid(st_data.space.elements[0], st_data.space.elements[1], indexing='ij')
    cont = ax.contourf(X_, Y_, st_data.elements[0], cvals, cmap=cmap)
    ax.set_xlabel('${}$'.format(st_data.space.elements_names[0]), fontsize=16)
    ax.set_ylabel('${}$'.format(st_data.space.elements_names[1]), fontsize=16)


def handle_field(args):
    plots_num = append_lists_to_achieve_equal_length(args.research, args.task, args.param, args.path, args.time)
    # Build plot grid according to the domain size
    if args.domain == 'large':
        fig = plt.figure(figsize=(12, 5*plots_num))
        gs = gridspec.GridSpec(nrows=plots_num, ncols=6, figure=fig)
    elif args.domain == 'wide':
        fig = plt.figure(figsize=(12, 5*plots_num))
        gs = gridspec.GridSpec(nrows=3*plots_num, ncols=1, figure=fig)
    elif args.domain == 'small':
        fig = plt.figure(figsize=(5, 6*plots_num))
        gs = gridspec.GridSpec(nrows=2*plots_num, ncols=1, figure=fig)
    else:
        raise ValueError('Unknown domain: {}'.format(args.domain))

    for i in range(plots_num):
        path_root = make_path_root_out_of_path_related_arguments(take_value_by_index(args.research, i),
                                                                 take_value_by_index(args.task, i),
                                                                 take_value_by_index(args.param, i))
        # Augment the path with a filename
        if args.path is not None:
            path = os.path.join(path_root, args.path[i])
            f, _ = read_field(path)
        elif args.time is not None:
            f = get_ti_builder(args.program).get_timeintegration(path_root).solution(args.time[i])
        else:
            raise ValueError('Neither --path nor --time are specified: cannot build the filename')
        # Plot field according to the domain size
        if args.domain == 'large':
            ax_zy = fig.add_subplot(gs[i, 0])
            ax_zx = fig.add_subplot(gs[i, 1:])
        elif args.domain == 'wide':
            ax_zx = fig.add_subplot(gs[3*i:3*i+2, 0])
            ax_zy = fig.add_subplot(gs[3*i+2:3*i+3, 0])
        elif args.domain == 'small':
            ax_zx = fig.add_subplot(gs[2*i, 0])
            ax_zy = fig.add_subplot(gs[2*i + 1, 0])
        else:
            raise ValueError('Unknown domain: {}'.format(args.domain))
        plot_field(ax_zx, ax_zy, f, args.domain)
    plt.tight_layout()
    if args.save is not None:
        plt.savefig(args.save, dpi=200)
    plt.show()


def handle_spacetime(args):
    plots_num = append_lists_to_achieve_equal_length(args.research, args.task, args.param, args.path, args.dataid)
    fig, axes = plt.subplots(plots_num, 1, figsize=(12, 4*plots_num))
    if plots_num == 1:
        axes = [axes]
    max_time = 0.
    for i in range(plots_num):
        path = make_path_root_out_of_path_related_arguments(take_value_by_index(args.research, i),
                                                            take_value_by_index(args.task, i),
                                                            take_value_by_index(args.param, i))
        # Augment the path with an additional bit from args.path
        if args.path is not None:
            path = os.path.join(path, args.path[i])
        ti = get_ti_builder(args.program).get_timeintegration(path)
        st_data = getattr(ti, args.dataid[i])
        if st_data.space.t[-1] > max_time:
            max_time = st_data.space.t[-1]
        # Plot contours
        plot_spacetime(axes[i], st_data)
    for ax in axes:
        x_min, x_max = ax.get_xlim()
        ax.set_xlim((x_min, max_time))
    plt.tight_layout()
    if args.save is not None:
        plt.savefig(args.save, dpi=200)
    plt.show()


def handle_timeseries(args):
    plots_num = append_lists_to_achieve_equal_length(args.research, args.task, args.param, args.path, args.dataid)
    get_descr = lambda name, descr: '{} = {}'.format(name, descr) if descr is not None else ''
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for i in range(plots_num):
        path = make_path_root_out_of_path_related_arguments(take_value_by_index(args.research, i),
                                                            take_value_by_index(args.task, i),
                                                            take_value_by_index(args.param, i))
        # Augment the path with an additional bit from args.path
        if args.path is not None:
            path = os.path.join(path, args.path[i])
        ti = get_ti_builder(args.program).get_timeintegration(path)
        t = ti.T
        data = getattr(ti, args.dataid[i])
        label = get_descr('res', take_value_by_index(args.research, i)) + \
                get_descr('task', take_value_by_index(args.task, i)) + \
                get_descr('param', take_value_by_index(args.param, i)) + \
                get_descr('path', take_value_by_index(args.path, i)) + \
                get_descr('dataid', take_value_by_index(args.dataid, i))
        ax.plot(t, data, linewidth=2, label=label)
    ax.set_xlabel('$t$', fontsize=16)
    ax.grid()
    ax.legend()
    plt.tight_layout()
    if args.save is not None:
        plt.savefig(args.save, dpi=200)
    plt.show()


def add_path_related_arguments(p: argparse.ArgumentParser):
    p.add_argument('-r', '--research', metavar='RESEARCH', action=create_list_action(),
                   help='research id')
    p.add_argument('-t', '--task', metavar='TASK', action=create_list_action(int),
                   help='task number')
    p.add_argument('-p', '--param', metavar='PARAM', action=create_list_action(),
                   help='parameter value (it is to be used when the task folder contains sub-folders of the '
                        'format "data-PARAMVALUE")')
    p.add_argument('--path', metavar='PATH', action=create_list_action(),
                   help='path to the object (if research and task are supplied, then the path is '
                        'relative with respect to them)')


def add_save_related_arguments(p: argparse.ArgumentParser):
    p.add_argument('-s', '--save', metavar='SAVE',
                   help='file to which figure should be saved (extension to be included)')


def add_program_related_arguments(p: argparse.ArgumentParser):
    p.add_argument('-pr', '--program', metavar='PARAM', choices=['cfv1', 'cfv2'], default='cfv2',
                   help='identification of the program generating data')


def add_field_parser(subparsers):
    field_parser = subparsers.add_parser('field')
    add_path_related_arguments(field_parser)
    add_save_related_arguments(field_parser)
    add_program_related_arguments(field_parser)
    field_parser.add_argument('-d', '--domain', metavar='PARAM', choices=['large', 'wide', 'small'], default='large',
                              help='size of the domain, will determine the scheme of representation')
    field_parser.add_argument('-T', '--time', metavar='TIME', action=create_list_action(float),
                              help='time to which the field corresponds')
    field_parser.set_defaults(func=handle_field)


def add_spacetime_parser(subparsers):
    st_parser = subparsers.add_parser('spacetime')
    add_path_related_arguments(st_parser)
    add_save_related_arguments(st_parser)
    add_program_related_arguments(st_parser)
    st_parser.add_argument('-di', '--dataid', metavar='DATAID', default=['ke_z'], action=create_list_action(),
                           help='data id (in the context of TimeIntegration) corresponding to space-time data')
    st_parser.set_defaults(func=handle_spacetime)


def add_timeseries_parser(subparsers):
    st_parser = subparsers.add_parser('timeseries')
    add_path_related_arguments(st_parser)
    add_save_related_arguments(st_parser)
    add_program_related_arguments(st_parser)
    st_parser.add_argument('-di', '--dataid', metavar='DATAID', default=['L2U'], action=create_list_action(),
                           help='data id (in the context of TimeIntegration) corresponding to scalar time-series')
    st_parser.set_defaults(func=handle_timeseries)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plots different objects related to research.')
    subparsers = parser.add_subparsers(title='objects to plot')
    add_field_parser(subparsers)
    add_spacetime_parser(subparsers)
    add_timeseries_parser(subparsers)
    args = parser.parse_args()
    args.func(args)
