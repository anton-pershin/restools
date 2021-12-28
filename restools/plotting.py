import subprocess
from inspect import getmembers, isclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from thequickmath.field import *


def cell_heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels (copied from matplotlib web-site!).

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
#    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
#             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def build_zooming_axes(fig, parent_ax, parent_point, child_box, connecting_vertices):
    po_ax = plt.axes(child_box) 
    po_ax.get_xaxis().set_visible(False)
    po_ax.get_yaxis().set_visible(False)

    po_ax_bbox = po_ax.get_position()
    ax_bbox = parent_ax.get_position()

    combined_transform = fig.transFigure + parent_ax.transData.inverted()
    data_to_figure = combined_transform.inverted()
    parent_point_fig = data_to_figure.transform(parent_point)
    vertices_coords = (
            (po_ax_bbox.x0, po_ax_bbox.y0),
            (po_ax_bbox.x0, po_ax_bbox.y1),
            (po_ax_bbox.x1, po_ax_bbox.y1),
            (po_ax_bbox.x1, po_ax_bbox.y0),
        )
    parent_ax.plot([parent_point_fig[0], vertices_coords[connecting_vertices[0]][0]], [parent_point_fig[1], vertices_coords[connecting_vertices[0]][1]], 'k', alpha=0.5, transform=fig.transFigure)
    parent_ax.plot([parent_point_fig[0], vertices_coords[connecting_vertices[1]][0]], [parent_point_fig[1], vertices_coords[connecting_vertices[1]][1]], 'k', alpha=0.5, transform=fig.transFigure)
    return po_ax

def build_zooming_axes_for_plotting(fig, parent_ax, parent_point, child_box, connecting_vertices):
    po_ax = plt.axes(child_box) 

    po_ax_bbox = po_ax.get_position()
    ax_bbox = parent_ax.get_position()

    combined_transform = fig.transFigure + parent_ax.transData.inverted()
    data_to_figure = combined_transform.inverted()
    parent_point_fig = data_to_figure.transform(parent_point)
    vertices_coords = (
            (po_ax_bbox.x0, po_ax_bbox.y0),
            (po_ax_bbox.x0, po_ax_bbox.y1),
            (po_ax_bbox.x1, po_ax_bbox.y1),
            (po_ax_bbox.x1, po_ax_bbox.y0),
        )
    parent_ax.plot([parent_point_fig[0], vertices_coords[connecting_vertices[0]][0]], [parent_point_fig[1], vertices_coords[connecting_vertices[0]][1]], 'k', alpha=0.5, transform=fig.transFigure)
    parent_ax.plot([parent_point_fig[0], vertices_coords[connecting_vertices[1]][0]], [parent_point_fig[1], vertices_coords[connecting_vertices[1]][1]], 'k', alpha=0.5, transform=fig.transFigure)
    return po_ax

def build_zooming_axes_for_plotting_with_box(fig, parent_ax, parent_box, child_box, parent_vertices, child_vertices, remove_axis=False):
    combined_transform = fig.transFigure + parent_ax.transData.inverted()
    data_to_figure = combined_transform.inverted()

    child_box_fig = (
            data_to_figure.transform((child_box[0], child_box[1])),
            data_to_figure.transform((child_box[0], child_box[1] + child_box[3])),
            data_to_figure.transform((child_box[0] + child_box[2], child_box[1] + child_box[3])),
            data_to_figure.transform((child_box[0] + child_box[2], child_box[1])),
        )
    po_ax = plt.axes((child_box_fig[0][0], child_box_fig[0][1], child_box_fig[3][0] - child_box_fig[0][0], child_box_fig[1][1] - child_box_fig[0][1]))
    if remove_axis:
        po_ax.get_xaxis().set_visible(False)
        po_ax.get_yaxis().set_visible(False)

    po_ax_bbox = po_ax.get_position()
    ax_bbox = parent_ax.get_position()

    #parent_point_fig = data_to_figure.transform(parent_point)
    parent_vertices_fig = (
            data_to_figure.transform((parent_box[0], parent_box[1])),
            data_to_figure.transform((parent_box[0], parent_box[1] + parent_box[3])),
            data_to_figure.transform((parent_box[0] + parent_box[2], parent_box[1] + parent_box[3])),
            data_to_figure.transform((parent_box[0] + parent_box[2], parent_box[1])),
        )
    child_vertices_fig = (
            (po_ax_bbox.x0, po_ax_bbox.y0),
            (po_ax_bbox.x0, po_ax_bbox.y1),
            (po_ax_bbox.x1, po_ax_bbox.y1),
            (po_ax_bbox.x1, po_ax_bbox.y0),
        )
    parent_ax.plot([parent_box[0], parent_box[0] + parent_box[2]], [parent_box[1], parent_box[1]], 'k',
                   linewidth=1, alpha=0.5)
    parent_ax.plot([parent_box[0], parent_box[0] + parent_box[2]], [parent_box[1] + parent_box[3], parent_box[1] + parent_box[3]], 'k',
                   linewidth=1, alpha=0.5)
    parent_ax.plot([parent_box[0], parent_box[0]], [parent_box[1], parent_box[1] + parent_box[3]], 'k',
                   linewidth=1, alpha=0.5)
    parent_ax.plot([parent_box[0] + parent_box[2], parent_box[0] + parent_box[2]], [parent_box[1], parent_box[1] + parent_box[3]], 'k',
                   linewidth=1, alpha=0.5)
    parent_ax.plot([parent_vertices_fig[parent_vertices[0]][0], child_vertices_fig[child_vertices[0]][0]],
        [parent_vertices_fig[parent_vertices[0]][1], child_vertices_fig[child_vertices[0]][1]], 'k',
                   alpha=0.5, linewidth=1, transform=fig.transFigure)
    parent_ax.plot([parent_vertices_fig[parent_vertices[1]][0], child_vertices_fig[child_vertices[1]][0]],
        [parent_vertices_fig[parent_vertices[1]][1], child_vertices_fig[child_vertices[1]][1]], 'k',
                   alpha=0.5, linewidth=1, transform=fig.transFigure)
    return po_ax

def put_fields_on_axes(f, ax_zx=None, ax_zy=None, enable_quiver=True, vertical=False):
    def _prepare_zx(field_):
        #y_averaged_field = average(field_, ['u', 'v', 'w'], 'y')
        y_averaged_field = at(field_, 'y', 0.0)
        y_averaged_field.change_space_order(['z', 'x'])
        return y_averaged_field, filter(filter(filter(y_averaged_field, 'z', 0.5), 'z', 0.5), 'x', 0.5)

    def _prepare_zy(field_):
        x_averaged_field = average(field_, ['u', 'v', 'w'], 'x')
        x_averaged_field.change_space_order(['z', 'y'])
        return x_averaged_field, filter(filter(filter(filter(x_averaged_field, 'z', 0.5), 'z', 0.5), 'y', 0.5), 'y', 0.5)

    def _plot_contours_and_arrow(ax_, X_cont, Y_cont, cont_field_raw, X_quiv, Y_quiv, quiv_field_X, quiv_field_Y, arrow_scale, enable_quiver):
        cvals = 50
        cont = ax_.contourf(X_cont, Y_cont, cont_field_raw, cvals, cmap=matplotlib.cm.jet)#, vmin=-0.6, vmax=0.6)
        if enable_quiver:
            ax_.quiver(X_quiv, Y_quiv, quiv_field_X, quiv_field_Y,  # assign to var
                   color='Teal', 
                   scale=arrow_scale,
                   headlength=3)
            #      linewidth=0.05)
        return cont

    zx_field, zx_field_quiv = _prepare_zx(f)
    zy_field, zy_field_quiv = _prepare_zy(f)

    # Generate data for plotting
    if vertical:
        X_zx, Y_zx = np.meshgrid(zx_field.space.elements[1], zx_field.space.elements[0], indexing='ij')
        X_zy, Y_zy = np.meshgrid(zy_field.space.elements[1], zy_field.space.elements[0], indexing='ij')
    else:
        X_zx, Y_zx = np.meshgrid(zx_field.space.elements[0], zx_field.space.elements[1], indexing='ij')
        X_zy, Y_zy = np.meshgrid(zy_field.space.elements[0], zy_field.space.elements[1], indexing='ij')

    X_zx_quiv, Y_zx_quiv = np.meshgrid(zx_field_quiv.space.elements[0], zx_field_quiv.space.elements[1], indexing='ij')
    X_zy_quiv, Y_zy_quiv = np.meshgrid(zy_field_quiv.space.elements[0], zy_field_quiv.space.elements[1], indexing='ij')

    cvals = 50
    #ax_top.set_aspect(2, adjustable='box-forced')
    #ax_bottom.set_aspect(4, adjustable='box-forced')

    #vmin = np.min((np.min(zx_field.elements[0]), np.min(zy_field.elements[0])))
    #vmax = np.max((np.max(zx_field.elements[0]), np.max(zy_field.elements[0])))

    #cont_zx = ax_top.contourf(X_zx, Y_zx, zx_field.elements[0], cvals, vmin=vmin, vmax=vmax)
    #cont_zy = ax_bottom.contourf(X_zy, Y_zy, zy_field.elements[0], cvals, vmin=vmin, vmax=vmax)
    arrow_scale_zx = 15. / np.sqrt(np.max(np.abs(zx_field.elements[0]))**2 + np.max(np.abs(zx_field.elements[1]))**2)
    arrow_scale_zy = 3. / np.sqrt(np.max(np.abs(zy_field.elements[0]))**2 + np.max(np.abs(zy_field.elements[1]))**2)

    conts = []
    if ax_zx is not None:
        f_raw = zx_field.elements[0].T if vertical else zx_field.elements[0]
        conts.append(_plot_contours_and_arrow(ax_zx, X_zx, Y_zx, f_raw, X_zx_quiv, Y_zx_quiv,
                                              zx_field_quiv.elements[2], zx_field_quiv.elements[0], arrow_scale_zx,
                                              enable_quiver))
#        ax_top.set_xlim((0, f.space.z[-1]))
#        ax_top.set_ylim((0, f.space.x[-1]))
    if ax_zy is not None:
        f_raw = zy_field.elements[0].T if vertical else zy_field.elements[0]
        conts.append(_plot_contours_and_arrow(ax_zy, X_zy, Y_zy, f_raw, X_zy_quiv, Y_zy_quiv,
                                              zy_field_quiv.elements[2], zy_field_quiv.elements[1], arrow_scale_zy,
                                              enable_quiver))
        if vertical:
            ax_zy.set_xlim((-1, 1))
            ax_zy.set_xticks((-1, 0., 1))
        else:
            ax_zy.set_ylim((-1, 1))
            ax_zy.set_yticks((-1, 0., 1))

    return conts

def plot_filled_contours(ax, field_2d, cvals, **contourf_kwargs):
    X_, Y_ = np.meshgrid(field_2d.space.elements[0], field_2d.space.elements[1], indexing='ij')
    return ax.contourf(X_, Y_, field_2d.elements[0], cvals, **contourf_kwargs)

def plot_lines(x_list, y_list, layout=None, titles=None, labels=None, legend_loc='lower right', elongated=None, ylog=False):
    if layout is None: # assume horizontal
        layout = (1, len(x_list))
    if titles is None:
        titles = [None for _ in x_list]
    has_legend = True if labels is not None else False
    standard_length = 4
    x_length = layout[1] * standard_length
    y_length = layout[0] * standard_length
    if elongated == 'x':
        x_length *= 2
    elif elongated == 'y':
        y_length *= 2

    # We suppose that 10 is maximum vertical length, so descrease the sizes proportionally if it is too large
    if y_length > 10:
        y_length = 10

    fig, axes = plt.subplots(layout[0], layout[1], figsize=(x_length, y_length))

    if len(x_list) == 1:
        if has_legend:
            _plotting_func(axes, ylog)(x_list[0].values, y_list[0].values, label=labels[0])
        else:
            _plotting_func(axes, ylog)(x_list[0].values, y_list[0].values)
        _put_data_on_2d_axes(axes, x_list[0].label, y_list[0].label, has_legend=has_legend, legend_loc=legend_loc)
    else:
        if layout[0] == 1 or layout[1] == 1:
            for i in range(len(x_list)):
                if has_legend:
                    _plotting_func(axes[i], ylog)(x_list[i].values, y_list[i].values, label=labels[i])
                else:
                    _plotting_func(axes[i], ylog)(x_list[i].values, y_list[i].values)
                _put_data_on_2d_axes(axes[i], x_list[i].label, y_list[i].label, title=titles[i], has_legend=has_legend, legend_loc=legend_loc)
        else: 
            for row in range(layout[0]):
                for col in range(layout[1]):
                    i = row + col
                    if has_legend:
                        _plotting_func(axes[row, col], ylog)(x_list[i].values, y_list[i].values, label=labels[i])
                    else:
                        _plotting_func(axes[row, col], ylog)(x_list[i].values, y_list[i].values)
                    _put_data_on_2d_axes(axes[row, col], x_list[i].label, y_list[i].label, title=titles[i], has_legend=has_legend, legend_loc=legend_loc)
    return fig, axes

def plot_lines_on_one_plot(x_list, y_list, labels=[], lines_types=[], ylog=False, elongated=None, linewidth=2, legend_loc='lower right', xlim=None, ylim=None):
    '''
    elongated keyword should be a string (either 'x' or 'y')
    '''
    standard_length = 4
    x_length = 2*standard_length if elongated == 'x' else standard_length
    y_length = 2*standard_length if elongated == 'y' else standard_length
    fig, ax = plt.subplots(1, 1, figsize=(x_length, y_length))
    has_labels = True if len(labels) != 0 else False
    if len(lines_types) == 0:
        lines_types = ['-' for _ in range(len(x_list))]
    for i in range(len(x_list)):
        if has_labels:
            _plotting_func(ax, ylog)(x_list[i].values, y_list[i].values, lines_types[i], linewidth=linewidth, label=labels[i])
        else:
            _plotting_func(ax, ylog)(x_list[i].values, y_list[i].values, lines_types[i], linewidth=linewidth)
    _put_data_on_2d_axes(ax, x_list[0].label, y_list[0].label, has_legend=has_labels, legend_loc=legend_loc, xlim=xlim, ylim=ylim)
    return fig, ax

def plot_composite(plot_targets, layout=None):
    if layout is None: # assume horizontal
        layout = (len(plot_targets), 1)
    fig, axes = plt.subplots(layout[0], layout[1])
    for i, target in enumerate(plot_targets):
        if isinstance(target, Field):
            coord1 = target.space.elements[0]
            coord2 = target.space.elements[1]
            x, y = np.meshgrid(coord1, coord2, indexing='ij') # ij-indexing guarantees correct order of indexes 
                                                              # where the first index correponds to the x-coordinate
            raw_scalar_field = target.elements[0]
            p = axes[i].contourf(x, y, raw_scalar_field, 100, cmap=matplotlib.cm.jet, vmin=raw_scalar_field.min(), vmax=raw_scalar_field.max(), aspect='auto')
            #ax.set_aspect('equal') # to make x,y axes scales correct
            #cb = fig.colorbar(p, ax=ax[0], shrink=0.25) # "shrink" is used to make color bar small enough (it is very long if not used)
            axes[i].set_xlim(coord1.min(), coord1.max())
            axes[i].set_ylim(coord2.min(), coord2.max())
        else:
            labeled_x = target[0]
            labeled_y = target[1]
            _plotting_func(axes[i], False)(labeled_x.values, labeled_y.values)
            _put_data_on_2d_axes(axes[i], labeled_x.label, labeled_y.label)
            axes[i].set_xlim(labeled_x.values.min(), labeled_x.values.max())
    return fig, axes

def plot_aligned(fields):
    max_x_length = 8
    each_y_length = 4
    max_coord1_val = 0
    layout = (len(fields), 1)
    fig, axes = plt.subplots(layout[0], layout[1], sharex=True)
    minmax = [(field_.elements[0].min(), field_.elements[0].max()) for field_ in fields]
    min_value = 0
    max_value = 0
    for field_ in fields:
        if min_value > field_.elements[0].min():
            min_value = field_.elements[0].min()
        if max_value < field_.elements[0].max():
            max_value = field_.elements[0].max()

    for i, field_ in enumerate(fields):
        coord1 = field_.space.elements[0]
        coord2 = field_.space.elements[1]
        x, y = np.meshgrid(coord1, coord2, indexing='ij') # ij-indexing guarantees correct order of indexes 
                                                          # where the first index correponds to the x-coordinate
        raw_scalar_field = field_.elements[0]
        #p = axes[i].contourf(x, y, raw_scalar_field, 100, cmap=matplotlib.cm.jet, vmin=min_value,\
        #                    vmax=max_value)
        p = axes[i].contourf(x, y, raw_scalar_field, 100, cmap=matplotlib.cm.jet, vmin=field_.elements[0].min(),\
                            vmax=field_.elements[0].max())
        #ax.set_aspect('equal') # to make x,y axes scales correct
        cb = fig.colorbar(p, ax=axes[i]) # "shrink" is used to make color bar small enough (it is very long if not used)
        #axes[i].set_xlim(coord1.min(), coord1.max())
        #axes[i].set_ylim(coord2.min(), coord2.max())
    return fig, axes

def quive_field_2d(field_2d):
    coord1 = field_2d.space.elements[0]
    coord2 = field_2d.space.elements[1]
    x, y = np.meshgrid(coord1, coord2, indexing='ij') # ij-indexing guarantees correct order of indexes 
    fig, ax = plt.subplots()
    #ax.quiver(x, y, field_2d.elements[0], field_2d.elements[1], length=0.15)
    ax.quiver(x, y, field_2d.elements[0], field_2d.elements[1], units='width')
    _put_labels_on_axes(ax, field_2d.space)
    #ax.set_aspect('equal')
    return fig, ax

def streamplot_field_2d(field_2d):
    coord1 = field_2d.space.elements[0]
    coord2 = field_2d.space.elements[1]
    fig, ax = plt.subplots()
    #ax.quiver(x, y, field_2d.elements[0], field_2d.elements[1], length=0.15)
    field_norm = np.sqrt(field_2d.elements[0]**2 + field_2d.elements[1]**2)
    linewidth_factor = 2.0/np.max(field_norm)
    #color=np.transpose(field_norm)
    ax.streamplot(coord1, coord2, np.transpose(field_2d.elements[0]), np.transpose(field_2d.elements[1]), \
        linewidth=linewidth_factor*np.transpose(field_norm), color=np.transpose(field_norm), cmap=plt.cm.inferno, density=2, arrowsize=2)

    _put_labels_on_axes(ax, field_2d.space)
    #ax.set_aspect('equal')
    return fig, ax

def quive_plot_field(field):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    x, y, z = np.meshgrid(field.x, field.y, field.z, indexing='ij')
    ax.quiver(x, y, z, field.u, field.v, field.w, length=0.15, color='Tomato')
    ax.set_title('Test it')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev=18, azim=30)
    ax.dist=8

    _build_fake_3d_box(ax, field.x, field.y, field.z)
    plt.show()

def _build_fake_3d_box(ax, x, y, z):
    # create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
    x_box = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(x.max() + x.min())
    y_box = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(y.max() + y.min())
    z_box = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(z.max() + z.min())

    for xb, yb, zb in zip(x_box, y_box, z_box):
        ax.plot([xb], [yb], [zb], 'w')

def _plotting_func(ax, ylog):
    return ax.semilogy if ylog else ax.plot

def _put_labels_on_axes(ax, space):
    ax.set_xlabel(space.elements_names[0])
    ax.set_ylabel(space.elements_names[1])

def _put_data_on_2d_axes(ax, xlabel, ylabel, has_grid=True, has_legend=False, legend_loc='lower right', title=None, xlim=None, ylim=None):
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if has_grid:
        ax.grid()
    if has_legend:
        ax.legend(loc=legend_loc, fontsize='x-small')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

def label_axes(ax, label, loc=None, **kwargs):
    """
    DISCLAIMER: adapted from tacaswell's gist.
    Labels axes.
    kwargs are collected and passed to `annotate`
    Parameters
    ----------
    ax : Axes
         Axes object to work on
    label : string
        string to use to label the axes.
    loc : len=2 tuple of floats
        Where to put the label in axes-fraction units
    """

    if loc is None:
        loc = (.9, .9)
    ax.annotate(label, xy=loc,
                xycoords='axes fraction',
                **kwargs)

def rasterise_and_save(fname, rasterise_list=None, fig=None, dpi=None, savefig_kw={}) -> None:
    """
    Saves a figure with raster and vector components. This function lets you specify which objects to rasterise at the
    export stage, rather than within each plotting call. Rasterising certain components of a complex figure can
    significantly reduce file size. If rasterise_list is not specified, then all contour, pcolor, and collects objects
    (e.g., ``scatter, fill_between`` etc) will be rasterised.
    Adapted from https://gist.github.com/hugke729/78655b82b885cde79e270f1c30da0b5f

    :param fname: output filename with extension
    :param rasterise_list: list of objects to rasterize (or a single object to rasterize)
    :param fig: defaults to current figure
    :param dpi: resolution (dots per inch) for rasterising
    :param savefig_kw: extra keywords to pass to matplotlib.pyplot.savefig

    :warning: does not work correctly with round=True in Basemap
    """

    # Behave like pyplot and act on current figure if no figure is specified
    fig = plt.gcf() if fig is None else fig

    # Need to set_rasterization_zorder in order for rasterizing to work
    zorder = -5  # Somewhat arbitrary, just ensuring less than 0
    zorder_min = -1000  # We will preserve all zorders by shifting them down to this amount

    if rasterise_list is None:
        # Have a guess at stuff that should be rasterised
        types_to_raster = ['QuadMesh', 'Contour', 'collections']
        rasterise_list = []

        print("""
        No rasterize_list specified, so the following objects will
        be rasterized: """)
        # Get all axes, and then get objects within axes
        for ax in fig.get_axes():
            for item in ax.get_children():
                if any(x in str(item) for x in types_to_raster):
                    rasterise_list.append(item)
        print('\n'.join([str(x) for x in rasterise_list]))
    else:
        # Allow rasterize_list to be input as an object to rasterize
        if type(rasterise_list) != list:
            rasterise_list = [rasterise_list]

    for item in rasterise_list:

        # Whether or not plot is a contour plot is important
        is_contour = (isinstance(item, matplotlib.contour.QuadContourSet) or
                      isinstance(item, matplotlib.tri.TriContourSet))

        # Whether or not collection of lines
        # This is commented as we seldom want to rasterize lines
        # is_lines = isinstance(item, matplotlib.collections.LineCollection)

        # Whether or not current item is list of patches
        all_patch_types = tuple(
            x[1] for x in getmembers(matplotlib.patches, isclass))
        try:
            is_patch_list = isinstance(item[0], all_patch_types)
        except TypeError:
            is_patch_list = False

        # Convert to rasterized mode and then change zorder properties
        if is_contour:
            curr_ax = item.ax.axes
            curr_ax.set_rasterization_zorder(zorder)
            # For contour plots, need to set each part of the contour
            # collection individually
            for contour_level in item.collections:
                contour_level.set_zorder(zorder_min + contour_level.get_zorder())
                contour_level.set_rasterized(True)
        elif is_patch_list:
            # For list of patches, need to set zorder for each patch
            for patch in item:
                curr_ax = patch.axes
                curr_ax.set_rasterization_zorder(zorder)
                patch.set_zorder(zorder_min + patch.get_zorder())
                patch.set_rasterized(True)
            if hasattr(item, 'errorbar'):
                if item.errorbar is not None:
                    data_line = item.errorbar.lines[0]
                    caplines_tuple = item.errorbar.lines[1]
                    barlinecols_tuple = item.errorbar.lines[2]
                    for obj in [data_line] + list(caplines_tuple) + list(barlinecols_tuple):
                        if obj is not None:
                            obj.set_zorder(zorder_min + obj.get_zorder())
        else:
            # For all other objects, we can just do it all at once
            curr_ax = item.axes
            curr_ax.set_rasterization_zorder(zorder)
            item.set_rasterized(True)
            item.set_zorder(zorder_min + item.get_zorder())
    # dpi is a savefig keyword argument, but treat it as special since it is
    # important to this function
    if dpi is not None:
        savefig_kw['dpi'] = dpi

    # Save resulting figure
    fig.savefig(fname, **savefig_kw)


def reduce_eps_size(fname: str) -> None:
    """
    Reduces the size of an eps-file fname by converting it to pdf and then back to eps.

    :param fname: eps-file which will be reduced
    """
    pdf_name = '{}.pdf'.format(fname[:-4])
    command_line = 'epstopdf {}; pdftops -eps {}'.format(fname, pdf_name)
    subprocess.call([command_line], shell=True)
    os.remove(pdf_name)
