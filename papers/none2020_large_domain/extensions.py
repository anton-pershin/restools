from functools import partial

import numpy as np

from thequickmath.field import read_field, write_field


def localise_random_field(random_field_path, output_field_path, a_x=30., b_x=10., a_z=10., b_z=10., c=1.):
    """
    Makes a localised turbulent spot out of a homogeneously random field located in file random_field_path and
    saves it as a Field object in file output_field_path
    :param random_field_path: path to the file with a random field
    :param output_field_path: path to the output file where the localised random field will be saved
    :param a_x: length of the turbulent spot along coordinate x
    :param b_x: `length` of the laminar-turbulent interface along coordinate x
    :param a_z: length of the turbulent spot along coordinate z
    :param b_z: `length` of the laminar-turbulent interface along coordinate z
    :return: output_field_path
    """
    f, attr = read_field(random_field_path)
    window_x = partial(window, a_x, b_x, attr['Lx'])
    window_z = partial(window, a_z, b_z, attr['Lz'])
    for i, x_ in enumerate(f.space.x):
        w_x = window_x(x_)
        for k, z_ in enumerate(f.space.z):
            w_z = window_z(z_)
            f.u[i, :, k] *= c * w_x * w_z
            f.v[i, :, k] *= c * w_x * w_z
            f.w[i, :, k] *= c * w_x * w_z
    write_field(f, attr, output_field_path)
    return output_field_path


def window(a, b, L, xi):
    """
    Returns the value of the windowing function for coordinate xi of a turbulent spot located in the middle of the
    domain
    :param a: length of the turbulent spot along coordinate xi
    :param b: `length` of the laminar-turbulent interface along coordinate xi
    :param xi: value of the coordinate xi
    :param L: length of the domain along coordinate xi
    """
    return 1./4 * (1. + np.tanh(6.*(a - xi + L/2.)/b + 3.)) * (1. + np.tanh(6.*(a + xi - L/2.)/b + 3.))
