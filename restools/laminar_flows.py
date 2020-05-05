import numpy as np

from thequickmath.field import Field, Space


class PlaneCouetteFlow:
    """
    Class PlaneCouetteFlow represents the laminar solution for plane Couette flow. It defines the following attributes:

    re
      Reynolds number

    solution
      Field object with component u (streamwise velocity) and space y (wall-normal coordinate)

    dissipation rate
      Laminar dissipation rate

    ke
      Laminar kinetic energy
    """
    def __init__(self, re: float, y: np.ndarray):
        self.re = re
        space = Space([y])
        space.set_elements_names(['y'])
        self.solution = Field([space.y], space)
        self.solution.set_elements_names(['u'])
        self.dissipation_rate = 1./re
        self.ke = 1./6


class PlaneCouetteFlowWithInPhaseSpanwiseOscillations:
    """
    Class PlaneCouetteFlowWithInPhaseSpanwiseOscillations represents the laminar solution for plane Couette flow in
    presence of sinusoidal in-phase wall oscillations in the spanwise direction. It defines the following attributes:

    re
      Reynolds number

    a
      Amplitude

    omega
      Frequency

    solution
      Field object with components u (streamwise velocity), w (spanwise velocity) and space t (time), y (wall-normal
      coordinate)

    dissipation rate
      Laminar dissipation rate
    """
    def __init__(self, re: float, a: float, omega: float, t: np.ndarray, y: np.ndarray):
        self.re = re
        self.a = a
        self.omega = omega
        big_omega = np.sqrt(omega * re / 2.)
        space = Space([t, y])
        space.set_elements_names(['t', 'y'])
        t_, y_ = np.meshgrid(space.t, space.y, indexing='ij')
        big_lambda = np.cos(2.*big_omega) + np.cosh(2.*big_omega)
        y_plus = big_omega*(1. + y_)
        y_minus = big_omega*(1. - y_)
        f = (np.cosh(y_plus)*np.cos(y_minus) + np.cosh(y_minus)*np.cos(y_plus)) / big_lambda
        g = -(np.sinh(y_plus)*np.sin(y_minus) + np.sinh(y_minus)*np.sin(y_plus)) / big_lambda
        self.solution = Field([y_, a*(f*np.sin(omega*t_) + g*np.cos(omega*t_))], space)
        self.solution.set_elements_names(['u', 'w'])
        self.dissipation_rate = 1./re * (1. + a**2 * big_omega / 2. * (np.sinh(2.*big_omega) - np.sin(2.*big_omega)) /
                                         (np.cosh(2.*big_omega) + np.cos(2.*big_omega)))


class StokesBoundaryLayer:
    """
    Class StokesBoundaryLayer represents the solution for Stokes (oscillating) boundary layer. It defines the following
    attributes:

    re
      Reynolds number

    a
      Amplitude

    omega
      Frequency

    solution
      Field object with components u (streamwise velocity), w (spanwise velocity) and space t (time), y (wall-normal
      coordinate)
    """
    def __init__(self, re: float, a: float, omega: float, t: np.ndarray, y: np.ndarray):
        self.re = re
        self.a = a
        self.omega = omega
        big_omega = np.sqrt(omega * re / 2.)
        space = Space([t, y])
        space.set_elements_names(['t', 'y'])
        t_, y_ = np.meshgrid(space.t, space.y, indexing='ij')
        self.solution = Field([y_, a * np.exp(-big_omega*y_) * np.sin(omega*t_ - big_omega*y_)], space)
        self.solution.set_elements_names(['u', 'w'])
