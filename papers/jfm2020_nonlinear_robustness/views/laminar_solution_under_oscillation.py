import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt

from restools.laminar_flows import PlaneCouetteFlowWithInPhaseSpanwiseOscillations, StokesBoundaryLayer


if __name__ == '__main__':
    plt.style.use('resources/default.mplstyle')

    A = 0.3
    omega = 1. / 64
    re = 500.
    big_omega = np.sqrt(omega * re / 2.)
    print('Stokes BL depth: {}'.format(1. / big_omega))

    t = np.linspace(0., np.pi / omega, 4, endpoint=False)
    y = np.linspace(-1., 1., 300)
    y_stokes = np.linspace(-1., 0., 300)
    pcf_wall_osc = PlaneCouetteFlowWithInPhaseSpanwiseOscillations(re, A, omega, t, y)
    stokes_bl = StokesBoundaryLayer(re, A, omega, t, y_stokes + 1.)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
    for i, ax in enumerate(axes):
        ax.plot(pcf_wall_osc.solution.w[i, :], y)
        ax.plot(stokes_bl.solution.w[i, :], y_stokes, '--', linewidth=2)
        ax.plot(stokes_bl.solution.w[i, :], y_stokes[::-1] + 1., '--', linewidth=2, color='k')
        ax.grid()
        ax.set_xlim((-0.33, 0.33))
        ax.set_ylim((-1.1, 1.1))
        ax.set_xticks([-A, 0., A])
        if i == 0:
            ttl = ax.set_title(r'$t = 0$')
            ax.set_ylabel(r'$y$')
        else:
            ax.set_yticklabels([])
            ttl = ax.set_title(r'$t = \dfrac{' + str(i) + r'}{8T}$', y=1.1)
        #ttl.set_position([.5, 0.5])
    plt.tight_layout(rect=(0, 0, 1, 1.04))
    plt.savefig('in_phase_lam_flow.eps')
    plt.show()
