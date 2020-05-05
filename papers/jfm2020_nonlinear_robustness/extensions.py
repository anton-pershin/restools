class LaminarSolutionInPlaneCouetteFlow

def W_stokes_layer(y, t): return A * np.exp(-Omega*y) * np.sin(omega_*t - Omega*y)
def W_inphase_spanwise_wall_oscillation(y, t):
    Lambda = np.cos(2.*Omega) + np.cosh(2.*Omega)
    y_plus = Omega*(1. + y)
    y_minus = Omega*(1. - y)
    f = (np.cosh(y_plus)*np.cos(y_minus) + np.cosh(y_minus)*np.cos(y_plus)) / Lambda
    g = -(np.sinh(y_plus)*np.sin(y_minus) + np.sinh(y_minus)*np.sin(y_plus)) / Lambda
    return A * (f*np.sin(omega_*t) + g*np.cos(omega_*t))
