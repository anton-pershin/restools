import numpy as np


def generate_random_perturbation(required_ke):
    rp = np.random.rand(9) - 0.5
    rp_ke = m.kinetic_energy(rp[np.newaxis, :])[0]
    norm_coeff = np.sqrt(rp_ke/required_ke)
    return rp / norm_coeff
