from typing import List, Union

from jsons import JsonSerializable


class LaminarisationProbabilityInfo:
    """
    Class LaminarisationProbabilityInfo represents a summary data from the campaign of simulations for different
    amplitudes and frequencies of in-phase spanwise wall oscillations performed to estimate the laminarisation
    probability:

    res_id
      Research ID associated with the tasks containing the relevant simulations

    res_id_exception_for_A_03_omega_1_16
      Research ID associated with the tasks for A = 0.3, omega = 1/16

    re
      Reynolds number

    amplitudes
      A list of amplitudes

    frequencies
      A list of frequencies

    tasks
      A 2D-list of task numbers containing the relevant simulations (1st index = 1st index in amplitudes, 2nd
      index = 2nd index in frequencies). If task number is -1, then there is no data for the corresponding
      combination of amplitude and frequency

    s
      A 2D-list of estimated laminarisation scores (1st index = index in amplitudes, 2nd index = index in frequencies).

    e_a
      A 2D-list of estimated E_a (1st index = index in amplitudes, 2nd index = index in frequencies).

    e_flex
      A 2D-list of estimated E_flex (1st index = index in amplitudes, 2nd index = index in frequencies).
    """
    def __init__(self, res_id: str, res_id_exception_for_A_03_omega_1_16: str, re: float, amplitudes: List[float],
                 frequencies: List[float], tasks: List[List[Union[int, List[int]]]], s: List[List[List[float]]],
                 e_a: List[List[List[float]]], e_flex: List[List[List[float]]]):
        self.res_id = res_id
        self.res_id_exception_for_A_03_omega_1_16 = res_id_exception_for_A_03_omega_1_16
        self.re = re
        self.amplitudes = amplitudes
        self.frequencies = frequencies
        self.tasks = tasks
        self.s = s
        self.e_a = e_a
        self.e_flex = e_flex


class SimulationsInfo:
    """
    Class SimulationsInfo represents a summary data from the campaign of simulations for different amplitudes and
    frequencies of in-phase spanwise wall oscillations:

    res_id
      Research ID associated with the tasks containing the relevant simulations

    re
      Reynolds number

    amplitudes
      A list of amplitudes

    frequencies
      A list of frequencies

    tasks
      A 2D-list of task numbers containing the relevant simulations (1st index = 1st index in amplitudes, 2nd
      index = 2nd index in frequencies). If task number is -1, then there is no data for the corresponding
      combination of amplitude and frequency

    task_for_uncontrolled_case
      A task for the uncontrolled case
    """
    def __init__(self, res_id: str, re: float, amplitudes: List[float], frequencies: List[float], tasks: List[List[int]],
                 task_for_uncontrolled_case: int):
        self.res_id = res_id
        self.re = re
        self.amplitudes = amplitudes
        self.frequencies = frequencies
        self.tasks = tasks
        self.task_for_uncontrolled_case = task_for_uncontrolled_case


class MinimalSeedInfo:
    """
    Class MinimalInfo represents a summary data from the minimal seed calculations for different amplitudes and
    frequencies of in-phase spanwise wall oscillations:

    re
      Reynolds number

    amplitudes
      A list of amplitudes

    frequencies
      A list of frequencies

    kinetic_energies
      A 2D-list of kinetic energies of minimal seeds (1st index = 1st index in amplitudes, 2nd
      index = 2nd index in frequencies).

    kinetic_energy_for_uncontrolled_case
      Minimal seed KE for the uncontrolled case
    """
    def __init__(self, re: float, amplitudes: List[float], frequencies: List[float], kinetic_energies: List[List[float]],
                 kinetic_energy_for_uncontrolled_case: float):
        self.re = re
        self.amplitudes = amplitudes
        self.frequencies = frequencies
        self.kinetic_energies = kinetic_energies
        self.kinetic_energy_for_uncontrolled_case = kinetic_energy_for_uncontrolled_case


class Summary(JsonSerializable):
    """
    Class Summary is a json-serializable summary of the study.

    edge_tracking_simulations
      SimulationsInfo object with edge tracking simulations

    simulations_with_full_fields_saved
      SimulationsInfo object with simulations where full flow fields were saved with small time intervals (dT = 1 or 10)

    p_lam_info
      LaminarisationProbabilityInfo object with simulations associated with the estimation of the laminarisation
      probability
    """

    def __init__(self, edge_states_info: SimulationsInfo, minimal_seed_info: MinimalSeedInfo,
                 simulations_with_full_fields_saved: SimulationsInfo,
                 p_lam_info: LaminarisationProbabilityInfo, seed: int, seed_for_bayesian_example: int,
                 default_sample_number: int, sample_size_per_energy_level: int,
                 minimum_sample_size_per_energy_level: int):
        self.edge_states_info = edge_states_info
        self.minimal_seed_info = minimal_seed_info
        self.simulations_with_full_fields_saved = simulations_with_full_fields_saved
        self.p_lam_info = p_lam_info
        self.seed = seed
        self.seed_for_bayesian_example = seed_for_bayesian_example
        self.default_sample_number = default_sample_number
        self.sample_size_per_energy_level = sample_size_per_energy_level
        self.minimum_sample_size_per_energy_level = minimum_sample_size_per_energy_level
