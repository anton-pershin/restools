from typing import List

from jsons import JsonSerializable


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


class Summary(JsonSerializable):
    """
    Class Summary is a json-serializable summary of the study.

    edge_tracking_simulations
      SimulationsInfo object with edge tracking simulations

    simulations_with_full_fields_saved
      SimulationsInfo object with simulations where full flow fields were saved with small time intervals (dT = 1 or 10)
    """

    def __init__(self, edge_states_info: SimulationsInfo, simulations_with_full_fields_saved: SimulationsInfo):
        self.edge_states_info = edge_states_info
        self.simulations_with_full_fields_saved = simulations_with_full_fields_saved
