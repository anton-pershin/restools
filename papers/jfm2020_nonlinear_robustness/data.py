from typing import List

from jsons import JsonSerializable


class EdgeStatesInfo:
    """
    Class EdgeStatesInfo represents a summary data from the campaign of edge tracking runs for different amplitudes and
    frequencies of in-phase spanwise wall oscillations:

    res_id
      Research ID associated with the tasks containing the relevant simulations

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
    def __init__(self, res_id: str, amplitudes: List[float], frequencies: List[float], tasks: List[List[int]],
                 task_for_uncontrolled_case: int):
        self.res_id = res_id
        self.amplitudes = amplitudes
        self.frequencies = frequencies
        self.tasks = tasks
        self.task_for_uncontrolled_case = task_for_uncontrolled_case


class Summary(JsonSerializable):
    """
    Class Summary is a json-serializable summary of the study.

    edge_states_info
      EdgeStatesInfo object
    """

    def __init__(self, edge_states_info: EdgeStatesInfo):
        self.edge_states_info = edge_states_info
