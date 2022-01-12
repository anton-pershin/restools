from typing import List, Union

from jsons import JsonSerializable


class ReducedPrecisionVersion(JsonSerializable):
    def __init__(self, name: str, descr: str):
        self.name = name
        self.descr = descr


class Summary(JsonSerializable):
    """
    Class Summary is a json-serializable summary of the study.

    All the outputs for a particular experiment can be found in a directory whose path is
    oxford_output_path + solver + number_of_bits + '_' + ReducedPrecisionVersion.name

    edge_tracking_simulations
      SimulationsInfo object with edge tracking simulations

    simulations_with_full_fields_saved
      SimulationsInfo object with simulations where full flow fields were saved with small time intervals (dT = 1 or 10)

    p_lam_info
      LaminarisationProbabilityInfo object with simulations associated with the estimation of the laminarisation
      probability
    """

    def __init__(self, res_id: str, task_for_oifs_results: int, task_for_era5_data:int, l91_file: str, l137_file: str, oxford_input_path: str, oxford_output_path: str,
                 solvers: List[str], rp_versions: List[ReducedPrecisionVersion]):
        self.res_id = res_id
        self.task_for_oifs_results = task_for_oifs_results
        self.task_for_era5_data = task_for_era5_data
        self.l91_file = l91_file
        self.l137_file = l137_file
        self.oxford_input_path = oxford_input_path
        self.oxford_output_path = oxford_output_path
        self.solvers = solvers
        self.rp_versions = rp_versions
