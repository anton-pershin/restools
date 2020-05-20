from typing import List

from jsons import JsonSerializable


class TurbTrajectory:
    """
    Class TurbTrajectory represents a summary data about a typical turbulent trajectory:

    task
      A task number where the trajectory should be sought

    data_dir
      A directory in task in which TimeIntegration data can be found
    """
    def __init__(self, task: int, data_dir: str):
        self.task = task
        self.data_dir = data_dir


class RPInfo:
    """
    Class RPInfo represents a summary data about a single random perturbation:

    A
      Coefficient multiplied on the random component orthogonal to the laminar solution

    B
      Coefficient multiplied on the laminar solution

    is_laminarised
      A laminarising (1) or transitioning (0) class of RP
    """
    def __init__(self, A: float, B: float, is_laminarised: int):
        self.A = A
        self.B = B
        self.is_laminarised = is_laminarised


class SingleConfiguration:
    """
    Class SingleConfiguration represents a summary data for a single set of parameters (Re, control parameter, etc.):

    description
      Short description of the configuration being used

    res_id
      Research ID associated with the tasks containing the relevant simulations

    tasks
      A list of task numbers containing the relevant simulations

    preprocess_task
      A task number where preprocessed data is stored

    dump_filename
      A name of the file with dumped data

    turb_trajectory
      TurbTrajectory instance

    rps_info
      A 2D-list of RPInfo instances (first index = energy level id)

    p_lam
      A list of laminarisation probabilities indexed by energy level id

    edge_state_energy_mean
      Mean of the edge state kinetic energy

    edge_state_energy_std
      Standard deviation of the edge state kinetic energy
    """
    def __init__(self, description: str, res_id: str, tasks: List[int], preprocess_task: int, dump_filename: str,
                 turb_trajectory: TurbTrajectory, rps_info: List[List[RPInfo]], p_lam: List[float],
                 edge_state_energy_mean: float, edge_state_energy_std: float):
        self.description = description
        self.res_id = res_id
        self.tasks = tasks
        self.preprocess_task = preprocess_task
        self.dump_filename = dump_filename
        self.turb_trajectory = turb_trajectory
        self.rps_info = rps_info
        self.p_lam = p_lam
        self.edge_state_energy_mean = edge_state_energy_mean
        self.edge_state_energy_std = edge_state_energy_std


class Summary(JsonSerializable):
    """
    Class Summary is a json-serializable summary of the study.

    confs
      A list of SingleConfiguration instances

    energy_levels
      A list of energy levels

    energy_deviations
      A list of deviations of the energy levels which should be taken into account because of an initial adjustment of
      random perturbations

    laminar_flow_ke
      Kinetic energy of the laminar solution of plane Couette flow
    """

    def __init__(self, confs: List[SingleConfiguration], energy_levels: List[float], energy_deviations: List[float],
                 laminar_flow_ke: float):
        self.confs = confs
        self.energy_levels = energy_levels
        self.energy_deviations = energy_deviations
        self.laminar_flow_ke = laminar_flow_ke

