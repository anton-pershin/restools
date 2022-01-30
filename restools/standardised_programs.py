import os
import shutil
from abc import ABC, abstractmethod

from restools.timeintegration import TimeIntegrationChannelFlowV1, TimeIntegrationChannelFlowV2, \
    TimeIntegrationLowDimensional
from comsdk.communication import BaseCommunication
from comsdk.graph import Func
from comsdk.edge import Edge, ExecutableProgramEdge, dummy_predicate, dummy_edge, InOutMapping
import comsdk.comaux as comaux


class StandardisedProgram:
    def __init__(self, name: str, keyword_names=(), trailing_args_keys=(),
                 chaining_command_at_start='', chaining_command_at_end=''):
        self.name = name
        self.keyword_names = keyword_names
        self.trailing_args_keys = trailing_args_keys
        self.chaining_command_at_start = chaining_command_at_start
        self.chaining_command_at_end = chaining_command_at_end


class StandardisedIntegrator(StandardisedProgram, ABC):
    def __init__(self, name: str, keyword_names=(), trailing_args_keys=(),
                 chaining_command_at_start='', chaining_command_at_end=''):
        StandardisedProgram.__init__(self, name, keyword_names=keyword_names, trailing_args_keys=trailing_args_keys,
                                     chaining_command_at_start=chaining_command_at_start,
                                     chaining_command_at_end=chaining_command_at_end)

    @property
    @classmethod
    @abstractmethod
    def ti_class(cls):
        raise NotImplementedError('Derived class must set the class member ti_class as a class'
                                  'derived from TimeIntegration')

    @classmethod
    @abstractmethod
    def concatenate_integration_piece(cls, d, input_datas_key='integration_subdir', output_data_key=None):
        raise NotImplementedError('Derived class must implement a class method allowing for the concatenation of '
                                  'several pieces of time-integration (i.e., several ordered data directories) into a '
                                  'single simulation')

    def postprocessor_edge(self, comm: BaseCommunication, predicate: Func = dummy_predicate,
                           io_mapping: InOutMapping = InOutMapping()):
        return dummy_edge


class StandardisedProgramEdge(ExecutableProgramEdge):
    def __init__(self, prog: StandardisedProgram, comm, relative_keys=(), keys_mapping={}, io_mapping=None,
                 output_dict={}, flag_names=(), remote=False, stdout_processor=None):
        io_mapping_ = io_mapping if io_mapping is not None else InOutMapping(relative_keys=relative_keys, keys_mapping=keys_mapping)
        super().__init__(prog.name, comm,
                         io_mapping=io_mapping_,
                         keyword_names=prog.keyword_names,
                         trailing_args_keys=prog.trailing_args_keys,
                         output_dict=output_dict,
                         flag_names=flag_names,
                         remote=remote,
                         stdout_processor=stdout_processor,
                         chaining_command_at_start=prog.chaining_command_at_start,
                         chaining_command_at_end=prog.chaining_command_at_end,
                         )


class CouetteChannelflowV1(StandardisedIntegrator):
    ti_class = TimeIntegrationChannelFlowV1

    def __init__(self, ic_filename_key='initial_condition'):
        super().__init__(name='couette',
                         keyword_names=('R', 'T0', 'T1', 'dt', 'dT', 'dPT', 'is', 'A', 'omega', 'phi', 'el', 'et', 'o', 'ke'),
                         trailing_args_keys=(ic_filename_key,))

    @classmethod
    def concatenate_integration_piece(cls, d, input_datas_key='integration_subdir', output_data_key=None):
        # I. Concatenate *.txt files
        filenames_to_concat = [
            'av_u.txt',
            'av_v.txt',
            'avenergy.txt',
            'summary.txt',
            'z.txt',
        ]

        if 'omega' in d:
            filenames_to_concat.append('wbase_t.txt')

        all_data_path = d['__WORKING_DIR__'] if input_datas_key is None else os.path.join(d['__WORKING_DIR__'], d[input_datas_key])
        result_data_path = all_data_path if output_data_key is None else os.path.join(d['__WORKING_DIR__'], d[output_data_key])
        if not os.path.exists(result_data_path):
            os.mkdir(result_data_path)

        data_paths = [os.path.join(all_data_path, '{}-{}'.format(d[output_data_key], i) if output_data_key is not None else 'data-{}'.format(i)) for i in range(1, d['i'] + 1)]
        get_time_unit = lambda str_: float(str_.split()[0])
        for filename_to_concat in filenames_to_concat:
            lines = []
            time_unit_step = 0.5
            last_time_unit = 0
            print('Concatenating {}...'.format(filename_to_concat))
            for data_path in data_paths:
                f = open(os.path.join(data_path, filename_to_concat), 'r')
                current_file_lines = f.readlines()
                if len(lines) == 0:
                    lines.append(current_file_lines[0])
                current_start_time_unit = get_time_unit(current_file_lines[1])
                print('\tCopy starting from t={} to t={}'.format(last_time_unit, get_time_unit(current_file_lines[-1])))
                start_index = 1 + int((last_time_unit - current_start_time_unit) / time_unit_step)  # the 1st line is description. The last line may be incomplete
                lines += current_file_lines[start_index:]
                last_time_unit = get_time_unit(lines[-1])
            f = open(os.path.join(result_data_path, filename_to_concat), 'w')
            f.writelines(lines)

        # II. Move *.h5 files and delete temporary data dirs
        for data_path in data_paths:
            filenames_and_params = \
                comaux.find_all_files_by_standardised_naming(cls.ti_class.solution_standardised_filename, data_path)
            files = [pair[0] for pair in filenames_and_params]
            #files = get_all_files_by_extension(data_path, 'h5')
            for file in files:
                if not os.path.exists(os.path.join(result_data_path, file)):
                    shutil.move(os.path.join(data_path, file), result_data_path)
            shutil.rmtree(data_path)


class SimulateflowChannelflowV2(StandardisedIntegrator):
    ti_class = TimeIntegrationChannelFlowV2

    def __init__(self, ic_filename_key='initial_condition'):
        super().__init__(name='simulateflow',
                         keyword_names=('R', 'T0', 'T', 'dt', 'vdt', 'dT', 's', 'o', 'e'),
                         trailing_args_keys=(ic_filename_key,))

    def postprocessor_edge(self, comm: BaseCommunication, predicate: Func = dummy_predicate,
                           io_mapping: InOutMapping = InOutMapping()):
        def glue_ke_z_measurements(d):
            measurements_dir = 'xyavg_energy'
            data_path = os.path.join(d['__REMOTE_WORKING_DIR__'], d['o'])
            measurements_path = os.path.join(data_path, measurements_dir)
            comm.execute(comm.host.commands['ncecat'] + ' ../ke_z.nc', measurements_path)
            comm.execute('rm -r {}'.format(measurements_dir), data_path)

        return Edge(predicate, Func(func=glue_ke_z_measurements), io_mapping=io_mapping)

    @classmethod
    def concatenate_integration_piece(cls, d, input_datas_key='integration_subdir', output_data_key=None):
        pass


class MoehlisModelIntegrator(StandardisedIntegrator):
    ti_class = TimeIntegrationLowDimensional
    def __init__(self, input_filename_key='input_filename', nohup=False):
        chaining_command_at_start = ''
        chaining_command_at_end = ''
        if nohup:
            chaining_command_at_start, chaining_command_at_end = nohup_command_start_and_end()
        super().__init__(name='time_integrate_moehlis.py',
                         keyword_names=('cores',),
                         trailing_args_keys=(input_filename_key,),
                         chaining_command_at_start=chaining_command_at_start,
                         chaining_command_at_end=chaining_command_at_end
                         )

    def postprocessor_edge(self, comm: BaseCommunication, predicate: Func = dummy_predicate,
                           io_mapping: InOutMapping = InOutMapping()):
        pass

    @classmethod
    def concatenate_integration_piece(cls, d, input_datas_key='integration_subdir', output_data_key=None):
        pass


class EsnIntegrator(StandardisedIntegrator):
    ti_class = TimeIntegrationLowDimensional

    def __init__(self, input_filename_key='input_filename', nohup=False):
        chaining_command_at_start = ''
        chaining_command_at_end = ''
        if nohup:
            chaining_command_at_start, chaining_command_at_end = nohup_command_start_and_end()
        super().__init__(name='time_integrate_esn.py',
                         keyword_names=('cores',),
                         trailing_args_keys=(input_filename_key,),
                         chaining_command_at_start=chaining_command_at_start,
                         chaining_command_at_end=chaining_command_at_end
                         )

    def postprocessor_edge(self, comm: BaseCommunication, predicate: Func = dummy_predicate,
                           io_mapping: InOutMapping = InOutMapping()):
        pass

    @classmethod
    def concatenate_integration_piece(cls, d, input_datas_key='integration_subdir', output_data_key=None):
        pass


class EsnTrainer(StandardisedIntegrator):
    ti_class = TimeIntegrationLowDimensional

    def __init__(self, input_filename_key='input_filename', nohup=False):
        chaining_command_at_start = ''
        chaining_command_at_end = ''
        if nohup:
            chaining_command_at_start, chaining_command_at_end = nohup_command_start_and_end()
        super().__init__(name='train_esn.py',
                         keyword_names=('cores',),
                         trailing_args_keys=(input_filename_key,),
                         chaining_command_at_start=chaining_command_at_start,
                         chaining_command_at_end=chaining_command_at_end
                         )

    def postprocessor_edge(self, comm: BaseCommunication, predicate: Func = dummy_predicate,
                           io_mapping: InOutMapping = InOutMapping()):
        pass

    @classmethod
    def concatenate_integration_piece(cls, d, input_datas_key='integration_subdir', output_data_key=None):
        pass


class RandomfieldChannelflowV1(StandardisedProgram):
    def __init__(self, output_filename_key='initial_condition'):
        super().__init__(name='randomfield',
                         keyword_names=('Nx', 'Ny', 'Nz', 'lx', 'lz', 'Lx', 'Lz', 'sd', 'm', 'symms'),
                         trailing_args_keys=(output_filename_key,))


class AddfieldsChannelflowV1(StandardisedProgram):
    def __init__(self, params_key, output_filename_key='initial_condition'):
        """
        :param params_key: key where a sequence of arguments like a1 file1 a2 file2 ... is stored. Here file1 will be
                           multiplied by a1 and then added to a2*file2 etc.
        :param output_filename_key: key where an output filename is stored
        """
        super().__init__(name='addfields',
                         trailing_args_keys=(params_key, output_filename_key,))


def nohup_command_start_and_end():
    if os.name == 'posix':
        start = r'nohup'
        end = r'> task.out 2> task.err < /dev/null &'
    elif os.name == 'nt':
        start = r'start'
        end = r'> task.out 2> task.err'
    else:
        raise ValueError(f'Unsupported "os.name": {os.name}')
    return start, end
