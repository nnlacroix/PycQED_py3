import numpy as np
from copy import copy
from copy import deepcopy
from itertools import zip_longest
import traceback
from pycqed.utilities.general import temporary_value
from pycqed.measurement.quantum_experiment import QuantumExperiment
from pycqed.measurement.waveform_control.circuit_builder import CircuitBuilder
from pycqed.measurement.waveform_control.block import Block, ParametricValue
from pycqed.measurement.waveform_control.segment import UnresolvedPulse
from pycqed.measurement.sweep_points import SweepPoints
from pycqed.measurement.calibration.calibration_points import CalibrationPoints
import pycqed.measurement.awg_sweep_functions as awg_swf
import pycqed.analysis_v2.timedomain_analysis as tda
from pycqed.instrument_drivers.meta_instrument.qubit_objects.QuDev_transmon \
    import QuDev_transmon
from pycqed.measurement import multi_qubit_module as mqm
import logging
log = logging.getLogger(__name__)

# TODO: docstrings (list all kw at the highest level with reference to where
#  they are explained, explain all kw where they are processed)
# TODO: add some comments that explain the way the code works


class MultiTaskingExperiment(QuantumExperiment):
    """
    This class adds the concept of tasks to the QuantumExperiment class and
    allows to run multiple tasks in parallel. There are no checks whether a
    parallel execution of tasks makes sense on the used hardware
    (connectivity, crosstalk etc.), i.e., it is up to the experimentalist
    to ensure this by passing a reasonable task_list.

    The concept is that each experiment inherited from this class should
    define a method that creates a block based on the parameters specified
    in a task. The method parallel_sweep can then be used to assemble these
    blocks in parallel.

    :param task_list: list of dicts, where each dict contains the parameters of
        a task (= keyword arguments for the block creation function)
    :param dev: device object, see QuantumExperiment
    :param qubits: list of qubit objects, see QuantumExperiment
    :param operation_dict: operations dictionary, see QuantumExperiment
    :param kw: keyword arguments. Some are processed directly in the init or
        the parent init, some are processed in other functions, e.g.,
        create_cal_points. (FIXME further documentation would help). The
        contents of kw are also stored to metadata and thus be used to pass
        options to the analysis.
    """

    # The following dictionary can be overwritten by child classes to
    # specify keyword arguments from which sweep_points should be generated
    # automatically (see docstring of generate_kw_sweep_points).
    kw_for_sweep_points = {}
    # The following list can be overwritten by child classes to specify keyword
    # arguments that should be automatically copied into each task (list of
    # str, each being a key to be searched in kw).
    kw_for_task_keys = ()

    def __init__(self, task_list, dev=None, qubits=None,
                 operation_dict=None, **kw):

        self.task_list = task_list
        # Process kw_for_sweep_points for the global keyword arguments kw
        self.generate_kw_sweep_points(kw)

        # Try to get qubits or at least qb_names
        _, qb_names = self.extract_qubits(dev, qubits, operation_dict)
        # Filter to the ones that are needed
        qb_names = self.find_qubits_in_tasks(qb_names, task_list)
        # Initialize the QuantumExperiment
        super().__init__(dev=dev, qubits=qubits,
                         operation_dict=operation_dict,
                         filter_qb_names=qb_names, **kw)

        if 'sweep_points' in kw:
            # Note that sweep points generated due to kw_for_sweep_points are
            # already part of kw['sweep_points'] at this point.
            self.sweep_points = kw.pop('sweep_points')

        self.cal_points = None
        self.cal_states = None
        self.exception = None
        self.all_main_blocks = []
        self.data_to_fit = {}

        # The following is done because the respective call in the init of
        # QuantumExperiment does not capture all kw since many are explicit
        # arguments of the init there.
        kw.pop('exp_metadata', None)
        self.exp_metadata.update(kw)

        # Create calibration points based on settings in kw (see docsring of
        # create_cal_points)
        self.create_cal_points(**kw)

    def add_to_meas_obj_sweep_points_map(self, meas_objs, sweep_point):
        """
        Add an entry to the meas_obj_sweep_points_map, which will later be
        stored to the metadata. Makes sure to not add entries twice.
        :param meas_objs: (str or list of str) name(s) of the measure
            object(s) for which the sweep_point should be added
        :param sweep_point: (str) name of the sweep_point that should be added
        """
        if 'meas_obj_sweep_points_map' not in self.exp_metadata:
            self.exp_metadata['meas_obj_sweep_points_map'] = {}
        if not isinstance(meas_objs, list):
            meas_objs = [meas_objs]
        for mo in meas_objs:
            # get name from object if an object was given
            mo = mo if isinstance(mo, str) else mo.name
            if mo not in self.exp_metadata['meas_obj_sweep_points_map']:
                self.exp_metadata['meas_obj_sweep_points_map'][mo] = []
            if sweep_point not in self.exp_metadata[
                    'meas_obj_sweep_points_map'][mo]:
                # if the entry does not exist yet
                self.exp_metadata['meas_obj_sweep_points_map'][mo].append(
                    sweep_point)

    def get_meas_objs_from_task(self, task):
        """
        Returns a list of all measure objects (e.g., qubits) of a task.
        Should be overloaded in child classes if the default behavior
        of returning all qubits found in the task is not desired.
        :param task: a task dictionary
        :return: list of a qubit objects (if available) or names
        """
        return self.find_qubits_in_tasks(self.qb_names, [task])

    def run_measurement(self, **kw):
        """
        Run the actual measurement. Stores some additional settings and
            then calls the respective method in QuantumExperiment.
        :param kw: keyword arguments
        """
        # allow the user to overwrite the automatically generated list of
        # channels to upload
        self.channels_to_upload = kw.get('channels_to_upload',
                                         self.channels_to_upload)
        # update the nr_averages based on the settings in the user measure
        # objects
        self.df_kwargs.update(
            {'nr_averages': max(qb.acq_averages() for qb in self.meas_objs)})

        # Store metadata that is not part of QuantumExperiment.
        self.exp_metadata.update({
            'preparation_params': self.get_prep_params(),
            'rotate': len(self.cal_states) != 0 and not self.classified,
            'sweep_points': self.sweep_points,
            'ro_qubits': self.meas_obj_names,
            'data_to_fit': self.data_to_fit,
        })
        if self.task_list is not None:
            self.exp_metadata.update({'task_list': self.task_list})

        super().run_measurement(**kw)

    def create_cal_points(self, n_cal_points_per_state=1, cal_states='auto',
                          for_ef=False, **kw):
        """
        Creates a CalibrationPoints object based on the given parameters and
            saves it to self.cal_points.

        :param n_cal_points_per_state: number of segments for each
            calibration state
        :param cal_states: str or tuple of str; the calibration states
            to measure
        :param for_ef: bool indicating whether to measure the |f> calibration
            state for each qubit
        :param kw: keyword arguments (to allow pass through kw even if it
            contains entries that are not needed)
        """
        self.cal_states = CalibrationPoints.guess_cal_states(
            cal_states, for_ef=for_ef)
        self.cal_points = CalibrationPoints.multi_qubit(
            self.meas_obj_names, self.cal_states,
            n_per_state=n_cal_points_per_state)
        self.exp_metadata.update({'cal_points': repr(self.cal_points)})

    def preprocess_task_list(self, **kw):
        """
        Calls preprocess task for all tasks in self.task_list. This adds
        prefixed sweep points to self.sweep_points and returns a
        preprocessed task list, for details see preprocess_task.

        :param kw: keyword arguments
        :return: the preprocessed task list
        """
        # keep a reference to the original sweep_points object
        given_sweep_points = self.sweep_points
        # Store a copy of the sweep_points (after ensuring that they are a
        # SweepPoints object). This copy will then be extended with prefixed
        # task-specific sweep_points.
        self.sweep_points = SweepPoints(from_dict_list=given_sweep_points)
        # Internally, 1D and 2D sweeps are handled as 2D sweeps.
        while len(self.sweep_points) < 2:
            self.sweep_points.add_sweep_dimension()
        preprocessed_task_list = []
        for task in self.task_list:
            # preprocessed_task_list requires both the sweep point that
            # should be modified and the original version of the sweep
            # points (to see which sweep points are valid for all tasks)
            preprocessed_task_list.append(
                self.preprocess_task(task, self.sweep_points,
                                     given_sweep_points, **kw))
        return preprocessed_task_list

    def preprocess_task(self, task, global_sweep_points, sweep_points=None,
                        **kw):
        """
        Preprocesses a task, which includes the following actions. The
        original task is not modified, but instead a new, preprocessed task
        is returned.
        - Create or cleanup task prefix.
        - Copy kwargs listed in kw_for_task_keys to the task.
        - Generate task-specific sweep points based on generate_kw_sweep_points
          if the respective keys are found as parameters of the task.
        - Copies sweep points valid for all tasks to the task.
        - Adds prefixed versions of task-specific sweep points to the global
          sweep points
        - Generate a list of sweep points whose names have to be prefixed
          when used as ParametricValue during block creation.
        - Update meas_obj_sweep_points_map for qubits involved in the task

        :param task: (dict) the task
        :param global_sweep_points: (SweepPoints object) global sweep points
            containing the sweep points valid for all tasks plus prefixed
            versions of task-specific sweep points. The object is updated
            by this method.
        :param sweep_points: (SweepPoints object or list of dicts or None)
            sweep points valid for all tasks. Remains unchanged in this method.
        :param kw: keyword arguments
        :return: the preprocessed task
        """
        # copy the task in order to not modify the original task
        task = copy(task)  # no deepcopy: might contain qubit objects
        # Create a prefix if it does not exist. Otherwise clean it up (add "_")
        prefix = task.get('prefix', None)
        if prefix is None:  # try to guess one based on contained qubits
            prefix = '_'.join(self.find_qubits_in_tasks(self.qb_names, [task]))
        prefix += ('_' if prefix[-1] != '_' else '')
        task['prefix'] = prefix
        # Get measure objects needed involved in this task. Will be used
        # below to generate entries for the meas_obj_sweep_points_map.
        mo = self.get_meas_objs_from_task(task)

        # Copy kwargs listed in kw_for_task_keys to the task.
        for param in self.kw_for_task_keys:
            if param not in task:
                task[param] = kw.get(param, None)

        # Start with sweep points valid for all tasks
        current_sweep_points = SweepPoints(from_dict_list=sweep_points)
        # generate kw sweep points for the task
        self.generate_kw_sweep_points(task)
        # Add all task sweep points to the current_sweep_points object.
        # If a task-specific sweep point has the same name as a sweep point
        # valid for all tasks, the task-specific one is used for this task.
        current_sweep_points.update(
            SweepPoints(from_dict_list=task['sweep_points']))
        # Create a list of lists containing for each dimension the names of
        # the task-specific sweep points. These sweep points have to be
        # prefixed with the task prefix later on (in the global sweep
        # points, see below, and when used as ParametricValue during block
        # creation).
        params_to_prefix = [d.keys() for d in task['sweep_points']]
        task['params_to_prefix'] = params_to_prefix
        # Save the current_sweep_points object to the preprocessed task
        task['sweep_points'] = current_sweep_points

        # Internally, 1D and 2D sweeps are handled as 2D sweeps.
        while len(current_sweep_points) < 2:
            current_sweep_points.add_sweep_dimension()
        while len(params_to_prefix) < 2:
            params_to_prefix.append([])
        # for all sweep dimensions
        for gsp, csp, params in zip(global_sweep_points,
                                    current_sweep_points,
                                    params_to_prefix):
            # for all sweep points in this dimension (both task-specific and
            # valid for all tasks)
            for k in csp.keys():
                if k in params:
                    # task-specific sweep point. Add prefixed version to
                    # global sweep points and to meas_obj_sweep_points_map
                    gsp[prefix + k] = csp[k]
                    self.add_to_meas_obj_sweep_points_map(mo, prefix + k)
                else:
                    # sweep point valid for all tasks. Add without prefix to
                    # meas_obj_sweep_points_map
                    self.add_to_meas_obj_sweep_points_map(mo, k)
        return task

    def parallel_sweep(self, preprocessed_task_list=(), block_func=None,
                       block_align=None, **kw):
        """
        Calls a block creation function for each task in a task list,
        puts these blocks in parallel and sweeps over the given sweep points.

        :param preprocessed_task_list: a list of dictionaries, each containing
            keyword arguments for block_func, plus a key 'prefix' with a unique
            prefix string, plus optionally a key 'params_to_prefix' created
            by preprocess_task indicating which sweep parameters have to be
            prefixed with the task prefix.
        :param block_func: a handle to a function that creates a block. As
            an alternative, a task-specific block_func can be given as a
            parameter of the task. If the block creation function instead
            returns a list of blocks, the i-th blocks of all tasks are
            assembled in parallel to each other, and the resulting
            multitask blocks are then assembled sequentially in the order of
            the list index.
        :param block_align: (str or list) alignment of the parallel blocks, see
            CircuitBuilder.simultaneous_blocks, default: center. If the block
            creation function creates a list of N blocks, block_align can be
            a list of N strings (otherwise the same alignment is used for
            all parallel blocks).
        :param kw: keyword arguments are passed to sweep_n_dim
        :return: see sweep_n_dim
        """
        parallel_blocks = []
        for task in preprocessed_task_list:
            # copy the task in order to not modify the original task
            task = copy(task)  # no deepcopy: might contain qubit objects
            # pop prefix and params_to_prefix since they are not needed by
            # the block creation function
            prefix = task.pop('prefix')
            params_to_prefix = task.pop('params_to_prefix', None)
            # the block_func passed as argument is used for all tasks that
            # do not define their own block_func
            if not 'block_func' in task:
                task['block_func'] = block_func
            # Call the block creation function. The items in the task dict
            # are used as kwargs for this function.
            new_block = task['block_func'](**task)
            # If a single block was returned, create a single-entry list to
            # have a unified treatment afterwards.
            if not isinstance(new_block, list):
                new_block = [new_block]
            for b in new_block:
                # prefix the block names to avoid naming conflicts later on
                b.name = prefix + b.name
                # For the sweep points that need to be prefixed (see
                # preprocess_task), call the respective method of the block
                # object.
                if params_to_prefix is not None:
                    # params_to_prefix is a list of lists (per dimension) and
                    # needs to be flattened
                    b.prefix_parametric_values(
                        prefix, [k for l in params_to_prefix for k in l])
            # add the new blocks to the lists of blocks
            parallel_blocks.append(new_block)

        # We currently require that all block functions must return the
        # same number of blocks.
        if not isinstance(block_align, list):
            block_align = [block_align] * len(parallel_blocks[0])
        # assemble all i-th blocks in parallel
        self.all_main_blocks = [
            self.simultaneous_blocks(
                f'all{i}', [l[i] for l in parallel_blocks],
                block_align=block_align[i])
            for i in range(len(parallel_blocks[0]))]
        if len(parallel_blocks[0]) > 1:
            # assemble the multitask blocks sequentially
            self.all_main_blocks = self.sequential_blocks(
                'all', self.all_main_blocks)
        else:
            self.all_main_blocks = self.all_main_blocks[0]
        if len(self.sweep_points[1]) == 0:
            # Internally, 1D and 2D sweeps are handled as 2D sweeps.
            # With this dummy soft sweep, exactly one sequence will be created
            # and the data format will be the same as for a true soft sweep.
            self.sweep_points.add_sweep_parameter('dummy_sweep_param', [0])
        # ro_qubits in kw determines for which qubits sweep_n_dim will add
        # readout pulses. If it is not provided (which is usually the case
        # since create_meas_objs_list pops it from kw) all qubits in
        # meas_obj_names will be used, except those for which there are
        # already readout pulses in the parallel blocks.
        if 'ro_qubits' not in kw:
            op_codes = [p['op_code'] for p in self.all_main_blocks.pulses if
                        'op_code' in p]
            kw = copy(kw)
            kw['ro_qubits'] = [m for m in self.meas_obj_names if f'RO {m}'
                               not in op_codes]
        # call sweep_n_dim to perform the actual sweep
        return self.sweep_n_dim(self.sweep_points,
                                body_block=self.all_main_blocks,
                                cal_points=self.cal_points, **kw)

    @staticmethod
    def find_qubits_in_tasks(qubits, task_list, search_in_operations=True):
        """
        Searches for qubit objects and all mentions of qubit names in the
        provided tasks.
        :param qubits: (list of str or objects) list of qubits whose mentions
            should be searched
        :param task_list: (list of dicts) list of tasks in which qubit objects
            and mentions of qubit names should be searched.
        :param search_in_operations: (bool) whether qubits should also be
            searched inside op_codes, default: True
        :return: list of a qubit object for each found qubit (if objects are
            available, otherwise list of qubit names)
        """
        # This dict maps from qubit names to qubit object if qubit objects
        # are available. Otherwise it is a trivial map from qubit names to
        # qubit names.
        qbs_dict = {qb if isinstance(qb, str) else qb.name: qb for qb in
                    qubits}
        found_qubits = []

        # helper function that checks candiates and calls itself recursively
        # if a candidate is a list
        def append_qbs(found_qubits, candidate):
            if isinstance(candidate, QuDev_transmon):
                if candidate not in found_qubits:
                    found_qubits.append(candidate)
            elif isinstance(candidate, str):
                if candidate in qbs_dict.keys():
                    # it is a mention of a qubit
                    if qbs_dict[candidate] not in found_qubits:
                        found_qubits.append(qbs_dict[candidate])
                elif ' ' in candidate and search_in_operations:
                    # If it contains spaces, it could be an op_code. To
                    # search in operations, we just split the potential op_code
                    # at the spaces and search again in the resulting list
                    append_qbs(found_qubits, candidate.split(' '))
            elif isinstance(candidate, list):
                # search inside each list element
                for v in candidate:
                    append_qbs(found_qubits, v)
            else:
                return None

        # search in all tasks
        for task in task_list:
            # search in all parameters of the task
            for v in task.values():
                append_qbs(found_qubits, v)
        return found_qubits

    def create_meas_objs_list(self, task_list=None, **kw):
        """
        Creates a list of all measure objects used in the measurement. The
        following measure objects are added:
        - qubits listed in kw['ro_qubits']
        - qubits listed in the parameter ro_qubits of each task
        - if kw['ro_qubits'] is not provided and a task does not have a
          parameter ro_qubits: the result of get_meas_objs_from_task for
          this task
        Stores two lists:
        - self.meas_objs: list of measure objects (None if not available)
        - self.meas_obj_names: and list of measure object names

        :param task_list: (list of dicts) the task list
        :param kw: keyword arguments
        """
        if task_list is None:
            task_list = self.task_list
        if task_list is None:
            task_list = [{}]
        # We can pop ro_qubits: if parallel_sweep does not find ro_qubits in
        # kw, it uses self.meas_obj_names, which we generate here
        ro_qubits = kw.pop('ro_qubits', None)
        if ro_qubits is None:
            # Combine for all tasks, fall back to get_meas_objs_from_task if
            # ro_qubits does not exist in a task.
            ro_qubits = [qb for task in task_list for qb in task.pop(
                'ro_qubits', self.get_meas_objs_from_task(task))]
        else:
            # Add ro_qubits from all tasks without falling back to
            # get_meas_objs_from_task.
            ro_qubits += [qb for task in task_list for qb in
                          task.pop('ro_qubits', [])]
        # Unique and sort. To make this possible, convert to str.
        ro_qubits = [qb if isinstance(qb, str) else qb.name for qb in
                     ro_qubits]
        ro_qubits = list(np.unique(ro_qubits))
        ro_qubits.sort()
        # Get the objects again if available, and store the lists.
        self.meas_objs, self.meas_obj_names = self.get_qubits(
            'all' if len(ro_qubits) == 0 else ro_qubits)

    def generate_kw_sweep_points(self, task):
        """
        Generates sweep_points based on task parameters (or kwargs if kw is
        passed instead of a task) according to the specification in the
        property kw_for_sweep_points. The generated sweep points are added to
        sweep_points in the task (or in kw). If needed, sweep_points is
        converted to a SweepPoints object before.

        Format of kw_for_sweep_points: dict with
         - key: key to be searched in kw and in tasks
         - val: dict of kwargs for SweepPoints.add_sweep_parameter, with the
           additional possibility of specifying 'values_func', a lambda
           function that processes the values in kw before using them as
           sweep values
        or a list of such dicts to create multiple sweep points based on a
        single keyword argument.

        :param task: a task dictionary ot the kw dictionary
        """
        # make sure that sweep_points is a SweepPoints object
        task['sweep_points'] = SweepPoints(
            from_dict_list=task.get('sweep_points', None))
        for k, sp_dict_list in self.kw_for_sweep_points.items():
            if isinstance(sp_dict_list, dict):
                sp_dict_list = [sp_dict_list]
            # This loop can create  multiple sweep points based on a single
            # keyword argument.
            for v in sp_dict_list:
                # copy to allow popping the values_func, which should not be
                # passed to SweepPoints.add_sweep_parameter
                v = copy(v)
                values_func = v.pop('values_func', None)
                # if the respective task parameter (or keyword argument) exists
                if k in task and task[k] is not None:
                    if values_func is not None:
                        values = values_func(task[k])
                    elif isinstance(task[k], int):
                        # A single int N as sweep value will be interpreted as
                        # a sweep over N indices.
                        values = np.arange(task[k])
                    else:
                        # Othervise it is assumed that list-like sweep
                        # values are provided.
                        values = task[k]
                    task['sweep_points'].add_sweep_parameter(
                        values=values, **v)


class CalibBuilder(MultiTaskingExperiment):
    """
    This class extends MultiTaskingExperiment with some methods that are
    useful for calibration measurements.

    :param task_list: see MultiTaskingExperiment
    :param kw: kwargs passed to MultiTaskingExperiment, plus in addition:
        update: (bool) whether instrument settings should be updated based on
            analysis results of the calibration measurement, default: False
    """
    def __init__(self, task_list, **kw):
        super().__init__(task_list=task_list, **kw)
        self.update = kw.pop('update', False)

    def max_pulse_length(self, pulse, sweep_points=None,
                         given_pulse_length=None):
        """
        Determines the maximum time duration of a pulse during a sweep,
        where the pulse length could be modified by a sweep parameter.
        Currently, this is implemented only for up to 2-dimensional sweeps.

        :param pulse: a pulse dictionary (which could contain params that
            are ParametricValue objects)
        :param sweep_points: a SweepPoints object describing the sweep
        :param given_pulse_length: overwrites the pulse_length determined by
            the sweep points with the given value (i.e., no actual sweep is
            performed). This is useful to conveniently process a
            user-provided fixed value for the maximum pulse length.
        """
        pulse = copy(pulse)
        # the following parameters are required to create an UnresolvedPulse
        pulse['name'] = 'tmp'
        pulse['element_name'] = 'tmp'

        if given_pulse_length is not None:
            pulse['pulse_length'] = given_pulse_length
            # generate a pulse object to extend the given length with buffer
            # times etc
            p = UnresolvedPulse(pulse)
            return p.pulse_obj.length

        # Even if we only need a single pulse, creating a block allows
        # us to easily perform a sweep.
        b = Block('tmp', [pulse])
        # Clean up sweep points
        sweep_points = deepcopy(sweep_points)
        if sweep_points is None:
            sweep_points = SweepPoints(from_dict_list=[{}, {}])
        while len(sweep_points) < 2:
            sweep_points.add_sweep_dimension()
        for i in range(len(sweep_points)):
            if len(sweep_points[i]) == 0:
                # Make sure that there exists at least a single sweep point
                # that does not overwrite default values of the pulse params.
                sweep_points[i].update({'dummy': ([0], '', 'dummy')})

        # determine number of sweep values per dimension
        nr_sp_list = [len(list(d.values())[0][0]) for d in sweep_points]
        max_length = 0
        for i in range(nr_sp_list[1]):
            for j in range(nr_sp_list[0]):
                # Perform sweep
                pulses = b.build(
                    sweep_dicts_list=sweep_points, sweep_index_list=[j, i])
                # generate a pulse object to extend the pulse length with
                # buffer times etc. The pulse with index 1 is needed because
                # the virtual block start pulse has index 0.
                p = UnresolvedPulse(pulses[1])
                max_length = max(p.pulse_obj.length, max_length)
        return max_length

    def prepend_pulses_block(self, prepend_pulse_dicts):
        """
        Generates a list of prepended pulses to run a calibration under the
        influence of previous operations (e.g.,  charge in the fluxlines).

        :param prepend_pulse_dicts: list of pulse dictionaries,
            each containing the op_code of the desired pulse, plus optional
            pulse parameters to overwrite the default values of the chosen
            pulse.
        :return: block containing the prepended pulses
        """
        prepend_pulses = []
        if prepend_pulse_dicts is not None:
            for i, pp in enumerate(prepend_pulse_dicts):
                # op_code determines which pulse to use
                prepend_pulse = self.get_pulse(pp['op_code'])
                # all other entries in the pulse dict are interpreted as
                # pulse parameters that overwrite the default values
                prepend_pulse.update(pp)
                prepend_pulses += [prepend_pulse]
        return Block('prepend', prepend_pulses)

    @staticmethod
    def add_default_ramsey_sweep_points(sweep_points, **kw):
        """
        Adds phase sweep points for Ramsey-type experiments to the provided
        sweep_points. Assumes that each phase is required twice (to measure a
        comparison between two scenarios, e.g., with flux pulses on and off
        in a dynamic phase measurement).

        :param sweep_points: (SweepPoints object, list of dicts, or None) the
            existing sweep points
        :param kw: keyword arguments
            nr_phases: how many phase sweep points should be added, default: 6.
                If there already exist sweep points in dimension 0, this
                parameter is ignored and the number of phases is adapted to
                the number of existing sweep points.
        :return: sweep_points with the added phase sweep points
        """
        # ensure that sweep_points is a SweepPoints object with at least two
        # dimensions
        sweep_points = SweepPoints(from_dict_list=sweep_points, min_length=2)
        # If there already exist sweep points in dimension 0, this adapt the
        # number of phases to the number of existing sweep points.
        if len(sweep_points[0]) > 0:
            nr_phases = sweep_points.length(0) // 2
        else:
            nr_phases = kw.get('nr_phases', 6)
        # create the phase sweep points (with each phase twice)
        hard_sweep_dict = SweepPoints()
        if 'phase' not in sweep_points[0]:
            hard_sweep_dict.add_sweep_parameter(
                'phase',
                np.tile(np.linspace(0, 2 * np.pi, nr_phases) * 180 / np.pi, 2),
                'deg')
        # add phase sweep points to the existing sweep points (overwriting
        # them if they exist already)
        sweep_points.update(hard_sweep_dict + [{}])
        return sweep_points


class CPhase(CalibBuilder):
    """
    class to measure the leakage and the phase acquired during a flux pulse
    conditioned on the state of another qubit (qbl).
    In this measurement, the phase from two Ramsey type measurements
    on qbr is measured, once with the control qubit in the excited state
    and once in the ground state. The conditional phase is calculated as the
    difference.

    Args:
        FIXME: add further args
        TODO
        :param cz_pulse_name: see CircuitBuilder
        :param n_cal_points_per_state: see CalibBuilder.get_cal_points()
    ...
    """

    def __init__(self, task_list, sweep_points=None, **kw):
        try:
            self.experiment_name = 'CPhase_measurement'
            for task in task_list:
                for k in ['qbl', 'qbr']:
                    if not isinstance(task[k], str):
                        task[k] = task[k].name
                if not 'prefix' in task:
                    task['prefix'] = f"{task['qbl']}{task['qbr']}_"

            kw['for_ef'] = kw.get('for_ef', True)

            super().__init__(task_list, sweep_points=sweep_points, **kw)

            self.cphases = None
            self.population_losses = None
            self.leakage = None
            self.cz_durations = {}
            self.cal_states_rotations = {}

            self.add_default_sweep_points(**kw)
            self.preprocessed_task_list = self.preprocess_task_list(**kw)
            # the block alignments are for: prepended pulses, initial
            # rotations, flux pulse, final rotations
            self.sequences, self.mc_points = self.parallel_sweep(
                self.preprocessed_task_list, self.cphase_block,
                block_align=['center', 'end', 'center', 'start'], **kw)

            self.exp_metadata.update({
                'cz_durations': self.cz_durations,
                'cal_states_rotations': self.cal_states_rotations,
            })

            self.autorun(**kw)
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def add_default_sweep_points(self, **kw):
        self.sweep_points = self.add_default_ramsey_sweep_points(
            self.sweep_points, **kw)
        nr_phases = self.sweep_points.length(0) // 2
        hard_sweep_dict = SweepPoints(
            'pi_pulse_off', [0] * nr_phases + [1] * nr_phases)
        self.sweep_points.update(hard_sweep_dict + [{}])

    def cphase_block(self, sweep_points,
                     qbl, qbr, num_cz_gates=1, max_flux_length=None,
                     prepend_pulse_dicts=None, **kw):
        """
        TODO
        :param cz_pulse_name: task-specific prefix of CZ gates (overwrites
            global choice passed to the class init)
        ...
        """

        hard_sweep_dict, soft_sweep_dict = sweep_points
        assert num_cz_gates % 2 != 0

        pb = self.prepend_pulses_block(prepend_pulse_dicts)

        pulse_modifs = {'all': {'element_name': 'cphase_initial_rots_el'}}
        ir = self.block_from_ops('initial_rots',
                                 [f'X180 {qbl}', f'X90 {qbr}'] +
                                 kw.get('spectator_op_codes', []),
                                 pulse_modifs=pulse_modifs)
        for p in ir.pulses[1:]:
            p['ref_point_new'] = 'end'
        ir.pulses[0]['pulse_off'] = ParametricValue(param='pi_pulse_off')

        fp = self.block_from_ops('flux', [f"{kw.get('cz_pulse_name', 'CZ')} "
                                          f"{qbl} {qbr}"] * num_cz_gates)
        # TODO here, we could do DD pulses (CH by 2020-06-19)
        # FIXME: currently, this assumes that only flux pulse parameters are
        #  swept in the soft sweep. In fact, channels_to_upload should be
        #  determined based on the sweep_points
        for k in ['channel', 'channel2']:
            if k in fp.pulses[0]:
                if fp.pulses[0][k] not in self.channels_to_upload:
                    self.channels_to_upload.append(fp.pulses[0][k])

        for k in soft_sweep_dict:
            for p in fp.pulses:
                p[k] = ParametricValue(k)
        if max_flux_length is not None:
            log.debug(f'max_flux_length = {max_flux_length * 1e9:.2f} ns, '
                      f'set by user')
        max_flux_length = self.max_pulse_length(fp.pulses[0], sweep_points,
                                                max_flux_length)
        w = self.block_from_ops('wait', [])
        w.block_end.update({'pulse_delay': max_flux_length * num_cz_gates})
        fp_w = self.simultaneous_blocks('sim', [fp, w], block_align='center')

        pulse_modifs = {'all': {'element_name': 'cphase_final_rots_el'}}
        fr = self.block_from_ops('final_rots', [f'X180 {qbl}', f'X90s {qbr}'],
                                 pulse_modifs=pulse_modifs)
        fr.set_end_after_all_pulses()
        fr.pulses[0]['pulse_off'] = ParametricValue(param='pi_pulse_off')
        for k in hard_sweep_dict.keys():
            if k != 'pi_pulse_on' and '=' not in k:
                fr.pulses[1][k] = ParametricValue(k)

        self.cz_durations.update({
            fp.pulses[0]['op_code']: fr.pulses[0]['pulse_delay']})
        self.cal_states_rotations.update({qbl: {'g': 0, 'e': 1, 'f': 2},
                                          qbr: {'g': 0, 'e': 1}})
        self.data_to_fit.update({qbl: 'pf', qbr: 'pe'})

        return [pb, ir, fp_w, fr]

    def guess_label(self, **kw):
        predictive_label = kw.pop('predictive_label', False)
        if self.label is None:
            if predictive_label:
                self.label = 'Predictive_' + self.experiment_name
            else:
                self.label = self.experiment_name
            if self.classified:
                self.label += '_classified'
            if 'active' in self.get_prep_params()['preparation_type']:
                self.label += '_reset'
            # if num_cz_gates > 1:
            #     label += f'_{num_cz_gates}_gates'
            for t in self.task_list:
                self.label += f"_{t['qbl']}{t['qbr']}"

    def get_meas_objs_from_task(self, task):
        return [task['qbl'], task['qbr']]

    def run_analysis(self, **kw):
        plot_all_traces = kw.get('plot_all_traces', True)
        plot_all_probs = kw.get('plot_all_probs', True)
        if self.classified:
            channel_map = {qb.name: [vn + ' ' +
                                     qb.instr_uhf() for vn in
                                     qb.int_avg_classif_det.value_names]
                           for qb in self.meas_objs}
        else:
            channel_map = {qb.name: [vn + ' ' +
                                     qb.instr_uhf() for vn in
                                     qb.int_avg_det.value_names]
                           for qb in self.meas_objs}
        self.analysis = tda.CPhaseLeakageAnalysis(
            qb_names=self.qb_names,
            options_dict={'TwoD': True, 'plot_all_traces': plot_all_traces,
                          'plot_all_probs': plot_all_probs,
                          'channel_map': channel_map})
        self.cphases = {}
        self.population_losses = {}
        self.leakage = {}
        for task in self.task_list:
            self.cphases.update({task['prefix'][:-1]: self.analysis.proc_data_dict[
                'analysis_params_dict'][f"cphase_{task['qbr']}"]['val']})
            self.population_losses.update(
                {task['prefix'][:-1]: self.analysis.proc_data_dict[
                    'analysis_params_dict'][
                    f"population_loss_{task['qbr']}"]['val']})
            self.leakage.update(
                {task['prefix'][:-1]: self.analysis.proc_data_dict[
                    'analysis_params_dict'][
                    f"leakage_{task['qbl']}"]['val']})

        return self.cphases, self.population_losses, self.leakage, \
               self.analysis


class DynamicPhase(CalibBuilder):
    def __init__(self, task_list, sweep_points=None, **kw):
        """
        Dynamic Phase Measurement TODO

        TODO kw args:
        simultaneous: (bool) measure all phases simultaneously (not possible if
            phases of both gate qubits should be measured), default: False
        simultaneous_groups: (list of list of qubit objects or names)
            specifies that the phases of all qubits within each sublist can
            be measured simultaneously.
            If simultaneous=False and no simultaneous_groups are specified,
            only one qubit per task will be measured in parallel.
        """

        try:
            self.simultaneous = kw.get('simultaneous', False)
            self.simultaneous_groups = kw.get('simultaneous_groups', None)
            self.reset_phases_before_measurement = kw.get(
                'reset_phases_before_measurement', True)

            self.dynamic_phase_analysis = {}
            self.dyn_phases = {}
            self.old_dyn_phases = {}
            for task in task_list:
                if task.get('qubits_to_measure', None) is None:
                    task['qubits_to_measure'] = task['op_code'].split(' ')[1:]
                else:
                    # copy to not modify the caller's list
                    task['qubits_to_measure'] = copy(task['qubits_to_measure'])

                for k, v in enumerate(task['qubits_to_measure']):
                    if not isinstance(v, str):
                        task['qubits_to_measure'][k] = v.name

                if 'prefix' not in task:
                    task['prefix'] = task['op_code'].replace(' ', '')

            qbm_all = [task['qubits_to_measure'] for task in task_list]
            if not self.simultaneous and max([len(qbs) for qbs in qbm_all]) > 1:
                # create a child for each measurement
                task_lists = []
                if self.simultaneous_groups is not None:
                    for group in self.simultaneous_groups:
                        new_task_list = []
                        for task in task_list:
                            group = [qb if isinstance(qb, str) else qb.name
                                     for qb in group]
                            new_task = copy(task)
                            new_task['qubits_to_measure'] = [
                                qb for qb in new_task['qubits_to_measure']
                                if qb in group]
                            new_task_list.append(new_task)
                        task_lists.append(new_task_list)
                    # the children measure simultaneously within each group
                    kw['simultaneous'] = True
                else:
                    # the number of required children is the length of the
                    # longest qubits_to_measure
                    for z in zip_longest(*qbm_all):
                        new_task_list = []
                        for task, new_qb in zip(task_list, z):
                            if new_qb is not None:
                                new_task = copy(task)
                                new_task['qubits_to_measure'] = [new_qb]
                                new_task_list.append(new_task)
                        task_lists.append(new_task_list)

                # device object will be needed for update
                self.dev = kw.get('dev', None)
                # children should not update
                self.update = kw.pop('update', False)
                self.measurements = [DynamicPhase(tl, sweep_points, **kw)
                                     for tl in task_lists]

                if self.measurements[0].analyze:
                    for m in self.measurements:
                        for k, v in m.dyn_phases.items():
                            if k not in self.dyn_phases:
                                self.dyn_phases[k] = {}
                            self.dyn_phases[k].update(v)
            else:
                # this happens if we are in child or if simultaneous=True or
                # if only one qubit per task is measured
                self.measurements = [self]
                super().__init__(task_list, sweep_points=sweep_points, **kw)

                self.add_default_sweep_points(**kw)

                if self.reset_phases_before_measurement:
                    for task in task_list:
                        self.operation_dict[self.get_cz_operation_name(
                            **task)]['basis_rotation'] = {}

                self.preprocessed_task_list = self.preprocess_task_list(**kw)
                # the block alignments are for: prepended pulses, initial
                # rotations, flux pulse, final rotations
                self.sequences, self.mc_points = self.parallel_sweep(
                    self.preprocessed_task_list, self.dynamic_phase_block,
                    block_align=['center', 'end', 'center', 'start'], **kw)
                self.autorun(**kw)

            if self.update:
                assert self.dev is not None, \
                    "Update only works with device object provided."
                assert self.measurements[0].analyze, \
                    "Update is only allowed with analyze=True."
                assert len(self.measurements[0].mc_points[1]) == 1, \
                    "Update is only allowed without a soft sweep."

                for op, dp in self.dyn_phases.items():
                    op_split = op.split(' ')
                    basis_rot_par = self.dev.get_pulse_par(
                        *op_split, param='basis_rotation')

                    if self.reset_phases_before_measurement:
                        basis_rot_par(dp)
                    else:
                        not_updated = {k: v for k, v in basis_rot_par().items()
                                       if k not in dp}
                        basis_rot_par().update(dp)
                        if len(not_updated) > 0:
                            log.warning(f'Not all basis_rotations stored in the '
                                        f'pulse settings for {op} have been '
                                        f'measured. Keeping the following old '
                                        f'value(s): {not_updated}')
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def add_default_sweep_points(self, **kw):
        self.sweep_points = self.add_default_ramsey_sweep_points(
            self.sweep_points, **kw)
        nr_phases = self.sweep_points.length(0) // 2
        hard_sweep_dict = SweepPoints(
            'flux_pulse_off', [0] * nr_phases + [1] * nr_phases)
        self.sweep_points.update(hard_sweep_dict + [{}])

    def guess_label(self, **kw):
        if self.label is None:
            self.label = f'Dynamic_phase_measurement'
            for task in self.task_list:
                self.label += "_" + task['prefix'] + "_"
                for qb_name in task['qubits_to_measure']:
                    self.label += f"{qb_name}"

    def dynamic_phase_block(self, sweep_points, op_code, qubits_to_measure,
                            prepend_pulse_dicts=None, **kw):

        assert (sum([qb in op_code.split(' ')[1:] for qb in qubits_to_measure])
                <= 1), \
            f"Dynamic phases of control and target qubit cannot be " \
            f"measured simultaneously ({op_code})."

        hard_sweep_dict, soft_sweep_dict = sweep_points

        pb = self.prepend_pulses_block(prepend_pulse_dicts)

        pulse_modifs = {
            'all': {'element_name': 'pi_half_start', 'ref_pulse': 'start'}}
        ir = self.block_from_ops('initial_rots',
                                 [f'X90 {qb}' for qb in qubits_to_measure],
                                 pulse_modifs=pulse_modifs)
        for p in ir.pulses[1:]:
            p['ref_point_new'] = 'end'

        # calling op_replace_cz() allows to have a custom cz_pulse_name in kw
        fp = self.block_from_ops(
            'flux', self.get_cz_operation_name(op_code=op_code, **kw))
        fp.pulses[0]['pulse_off'] = ParametricValue('flux_pulse_off')
        # FIXME: currently, this assumes that only flux pulse parameters are
        #  swept in the soft sweep. In fact, channels_to_upload should be
        #  determined based on the sweep_points
        for k in ['channel', 'channel2']:
            if k in fp.pulses[0]:
                if fp.pulses[0][k] not in self.channels_to_upload:
                    self.channels_to_upload.append(fp.pulses[0][k])

        for k in soft_sweep_dict:
            if '=' not in k:  # pulse modifier in the sweep dict
                fp.pulses[0][k] = ParametricValue(k)

        pulse_modifs = {
            'all': {'element_name': 'pi_half_end', 'ref_pulse': 'start'}}
        fr = self.block_from_ops('final_rots',
                                 [f'X90 {qb}' for qb in qubits_to_measure],
                                 pulse_modifs=pulse_modifs)
        fr.set_end_after_all_pulses()
        for p in fr.pulses:
            for k in hard_sweep_dict.keys():
                if '=' not in k and k != 'flux_pulse_off':
                    p[k] = ParametricValue(k)

        self.data_to_fit.update({qb: 'pe' for qb in qubits_to_measure})
        return [pb, ir, fp, fr]

    def get_meas_objs_from_task(self, task):
        return task['qubits_to_measure']

    def run_analysis(self, **kw):
        extract_only = kw.pop('extract_only', False)
        for task in self.task_list:
            op = self.get_cz_operation_name(**task)
            op_split = op.split(' ')
            self.dynamic_phase_analysis[task['prefix']] = \
                tda.DynamicPhaseAnalysis(
                    qb_names=task['qubits_to_measure'],
                    options_dict={
                        'flux_pulse_length': self.dev.get_pulse_par(
                            *op_split, param='pulse_length')(),
                        'flux_pulse_amp': self.dev.get_pulse_par(
                            *op_split, param='amplitude')(),
                        # FIXME in analysis: in case of a soft sweep, analysis
                        #  has to overwrite length and amp with values from the
                        #  sweep_points
                        'save_figs': ~extract_only}, extract_only=extract_only)
            self.dyn_phases[op] = {}
            for qb_name in task['qubits_to_measure']:
                self.dyn_phases[op][qb_name] = \
                    self.dynamic_phase_analysis[task['prefix']].proc_data_dict[
                        'analysis_params_dict'][f"dynamic_phase_{qb_name}"][
                        'val'] * 180 / np.pi

        return self.dyn_phases, self.dynamic_phase_analysis



class Chevron(CalibBuilder):
    """
    TODO

    Args:
        FIXME: add further args
        TODO
        :param cz_pulse_name: see CircuitBuilder
        :param n_cal_points_per_state: see CalibBuilder.get_cal_points()
    ...
    """

    def __init__(self, task_list, sweep_points=None, **kw):
        try:
            self.experiment_name = 'Chevron'
            for task in task_list:
                if task.get('qbr', None) is None:
                    task['qbr'] = task['qbt']
                for k in ['qbc', 'qbt', 'qbr']:
                    if not isinstance(task[k], str):
                        task[k] = task[k].name
                if task['qbr'] not in [task['qbc'], task['qbt']]:
                    raise ValueError(
                        'Only target or control qubit can be read out!')
                if not 'prefix' in task:
                    task['prefix'] = f"{task['qbc']}{task['qbt']}_"

            super().__init__(task_list, sweep_points=sweep_points, **kw)

            self.preprocessed_task_list = self.preprocess_task_list(**kw)
            # the block alignments are for: prepended pulses, initial
            # rotations, flux pulse
            self.sequences, self.mc_points = self.parallel_sweep(
                self.preprocessed_task_list, self.sweep_block,
                block_align = ['center', 'end', 'center'], **kw)

            self.autorun(**kw)
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def add_default_sweep_points(self, sweep_points, **kw):
        sweep_points = self.add_default_ramsey_sweep_points(sweep_points, **kw)
        nr_phases = sweep_points.length(0) // 2
        hard_sweep_dict = SweepPoints(
            'pi_pulse_off', [0] * nr_phases + [1] * nr_phases)
        sweep_points.update(hard_sweep_dict + [{}])
        return sweep_points

    def sweep_block(self, sweep_points,
                     qbc, qbt, qbr, num_cz_gates=1, max_flux_length=None,
                     prepend_pulse_dicts=None, **kw):
        """
        chevron block (sweep of flux pulse parameters)

        Timings of sequence
                                      <-- length -->
        qb_control:    |X180|  ---   |  fluxpulse   |

        qb_target:     |X180|  --------------------------------------  |RO|

        TODO
        :param cz_pulse_name: task-specific prefix of CZ gates (overwrites
            global choice passed to the class init)
        ...
        """

        hard_sweep_dict, soft_sweep_dict = sweep_points

        pb = self.prepend_pulses_block(prepend_pulse_dicts)
        pulse_modifs = {'all': {'element_name': 'initial_rots_el'}}
        ir = self.block_from_ops('initial_rots',
                                 [f'X180 {qbc}', f'X180 {qbt}'],
                                 pulse_modifs=pulse_modifs)
        ir.pulses[1]['ref_point_new'] = 'end'

        fp = self.block_from_ops('flux', [f"{kw.get('cz_pulse_name', 'CZ')} "
                                          f"{qbc} {qbt}"] * num_cz_gates)
        # FIXME: currently, this assumes that only flux pulse parameters are
        #  swept in the soft sweep. In fact, channels_to_upload should be
        #  determined based on the sweep_points
        for k in ['channel', 'channel2']:
            if k in fp.pulses[0]:
                if fp.pulses[0][k] not in self.channels_to_upload:
                    self.channels_to_upload.append(fp.pulses[0][k])

        for k in list(hard_sweep_dict.keys()) + list(soft_sweep_dict.keys()):
            for p in fp.pulses:
                p[k] = ParametricValue(k)

        if max_flux_length is not None:
            log.debug(f'max_flux_length = {max_flux_length * 1e9:.2f} ns, '
                      f'set by user')
        max_flux_length = self.max_pulse_length(fp.pulses[0], sweep_points,
                                                max_flux_length)
        w = self.block_from_ops('wait', [])
        w.block_end.update({'pulse_delay': max_flux_length * num_cz_gates})
        fp_w = self.simultaneous_blocks('sim', [fp, w], block_align='center')

        self.data_to_fit.update({qbr: 'pe'})
        return [pb, ir, fp_w]

    def guess_label(self, **kw):
        if self.label is None:
            self.label = self.experiment_name
            for t in self.task_list:
                self.label += f"_{t['qbc']}{t['qbt']}"

    def get_meas_objs_from_task(self, task):
        # FIXME is this correct? it will prevent us from doing
        #  preselection/reset on the other qubit
        return [task['qbr']]

    def run_analysis(self, analysis_kwargs=None, **kw):
        """
        Runs analysis and stores analysis instance in self.analysis.
        :param analysis_kwargs: (dict) keyword arguments for analysis
        :param kw: currently ignored
        :return: the analysis instance
        """
        if analysis_kwargs is None:
            analysis_kwargs = {}
        if 'options_dict' not in analysis_kwargs:
            analysis_kwargs['options_dict'] = {}
        if 'TwoD' not in analysis_kwargs['options_dict']:
            analysis_kwargs['options_dict']['TwoD'] = True
        self.analysis = tda.MultiQubit_TimeDomain_Analysis(
            qb_names=[task['qbr'] for task in self.task_list],
            t_start=self.timestamp, **analysis_kwargs)
        return self.analysis
