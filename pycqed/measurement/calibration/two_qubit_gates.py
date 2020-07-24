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
    kw_for_sweep_points = {}
    kw_for_task_keys = ()

    def __init__(self, task_list, dev=None, qubits=None,
                 operation_dict=None, **kw):

        self.task_list = task_list
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
            self.sweep_points = kw.pop('sweep_points')
        self.cal_points = None
        self.cal_states = None
        self.exception = None
        self.all_main_blocks = []
        self.data_to_fit = {}
        self.experiment_name = kw.pop(
            'experiment_name', getattr(self, 'experiment_name', 'Experiment'))

        # The following is done because the respective call in the init of
        # QuantumExperiment does not capture all kw since many are explicit
        # arguments of the init there.
        kw.pop('exp_metadata', None)
        self.exp_metadata.update(kw)

        self.create_cal_points(**kw)

    def add_to_meas_obj_sweep_points_map(self, meas_objs, sweep_point):
        if 'meas_obj_sweep_points_map' not in self.exp_metadata:
            self.exp_metadata['meas_obj_sweep_points_map'] = {}
        if not isinstance(meas_objs, list):
            meas_objs = [meas_objs]
        for mo in meas_objs:
            if mo not in self.exp_metadata['meas_obj_sweep_points_map']:
                self.exp_metadata['meas_obj_sweep_points_map'][mo] = []
            if sweep_point not in self.exp_metadata[
                    'meas_obj_sweep_points_map'][mo]:
                self.exp_metadata['meas_obj_sweep_points_map'][mo].append(
                    sweep_point)

    def get_meas_objs_from_task(self, task):
        return self.find_qubits_in_tasks(self.qb_names, [task])

    def guess_label(self, **kw):
        if self.label is None:
            self.label = self.experiment_name
            if self.dev is not None:
                self.label += self.dev.get_msmt_suffix(self.meas_obj_names)
            else:
                # guess_label is called from run_measurement -> we have qubits
                self.label += mqm.get_multi_qubit_msmt_suffix(self.meas_objs)

    def run_measurement(self, **kw):
        # allow the user to overwrite the automatically generated list of
        # channels to upload
        self.channels_to_upload = kw.get('channels_to_upload',
                                         self.channels_to_upload)
        self.df_kwargs.update(
            {'nr_averages': max(qb.acq_averages() for qb in self.meas_objs)})

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
        Creates a CalibrationPoints object based on the given parameters.

        :param n_cal_points_per_state: number of segments for each
            calibration state
        :param cal_states: str or tuple of str; the calibration states
            to measure
        :param for_ef: bool indicating whether to measure the |f> calibration
            state for each qubit
        :param kw: keyword arguments (to allow pass through kw even if it
            contains entries that are not needed)
        :return: CalibrationPoints object
        """
        self.cal_states = CalibrationPoints.guess_cal_states(
            cal_states, for_ef=for_ef)
        self.cal_points = CalibrationPoints.multi_qubit(
            self.meas_obj_names, self.cal_states,
            n_per_state=n_cal_points_per_state)
        self.exp_metadata.update({'cal_points': repr(self.cal_points)})

    def preprocess_task_list(self, **kw):
        given_sweep_points = self.sweep_points
        self.sweep_points = SweepPoints(from_dict_list=given_sweep_points)
        while len(self.sweep_points) < 2:
            self.sweep_points.add_sweep_dimension()
        preprocessed_task_list = []
        for task in self.task_list:
            preprocessed_task_list.append(
                self.preprocess_task(task, self.sweep_points,
                                     given_sweep_points, **kw))
        return preprocessed_task_list

    def preprocess_task(self, task, global_sweep_points, sweep_points=None,
                        **kw):
        task = copy(task)  # no deepcopy: might contain qubit objects
        prefix = task.get('prefix', None)
        if prefix is None:  # try to guess one based on contained qubits
            prefix = '_'.join(self.find_qubits_in_tasks(self.qb_names, [task]))
        prefix += ('_' if prefix[-1] != '_' else '')
        task['prefix'] = prefix
        mo = self.get_meas_objs_from_task(task)

        for param in self.kw_for_task_keys:
            if param not in task:
                task[param] = kw.get(param, None)

        current_sweep_points = SweepPoints(from_dict_list=sweep_points)
        self.generate_kw_sweep_points(task)
        current_sweep_points.update(
            SweepPoints(from_dict_list=task['sweep_points']))
        params_to_prefix = [d.keys() for d in task['sweep_points']]
        task['params_to_prefix'] = params_to_prefix
        task['sweep_points'] = current_sweep_points

        while len(current_sweep_points) < 2:
            current_sweep_points.add_sweep_dimension()
        while len(params_to_prefix) < 2:
            params_to_prefix.append([])
        for gsp, csp, params in zip(global_sweep_points,
                                    current_sweep_points,
                                    params_to_prefix):
            for k in csp.keys():
                if k in params:
                    gsp[prefix + k] = csp[k]
                    self.add_to_meas_obj_sweep_points_map(mo, prefix + k)
                else:
                    self.add_to_meas_obj_sweep_points_map(mo, k)
        return task

    def parallel_sweep(self, preprocessed_task_list=(), block_func=None,
                       block_align='start', **kw):
        """
        Calls a block creation function for each task in a task list,
        puts these blocks in parallel and sweeps over the given sweep points.

        :param task_list: a list of dictionaries, each containing keyword
            arguments for block_func, plus a key 'prefix' with a unique
            prefix string, plus optionally a key 'params_to_prefix' created
            by preprocess_task indicating which sweep parameters have to be
            prefixed with the task prefix.
        :param block_func: a handle to a function that creates a block
        :param kw: keyword arguments are passed to sweep_n_dim
        :return: see sweep_n_dim
        """
        parallel_blocks = []
        for task in preprocessed_task_list:
            task = copy(task)
            prefix = task.pop('prefix')
            params_to_prefix = task.pop('params_to_prefix', None)
            if not 'block_func' in task:
                task['block_func'] = block_func
            new_block = task['block_func'](**task)
            if params_to_prefix is not None:
                new_block.prefix_parametric_values(
                    prefix, [k for l in params_to_prefix for k in l])
            parallel_blocks.append(new_block)

        self.all_main_blocks = self.simultaneous_blocks('all', parallel_blocks,
                                                        block_align=block_align)
        if len(self.sweep_points[1]) == 0:
            # with this dummy soft sweep, exactly one sequence will be created
            # and the data format will be the same as for a true soft sweep
            self.sweep_points.add_sweep_parameter('dummy_sweep_param', [0])
        # only measure meas_objs
        if 'ro_qubits' not in kw:
            op_codes = [p['op_code'] for p in self.all_main_blocks.pulses if
                        'op_code' in p]
            kw = copy(kw)
            kw['ro_qubits'] = [m for m in self.meas_obj_names if f'RO {m}'
                               not in op_codes]

        return self.sweep_n_dim(self.sweep_points,
                                body_block=self.all_main_blocks,
                                cal_points=self.cal_points, **kw)

    @staticmethod
    def find_qubits_in_tasks(qubits, task_list, search_in_operations=True):
        qbs_dict = {qb if isinstance(qb, str) else qb.name: qb for qb in
                    qubits}
        found_qubits = []

        def append_qbs(found_qubits, candidate):
            if isinstance(candidate, QuDev_transmon):
                if candidate not in found_qubits:
                    found_qubits.append(candidate)
            elif isinstance(candidate, str):
                if candidate in qbs_dict.keys():
                    if qbs_dict[candidate] not in found_qubits:
                        found_qubits.append(qbs_dict[candidate])
                elif ' ' in candidate and search_in_operations:
                    append_qbs(found_qubits, candidate.split(' '))
            elif isinstance(candidate, list):
                for v in candidate:
                    append_qbs(found_qubits, v)
            else:
                return None

        for task in task_list:
            for v in task.values():
                append_qbs(found_qubits, v)
        return found_qubits

    def create_meas_objs_list(self, task_list=None, **kw):
        if task_list is None:
            task_list = self.task_list
        if task_list is None:
            task_list = [{}]
        ro_qubits = kw.pop('ro_qubits', None)
        if ro_qubits is None:
            ro_qubits = [qb for task in task_list for qb in task.pop(
                'ro_qubits', self.get_meas_objs_from_task(task))]
        else:
            ro_qubits += [qb for task in task_list for qb in
                          task.pop('ro_qubits', [])]
        # unique and sort
        ro_qubits = [qb if isinstance(qb, str) else qb.name for qb in
                     ro_qubits]
        ro_qubits = list(np.unique(ro_qubits))
        ro_qubits.sort()
        self.meas_objs, self.meas_obj_names = self.get_qubits(
            'all' if len(ro_qubits) == 0 else ro_qubits)

    def generate_kw_sweep_points(self, task):
        # instead of a task, a kw dict can also be passed
        task['sweep_points'] = SweepPoints(
            from_dict_list=task.get('sweep_points', None))
        for k, vals in self.kw_for_sweep_points.items():
            if isinstance(vals, dict):
                vals = [vals]
            for v in vals:
                values_func = v.pop('values_func', None)
                if k in task and task[k] is not None:
                    if values_func is not None:
                        values = values_func(task[k])
                    elif isinstance(task[k], int):
                        values = np.arange(task[k])
                    else:
                        values = task[k]
                    task['sweep_points'].add_sweep_parameter(
                        values=values, **v)

class CalibBuilder(MultiTaskingExperiment):
    def __init__(self, task_list, **kw):
        super().__init__(task_list=task_list, **kw)
        self.update = kw.pop('update', False)

    def max_pulse_length(self, pulse, sweep_points=None,
                         given_pulse_length=None):
        pulse = copy(pulse)
        pulse['name'] = 'tmp'
        pulse['element_name'] = 'tmp'

        if given_pulse_length is not None:
            pulse['pulse_length'] = given_pulse_length
            p = UnresolvedPulse(pulse)
            return p.pulse_obj.length

        b = Block('tmp', [pulse])
        sweep_points = deepcopy(sweep_points)
        if sweep_points is None:
            sweep_points = SweepPoints(from_dict_list=[{}, {}])
        if len(sweep_points) == 1:
            sweep_points.add_sweep_dimension()
        for i in range(len(sweep_points)):
            if len(sweep_points[i]) == 0:
                sweep_points[i].update({'dummy': ([0], '', 'dummy')})

        nr_sp_list = [len(list(d.values())[0][0]) for d in sweep_points]
        max_length = 0
        for i in range(nr_sp_list[1]):
            for j in range(nr_sp_list[0]):
                pulses = b.build(
                    sweep_dicts_list=sweep_points, sweep_index_list=[j, i])
                p = UnresolvedPulse(pulses[1])
                max_length = max(p.pulse_obj.length, max_length)
        return max_length

    def prepend_pulses_block(self, prepend_pulse_dicts):
        prepend_pulses = []
        if prepend_pulse_dicts is not None:
            for i, pp in enumerate(prepend_pulse_dicts):
                prepend_pulse = self.get_pulse(pp['op_code'])
                prepend_pulse.update(pp)
                prepend_pulses += [prepend_pulse]
        return Block('prepend', prepend_pulses)

    @staticmethod
    def add_default_ramsey_sweep_points(sweep_points, **kw):
        sweep_points = SweepPoints(from_dict_list=sweep_points, min_length=2)
        if len(sweep_points[0]) > 0:
            nr_phases = sweep_points.length(0) // 2
        else:
            nr_phases = kw.get('nr_phases', 6)
        hard_sweep_dict = SweepPoints()
        if 'phase' not in sweep_points[0]:
            hard_sweep_dict.add_sweep_parameter(
                'phase',
                np.tile(np.linspace(0, 2 * np.pi, nr_phases) * 180 / np.pi, 2),
                'deg')
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
            self.sequences, self.mc_points = self.parallel_sweep(
                self.preprocessed_task_list, self.cphase_block, **kw)

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
                                 [f'X180 {qbl}', f'X90s {qbr}'] +
                                 kw.get('spectator_op_codes', []),
                                 pulse_modifs=pulse_modifs)
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

        pulse_modifs = {'all': {'element_name': 'cphase_final_rots_el'}}
        fr = self.block_from_ops('final_rots', [f'X180 {qbl}', f'X90s {qbr}'],
                                 pulse_modifs=pulse_modifs)
        fr.pulses[0]['pulse_delay'] = max_flux_length * num_cz_gates
        fr.pulses[0]['pulse_off'] = ParametricValue(param='pi_pulse_off')
        for k in hard_sweep_dict.keys():
            if k != 'pi_pulse_on' and '=' not in k:
                fr.pulses[1][k] = ParametricValue(k)

        self.cz_durations.update({
            fp.pulses[0]['op_code']: fr.pulses[0]['pulse_delay']})
        self.cal_states_rotations.update({qbl: {'g': 0, 'e': 1, 'f': 2},
                                          qbr: {'g': 0, 'e': 1}})
        self.data_to_fit.update({qbl: 'pf', qbr: 'pe'})

        fp_fr = self.simultaneous_blocks('sim', [fp, fr])
        return self.sequential_blocks(f'cphase {qbl} {qbr}', [pb, ir, fp_fr])

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
        try:
            self.simultaneous = kw.get('simultaneous', False)
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
                self.sequences, self.mc_points = self.parallel_sweep(
                    self.preprocessed_task_list, self.dynamic_phase_block, **kw)
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
        for p in fr.pulses:
            for k in hard_sweep_dict.keys():
                if '=' not in k and k != 'flux_pulse_off':
                    p[k] = ParametricValue(k)

        self.data_to_fit.update({qb: 'pe' for qb in qubits_to_measure})
        return self.sequential_blocks(
            f"dynphase {'_'.join(qubits_to_measure)}", [pb, ir, fp, fr])

    def get_meas_objs_from_task(self, task):
        return task['qubits_to_measure']

    def run_analysis(self, **kw):
        extract_only = kw.pop('extract_only', False)
        for task in self.task_list:
            op = self.get_cz_operation_name(**task)
            op_split = op.split(' ')
            self.dynamic_phase_analysis[task['prefix']] = \
                tda.CZDynamicPhaseAnalysis(
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
                        'analysis_params_dict'][qb_name][
                        'dynamic_phase']['val'] * 180 / np.pi

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
            self.sequences, self.mc_points = self.parallel_sweep(
                self.preprocessed_task_list, self.sweep_block, **kw)

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
        b = self.sequential_blocks(f'chevron {qbc} {qbt}', [pb, ir, fp])
        b.block_end.update({'ref_pulse': 'initial_rots-|-end',
                            'pulse_delay': max_flux_length * num_cz_gates})

        self.data_to_fit.update({qbr: 'pe'})

        return b

    def guess_label(self, **kw):
        if self.label is None:
            self.label = self.experiment_name
            for t in self.task_list:
                self.label += f"_{t['qbc']}{t['qbt']}"

    def get_meas_objs_from_task(self, task):
        # FIXME is this correct? it will prevent us from doing
        #  preselection/reset on the other qubit
        return [task['qbr']]

    def run_analysis(self, **kw):
        self.analysis = tda.MultiQubit_TimeDomain_Analysis(
            qb_names=[task['qbr'] for task in self.task_list],
            options_dict={'TwoD': True})
        return self.analysis
