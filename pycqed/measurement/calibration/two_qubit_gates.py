import numpy as np
from copy import copy
from copy import deepcopy
from itertools import zip_longest
import traceback
from pycqed.utilities.general import temporary_value
from pycqed.measurement.waveform_control.circuit_builder import CircuitBuilder
from pycqed.measurement.waveform_control.block import Block, ParametricValue
from pycqed.measurement.waveform_control.segment import UnresolvedPulse
from pycqed.measurement.sweep_points import SweepPoints
from pycqed.measurement.calibration.calibration_points import CalibrationPoints
import pycqed.measurement.awg_sweep_functions as awg_swf
import pycqed.analysis_v2.timedomain_analysis as tda
from pycqed.instrument_drivers.meta_instrument.qubit_objects.QuDev_transmon \
    import QuDev_transmon
from pycqed.measurement.multi_qubit_module import \
    get_multiplexed_readout_detector_functions
import logging
log = logging.getLogger(__name__)

# TODO: dostrings (list all kw at the highest level with reference to where
#  they are explained, explain all kw where they are processed)
# TODO: add some comments that explain the way the code works

class CalibBuilder(CircuitBuilder):
    def __init__(self, dev, **kw):
        super().__init__(dev=dev, **kw)
        self.MC = dev.instr_mc.get_instr()

        self.cal_points = None
        self.sweep_points = None
        self.cal_states = None
        self.task_list = None
        self.ro_qubits = None

        self.classified = kw.pop('classified', False)
        self.label = kw.pop('label', None)
        self.upload = kw.pop('upload', True)
        self.update = kw.pop('update', False)
        self.measure = kw.pop('measure', True)
        self.analyze = kw.pop('analyze', True)
        self.exp_metadata = kw.pop('exp_metadata', None)

        self.sequences = []
        self.hard_sweep_points = []
        self.soft_sweep_points = []
        self.channels_to_upload = []
        self.det_get_values_kws = {'classified': self.classified,
                                   'correlated': False,
                                   'thresholded': True,
                                   'averaged': True}

        self.exp_metadata = kw.pop('exp_metadata', None)
        if self.exp_metadata is None:
            self.exp_metadata = {}
        kw.pop('qubits', None)
        self.exp_metadata.update(kw)
        self.exp_metadata.update({'classified_ro': self.classified})

    def run_measurement(self, **kw):
        # only measure ro_qubits
        for qb in self.ro_qubits:
            qb.prepare(drive='timedomain')

        if len(self.sweep_points) == 2:
            # compress 2D sweep
            compression_seg_lim = kw.pop('compression_seg_lim', None)
            if compression_seg_lim is not None:
                self.sequences, self.hard_sweep_points, \
                self.soft_sweep_points, cf = \
                    self.sequences[0].compress_2D_sweep(self.sequences,
                                                        compression_seg_lim)
                self.exp_metadata.update({'compression_factor': cf})

        hard_sweep_func = awg_swf.SegmentHardSweep(
            sequence=self.sequences[0], upload=self.upload,
            parameter_name=list(self.sweep_points[0])[0],
            unit=list(self.sweep_points[0].values())[0][2])

        self.MC.set_sweep_function(hard_sweep_func)
        self.MC.set_sweep_points(self.hard_sweep_points)

        if len(self.sweep_points) == 2:
            self.channels_to_upload = kw.get('channels_to_upload',
                                             self.channels_to_upload)
            self.MC.set_sweep_function_2D(awg_swf.SegmentSoftSweep(
                hard_sweep_func, self.sequences,
                list(self.sweep_points[1])[0],
                list(self.sweep_points[1].values())[0][2],
                channels_to_upload=self.channels_to_upload))
            self.MC.set_sweep_points_2D(self.soft_sweep_points)

        det_name = 'int_avg{}_det'.format('_classif' if self.classified else '')
        det_func = get_multiplexed_readout_detector_functions(
            self.ro_qubits, nr_averages=max(qb.acq_averages() for qb in
                                            self.ro_qubits),
            det_get_values_kws=self.det_get_values_kws)[det_name]
        self.MC.set_detector_function(det_func)

        self.exp_metadata.update({
            'preparation_params': self.get_prep_params(),
            'rotate': len(self.cal_states) != 0 and not self.classified,
            'sweep_points': self.sweep_points,
            'meas_obj_value_names_map':
                self.dev.get_meas_obj_value_names_map(self.ro_qubits, det_func),
            'ro_qubits': [qb.name for qb in self.ro_qubits]
        })
        if self.task_list is not None:
            self.exp_metadata.update({'task_list': self.task_list})

        if len(self.sweep_points) == 2:
            self.MC.run_2D(self.label, exp_metadata=self.exp_metadata)
        else:
            self.MC.run(self.label, exp_metadata=self.exp_metadata)

    def get_cal_points(self, n_cal_points_per_state=1, cal_states='auto',
                       for_ef=True, **kw):
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
        self.cal_states = CalibrationPoints.guess_cal_states(cal_states,
                                                        for_ef=for_ef)
        cp = CalibrationPoints.multi_qubit([qb.name for qb in self.ro_qubits],
                                           self.cal_states,
                                           n_per_state=n_cal_points_per_state)
        return cp

    def parallel_sweep(self, sweep_points, task_list=(), block_func=None,
                       sweep_block_func=None, **kw):
        """
        Calls a block creation function for each task in a task list,
        puts these blocks in parallel and sweeps over the given sweep points.

        :param sweep_points: SweepPoints object (or list of sweep_dicts)
        :param task_list: a list of dictionaries, each containing keyword
            arguments for block_func, plus a key 'prefix' with a unique
            prefix string
        :param block_func: a handle to a function that creates a block
        :param sweep_block_func: a function that is passed to sweep_n_dim
            as the argument body_block_func (see docstring there).
        :param kw: keyword arguments are passed to sweep_n_dim
        :return: see sweep_n_dim
        """
        global_sweep_points = SweepPoints(from_dict_list=sweep_points)
        parallel_blocks = []
        for this_task in task_list:
            task = copy(this_task)  # no deepcopy: might qubit objects
            prefix = task.pop('prefix', None)
            if prefix is None:  # try to guess one based on contained qubits
                task_qbs = self.find_qubits_in_tasks(self.qubits, [task])
                prefix = '_'.join([qb.name for qb in task_qbs])
            prefix += ('_' if prefix[-1] != '_' else '')
            current_sweep_points = SweepPoints(from_dict_list=sweep_points)
            if 'sweep_points' in task:
                current_sweep_points.update(
                    SweepPoints(from_dict_list=task['sweep_points']))
                params_to_prefix = [d.keys() for d in task['sweep_points']]
            else:
                params_to_prefix = None
            task['sweep_points'] = current_sweep_points

            if params_to_prefix is not None:
                for d, u, params in zip(global_sweep_points,
                                        current_sweep_points,
                                        params_to_prefix):
                    for k in params:
                        d[prefix + k] = u[k]

            if not 'block_func' in task:
                task['block_func'] = block_func
            if task['block_func'] is not None:
                new_block = task['block_func'](**task)
                if params_to_prefix is not None:
                    new_block.prefix_parametric_values(
                        prefix, [k for l in params_to_prefix for k in l])
                parallel_blocks.append(new_block)

        if len(parallel_blocks) > 0:
            all_main_blocks = self.simultaneous_blocks('all', parallel_blocks)
        else:
            all_main_blocks = None
        if len(global_sweep_points[1]) == 0:
            # TODO add a fake soft sweep instead
            global_sweep_points = \
                SweepPoints(from_dict_list=[global_sweep_points[0]])
        self.sweep_points = global_sweep_points
        self.exp_metadata.update({'cal_points': repr(self.cal_points)})
        # only measure ro_qubits
        if 'ro_kwargs' in kw:
            kw['ro_kwargs'].update({'qb_names': [qb.name for qb in
                                                 self.ro_qubits]})
        else:
            kw.update({'ro_kwargs': {'qb_names': [qb.name for qb in
                                                  self.ro_qubits]}})
        return self.sweep_n_dim(global_sweep_points, body_block=all_main_blocks,
                                body_block_func=sweep_block_func,
                                cal_points=self.cal_points, **kw)

    def max_pulse_length(self, pulse, sweep_points, given_pulse_length=None):
        pulse = copy(pulse)
        pulse['element_name'] = 'tmp'

        if given_pulse_length is not None:
            pulse['pulse_length'] = given_pulse_length
            p = UnresolvedPulse(pulse)
            return p.pulse_obj.length

        b = Block('tmp', [pulse])
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

    def get_ro_qubits(self, task_list=None):
        if task_list is None:
            task_list = self.task_list
        if task_list is None:
            task_list = [{}]
        ro_qubit_names = [task.get('qubits_to_measure', []) for
                          task in task_list]
        if any([not isinstance(qbn, str) for qbn in ro_qubit_names]):
            # flatten
            ro_qubit_names = [i for j in ro_qubit_names for i in j]
        # sort
        ro_qubit_names.sort()
        ro_qubits = self.get_qubits('all' if len(ro_qubit_names) == 0
                                    else ro_qubit_names)[0]
        return ro_qubits

    @staticmethod
    def add_default_ramsey_sweep_points(sweep_points, **kw):
        if sweep_points is None:
            sweep_points = [{}, {}]
        sweep_points = SweepPoints(from_dict_list=sweep_points)
        if len(sweep_points) == 1:
            sweep_points.add_sweep_dimension()
        if len(sweep_points[0]) > 0:
            nr_phases = len(list(sweep_points[0].values())[0][0]) // 2
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

    @staticmethod
    def find_qubits_in_tasks(qubits, task_list, search_in_operations=True):
        qbs_dict = {qb.name: qb for qb in qubits}
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


class CPhase(CalibBuilder):
    """
    Creates a CalibrationPoints object based on the given parameters.

    TODO
    :param cz_pulse_name: see CircuitBuilder
    :param n_cal_points_per_state: see CalibBuilder.get_cal_points()
    ...
    """

    def __init__(self, dev, task_list, sweep_points=None, **kw):

        try:
            for task in task_list:
                for k in ['qbl', 'qbr']:
                    if not isinstance(task[k], str):
                        task[k] = task[k].name
                if not 'prefix' in task:
                    task['prefix'] = f"{task['qbl']}{task['qbr']}_"

            qubits = self.find_qubits_in_tasks(dev.qubits(), task_list)
            super().__init__(dev, qubits=qubits, **kw)

            self.cphases = None
            self.population_losses = None
            self.leakage = None
            self.analysis = None
            self.cz_durations = {}
            self.cal_states_rotations = {}
            self.data_to_fit = {}

            self.task_list = task_list
            self.ro_qubits = self.get_ro_qubits()
            self.guess_label(**kw)
            sweep_points = self.add_default_sweep_points(sweep_points, **kw)
            self.cal_points = self.get_cal_points(**kw)
            self.sequences, sp = \
                self.parallel_sweep(sweep_points, task_list, self.cphase_block,
                                    **kw)
            self.hard_sweep_points, self.soft_sweep_points = sp
            self.exp_metadata.update({
                'cz_durations': self.cz_durations,
                'cal_states_rotations': self.cal_states_rotations,
                'data_to_fit': self.data_to_fit
            })

            if self.measure:
                self.run_measurement(**kw)
            if self.analyze:
                self.run_analysis(**kw)
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def add_default_sweep_points(self, sweep_points, **kw):
        sweep_points = self.add_default_ramsey_sweep_points(sweep_points, **kw)
        nr_phases = len(list(sweep_points[0].values())[0][0]) // 2
        hard_sweep_dict = SweepPoints(
            'pi_pulse_off', [0] * nr_phases + [1] * nr_phases)
        sweep_points.update(hard_sweep_dict + [{}])
        return sweep_points

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
                self.label = 'Predictive_cphase_measurement'
            else:
                self.label = 'CPhase_measurement'
            if self.classified:
                self.label += '_classified'
            if 'active' in self.get_prep_params()['preparation_type']:
                self.label += '_reset'
            # if num_cz_gates > 1:
            #     label += f'_{num_cz_gates}_gates'
            for t in self.task_list:
                self.label += f"_{t['qbl']}{t['qbr']}"

    def run_analysis(self, **kw):
        plot_all_traces = kw.get('plot_all_traces', True)
        plot_all_probs = kw.get('plot_all_probs', True)
        if self.classified:
            channel_map = {qb.name: [vn + ' ' +
                                     qb.instr_uhf() for vn in
                                     qb.int_avg_classif_det.value_names]
                           for qb in self.qubits}
        else:
            channel_map = {qb.name: [vn + ' ' +
                                     qb.instr_uhf() for vn in
                                     qb.int_avg_det.value_names]
                           for qb in self.qubits}
        self.analysis = tda.CPhaseLeakageAnalysis(
            qb_names=[qb.name for qb in self.qubits],
            options_dict={'TwoD': True, 'plot_all_traces': plot_all_traces,
                          'plot_all_probs': plot_all_probs,
                          'channel_map': channel_map})
        self.cphases = self.analysis.proc_data_dict[
            'analysis_params_dict']['cphase']['val']
        self.population_losses = self.analysis.proc_data_dict[
            'analysis_params_dict']['population_loss']['val']
        self.leakage = self.analysis.proc_data_dict[
            'analysis_params_dict']['leakage']['val']

        return self.cphases, self.population_losses, self.leakage, \
               self.analysis


class DynamicPhase(CalibBuilder):
    def __init__(self, dev, task_list, sweep_points=None, **kw):
        try:
            self.simultaneous = kw.get('simultaneous', False)
            self.reset_phases_before_measurement = kw.get(
                'reset_phases_before_measurement', True)

            self.dynamic_phase_analysis = {}
            self.dyn_phases = {}
            self.old_dyn_phases = {}
            for task in task_list:
                if task.get('qubits_to_measure', None) is None:
                    task['qubits_to_measure'] = self.find_qubits_in_tasks(
                        dev.qubits(), [task])
                else:
                    task['qubits_to_measure'] = copy(task['qubits_to_measure'])

                for k, v in enumerate(task['qubits_to_measure']):
                    if not isinstance(v, str):
                        task['qubits_to_measure'][k] = v.name

                if 'prefix' not in task:
                    task['prefix'] = task['op_code'].replace(' ', '')
                self.dyn_phases[task['prefix']] = {}
                self.old_dyn_phases[task['prefix']] = {}

            qbs_all = [task['qubits_to_measure'] for task in task_list]
            if not self.simultaneous and max([len(qbs) for qbs in qbs_all]) > 1:
                task_lists = []
                for z in zip_longest(*qbs_all):
                    new_task_list = []
                    for task, new_qb in zip(task_list, z):
                        if new_qb is not None:
                            new_task = copy(task)
                            new_task['qubits_to_measure'] = [new_qb]
                            new_task_list.append(new_task)
                    task_lists.append(new_task_list)

                # children should not update
                self.update = kw.pop('update', False)
                self.measurements = [DynamicPhase(dev, tl, sweep_points, **kw)
                                     for tl in task_lists]

                if self.measurements[0].analyze:
                    for m in self.measurements:
                        [d.update(u) for d, u in zip(self.dyn_phases.values(),
                                                     m.dyn_phases.values())]
                        [d.update(u) for d, u in zip(
                            self.old_dyn_phases.values(),
                            m.old_dyn_phases.values())]
            else:
                self.measurements = [self]
                qubits = self.find_qubits_in_tasks(dev.qubits(), task_list)
                super().__init__(dev, qubits=qubits, **kw)

                self.cal_states_rotations = {}
                self.data_to_fit = {}

                self.task_list = task_list
                self.ro_qubits = self.get_ro_qubits(task_list)
                self.guess_label(**kw)
                sweep_points = self.add_default_sweep_points(sweep_points, **kw)
                if 'for_ef' not in kw:
                    kw['for_ef'] = False
                self.cal_points = self.get_cal_points(**kw)
                self.basis_rot_pars = {}

                for task in task_list:
                    op_split = task['op_code'].split(' ')
                    if op_split[0] == 'CZ':
                        op_split[0] = self.cz_pulse_name
                    self.basis_rot_pars[task['prefix']] = dev.get_pulse_par(
                        *op_split, param='basis_rotation')
                    if self.reset_phases_before_measurement:
                        self.old_dyn_phases[task['prefix']] = {}
                    else:
                        self.old_dyn_phases[task['prefix']] = deepcopy(
                            self.basis_rot_pars[task['prefix']]())

                tmpvals = [(v, self.old_dyn_phases[k]) for k, v in
                           self.basis_rot_pars.items()]
                with temporary_value(*tmpvals):
                    self.sequences, sp = self.parallel_sweep(
                        sweep_points, task_list, self.dynamic_phase_block, **kw)
                    self.hard_sweep_points = sp[0]
                    if len(sp) > 1:
                        self.soft_sweep_points = sp[1]

                    self.exp_metadata.update({
                        'cal_states_rotations': self.cal_states_rotations,
                        'data_to_fit': self.data_to_fit
                    })

                    if self.measure:
                        self.run_measurement(**kw)

                if self.analyze:
                    self.run_analysis(**kw)

            if self.update:
                assert self.measurements[0].analyze, \
                    "Update is only allowed with analyze=True."
                assert len(self.measurements[0].sweep_points) == 1, \
                    "Update is only allowed without a soft sweep."

                for op, dp in self.dyn_phases.items():
                    basis_rot_par = self.measurements[0].basis_rot_pars[op]
                    if self.reset_phases_before_measurement:
                        basis_rot_par(dp)
                    else:
                        basis_rot_par().update(dp)

                    not_updated = {k: v for k, v in
                                   self.old_dyn_phases[op].items()
                                   if k not in dp}
                    if len(not_updated) > 0:
                        log.warning(f'Not all basis_rotations stored in the '
                                    f'pulse settings for {op} have been '
                                    f'measured. Keeping the following old '
                                    f'value(s): {not_updated}')
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def add_default_sweep_points(self, sweep_points, **kw):
        sweep_points = self.add_default_ramsey_sweep_points(sweep_points, **kw)
        nr_phases = len(list(sweep_points[0].values())[0][0]) // 2
        hard_sweep_dict = SweepPoints(
            'flux_pulse_off', [0] * nr_phases + [1] * nr_phases)
        sweep_points.update(hard_sweep_dict + [{}])
        return sweep_points

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

        fp = self.block_from_ops('flux', op_code)
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

        self.cal_states_rotations.update(self.cal_points.get_rotations(
            qb_names=qubits_to_measure, **kw))
        self.data_to_fit.update({qb: 'pe' for qb in qubits_to_measure})
        return self.sequential_blocks(
            f"dynphase {'_'.join(qubits_to_measure)}", [pb, ir, fp, fr])

    def run_analysis(self, **kw):
        extract_only = kw.pop('extract_only', False)
        for task in self.task_list:
            op_split = task['op_code'].split(' ')
            if op_split[0] == 'CZ':
                op_split[0] = self.cz_pulse_name
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
                        'sweep_points': self.sweep_points,
                        'save_figs': ~extract_only}, extract_only=extract_only)
            for qb_name in task['qubits_to_measure']:
                self.dyn_phases[task['prefix']][qb_name] = \
                    self.dynamic_phase_analysis[task['prefix']].proc_data_dict[
                        'analysis_params_dict'][qb_name][
                        'dynamic_phase']['val'] * 180 / np.pi

        return self.dyn_phases, self.dynamic_phase_analysis