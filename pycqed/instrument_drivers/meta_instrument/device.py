# General imports
import itertools
import logging
from copy import deepcopy

import numpy as np
import pycqed.analysis.fitting_models as fms
import pycqed.analysis_v2.timedomain_analysis as tda
import pycqed.measurement.awg_sweep_functions as awg_swf
import pycqed.measurement.awg_sweep_functions_multi_qubit as awg_swf2
import pycqed.measurement.multi_qubit_module as mqm
import pycqed.measurement.pulse_sequences.fluxing_sequences as fsqs
import pycqed.measurement.pulse_sequences.multi_qubit_tek_seq_elts as mqs
import pycqed.measurement.waveform_control.sequence as sequence
import pycqed.instrument_drivers.meta_instrument.qubit_objects.QuDev_transmon as qdt
import pycqed.measurement.waveform_control.pulse as bpl
import scipy.optimize as opti
from pycqed.analysis import measurement_analysis as ma
from pycqed.analysis_v3 import pipeline_analysis as pla
from pycqed.analysis_v3.processing_pipeline import ProcessingPipeline
from pycqed.measurement import sweep_functions as swf
from pycqed.measurement.calibration_points import CalibrationPoints
from pycqed.measurement.sweep_points import SweepPoints
from pycqed.measurement.waveform_control import pulsar as ps
from pycqed.utilities import general as gen
from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import (
    ManualParameter, InstrumentRefParameter)
from qcodes.utils import validators as vals

log = logging.getLogger(__name__)


class Device(Instrument):
    def __init__(self, name, qubits, connections, **kw):
        super().__init__(name, **kw)

        qb_names = [qb.name for qb in qubits]
        connectivity_graph = [[qb1.name, qb2.name] for [qb1, qb2] in connections]

        for qb_name in qb_names:
            setattr(self, qb_name, self.find_instrument(qb_name))

        self._operations = {}  # dictionary containing dictionaries of operations with parameters

        self.add_parameter('qubits',
                           vals=vals.Lists(),
                           initial_value=qb_names,
                           parameter_class=ManualParameter)

        # Instrument reference parameters
        self.add_parameter('instr_mc',
                           parameter_class=InstrumentRefParameter)
        self.add_parameter('instr_pulsar',
                           parameter_class=InstrumentRefParameter)
        self.add_parameter('instr_dc_source',
                           parameter_class=InstrumentRefParameter)
        self.add_parameter('instr_trigger',
                           parameter_class=InstrumentRefParameter)

        self.add_parameter('connectivity_graph',
                           vals=vals.Lists(),
                           label="Qubit Connectivity Graph",
                           docstring="Stores the connections between the qubits "
                                     "in form of a list of lists [qbi, qbj]",
                           parameter_class=ManualParameter,
                           initial_value=connectivity_graph
                           )
        self.add_parameter('last_calib',
                           vals=vals.Strings(),
                           initial_value='',
                           docstring='stores timestamp of last calibration',
                           parameter_class=ManualParameter)

        self.add_parameter('operations',
                           docstring='a list of all operations available on the qubit',
                           get_cmd=self._get_operations)
        self.add_parameter('two_qb_gates',
                           vals=vals.Lists(),
                           initial_value=[],
                           docstring='stores all two qubit gate names',
                           parameter_class=ManualParameter)

        # Pulse preparation parameters
        default_prep_params = dict(preparation_type='wait',
                                   post_ro_wait=1e-6, reset_reps=1)

        self.add_parameter('preparation_params', parameter_class=ManualParameter,
                           initial_value=default_prep_params, vals=vals.Dict())

    # General Class Methods

    def add_operation(self, operation_name):
        """
        Adds the name of an operation to the operations dictionary.

        Args:
            operation_name (str): name of the operation
        """

        self._operations[operation_name] = {}

    def add_pulse_parameter(self, operation_name, parameter_name, argument_name,
                            initial_value=None, **kw):
        """
        Adds a pulse parameter to an operation. Makes sure that parameters are not duplicated.
        Adds the pulse parameter to the device instrument.

        Args:
            operation_name (tuple): name of operation in format (gate_name, qb1, qb2)
            parameter_name (str): name of parameter
            argument_name (str): name of the argument that is added to the operations dict
            initial_value: initial value of parameter
        """
        if parameter_name in self.parameters:
            raise KeyError(
                'Duplicate parameter name {}'.format(parameter_name))

        if operation_name in self.operations().keys():
            self._operations[operation_name][argument_name] = parameter_name
        else:
            raise KeyError('Unknown operation {}, add '.format(operation_name) +
                           'first using add operation')

        self.add_parameter(parameter_name,
                           initial_value=initial_value,
                           parameter_class=ManualParameter, **kw)

    def _get_operations(self):
        """
        Private method that is used as getter function for operations parameter
        """
        return self._operations

    def get_operation_dict(self, operation_dict=None):
        """
        Returns the operations dictionary of the device and qubits, combined with the input
        operation_dict.

        Args:
            operation_dict (dict): input dictionary the operations should be added to

        Returns:
            operation_dict (dict): dictionary containing both qubit and device operations

        """
        if operation_dict is None:
            operation_dict = dict()

        # add 2qb operations
        two_qb_operation_dict = {}
        for op_tag, op in self.operations().items():
            # op_tag is the tuple (gate_name, qb1, qb2) and op the dictionary of the
            # operation

            # Add both qubit combinations to operations dict
            # Still return a string instead of tuple as keys to be consisten
            # with QudevTransmon class
            for op_name in [op_tag[0] + ' ' + op_tag[1] + ' ' + op_tag[2],
                            op_tag[0] + ' ' + op_tag[2] + ' ' + op_tag[1]]:
                two_qb_operation_dict[op_name] = {}
                for argument_name, parameter_name in op.items():
                    two_qb_operation_dict[op_name][argument_name] = \
                        self.get(parameter_name)

        operation_dict.update(two_qb_operation_dict)

        # add sqb operations
        for qb in self.qubits():
            operation_dict.update(self.get_qb(qb).get_operation_dict())

        return operation_dict

    def get_qb(self, qb_name):
        """
        Wrapper: Returns the qubit instance with name qb_name

        Args:
            qb_name (str): name of the qubit
        Returns:
            qubit instrument with name qubit_name

        """
        return self.find_instrument(qb_name)

    def get_pulse_par(self, gate_name, qb1, qb2, param):
        """
        Returns the value of a two qubit gate parameter.

        Args:
            gate_name (str): Name of the gate
            qb1 (str, QudevTransmon): Name of one qubit
            qb2 (str, QudevTransmon): Name of other qubit
            param (str): name of parameter
        Returns:
            the value of the parameter
        """

        if isinstance(qb1, qdt.QuDev_transmon):
            qb1_name = qb1.name
        else:
            qb1_name = qb1

        if isinstance(qb2, qdt.QuDev_transmon):
            qb2_name = qb2.name
        else:
            qb2_name = qb2

        try:
            self.get(f'{gate_name}_{qb1_name}_{qb2_name}_{param}', value)
        except KeyError:
            try:
                self.get(f'{gate_name}_{qb2_name}_{qb1_name}_{param}', value)
            except KeyError:
                raise ValueError(f'Parameter {param} for the gate '
                                 f'{gate_name} {qb1_name} {qb2_name} '
                                 f'does not exist!')

    def get_prep_params(self, qb_list):
        """
        Returns the preparation paramters for all qubits in qb_list.

        Args:
            qb_list (list): list of qubit names or objects

        Returns:
            dictionary of preparation parameters
        """

        for i, qb in enumerate(qb_list):
            if isinstance(qb, str):
                qb_list[i] = self.get_qb(qb)

        # threshold_map has to be updated for all qubits
        thresh_map = {}
        for prep_params in [qb.preparation_params() for qb in qb_list]:
            if 'threshold_mapping' in prep_params:
                thresh_map.update(prep_params['threshold_mapping'])

        prep_params = deepcopy(self.preparation_params())
        prep_params['threshold_mapping'] = thresh_map

        return prep_params

    def set_pulse_par(self, gate_name, qb1, qb2, param, value):
        """
        Sets a value to a two qubit gate parameter.

        Args:
            gate_name (str): Name of the gate
            qb1 (str, QudevTransmon): Name of one qubit
            qb2 (str, QudevTransmon): Name of other qubit
            param (str): name of parameter
            value: value of parameter
        """

        if isinstance(qb1, qdt.QuDev_transmon):
            qb1_name = qb1.name
        else:
            qb1_name = qb1

        if isinstance(qb2, qdt.QuDev_transmon):
            qb2_name = qb2.name
        else:
            qb2_name = qb2

        try:
            self.set(f'{gate_name}_{qb1_name}_{qb2_name}_{param}', value)
        except KeyError:
            try:
                self.set(f'{gate_name}_{qb2_name}_{qb1_name}_{param}', value)
            except KeyError:
                raise ValueError(f'Parameter {param} for the gate '
                                 f'{gate_name} {qb1_name} {qb2_name} '
                                 f'does not exist!')

    def add_2qb_gate(self, gate_name, pulse_type='BufferedNZHalfwayPulse'):
        """
        Method to add a two qubit gate with name gate_name with parameters for
        all connected qubits. The parameters including their default values are taken
        for the Class pulse_type in pulse_library.py.

        Args:
            gate_name (str): Name of gate
            pulse_type (str): Two qubit gate class from pulse_library.py
        """

        # add gate to list of two qubit gates
        self.set('two_qb_gates', self.get('two_qb_gates') + [gate_name])

        # for all connected qubits add the operation with name gate_name
        for [qb1, qb2] in self.connectivity_graph():
            op_name = (gate_name, qb1, qb2)
            par_name = f'{gate_name}_{qb1}_{qb2}'
            self.add_operation(op_name)

            # find pulse module
            pulse_func = None
            for module in bpl.pulse_libraries:
                try:
                    pulse_func = getattr(module, pulse_type)
                except AttributeError:
                    pass
            if pulse_func is None:
                raise KeyError('pulse_type {} not recognized'.format(pulse_type))

            # get default pulse params for the pulse type
            params = pulse_func.pulse_params()

            for param, init_val in params.items():
                self.add_pulse_parameter(op_name, par_name + '_' + param, param,
                                         initial_value=init_val)

            # Update flux pulse channels
            for qb, c in zip([qb1, qb2], ['channel', 'channel2']):
                if c in params:
                    channel = self.get_qb(qb).flux_pulse_channel()
                    if channel == '':
                        raise ValueError(f'No flux pulse channel defined for {qb}!')
                    else:
                        self.set_pulse_par(gate_name, qb1, qb2, c, channel)

    # Device Algorithms #

    def measure_J_coupling(self, qbm, qbs, freqs, cz_pulse_name,
                           label=None, cal_points=False, prep_params=None,
                           cal_states='auto', n_cal_points_per_state=1,
                           freq_s=None, f_offset=0, exp_metadata=None,
                           upload=True, analyze=True):

        """
        Measure the J coupling between the qubits qbm and qbs at the interaction
        frequency freq.

        :param qbm:
        :param qbs:
        :param freq:
        :param cz_pulse_name:
        :param label:
        :param cal_points:
        :param prep_params:
        :return:
        """

        if isinstance(qbm, str):
            qbm = self.get_qb(qbm)
        if isinstance(qbs, str):
            qbs = self.get_qb(qbs)

        if label is None:
            label = f'J_coupling_{qbm.name}{qbs.name}'
        MC = self.instr_mc.get_instr()

        for qb in [qbm, qbs]:
            qb.prepare(drive='timedomain')

        if cal_points:
            cal_states = CalibrationPoints.guess_cal_states(cal_states)
            cp = CalibrationPoints.single_qubit(
                qbm.name, cal_states, n_per_state=n_cal_points_per_state)
        else:
            cp = None
        if prep_params is None:
            prep_params = self.get_prep_params([qbm, qbs])

        operation_dict = self.get_operation_dict()

        # Adjust amplitude of stationary qubit
        if freq_s is None:
            freq_s = freqs.mean()

        amp_s = fms.Qubit_freq_to_dac(freq_s,
                                      **qbs.fit_ge_freq_from_flux_pulse_amp())

        fit_paras = qbm.fit_ge_freq_from_flux_pulse_amp()

        amplitudes = fms.Qubit_freq_to_dac(freqs,
                                           **fit_paras)

        amplitudes = np.array(amplitudes)

        if np.any((amplitudes > abs(fit_paras['V_per_phi0']) / 2)):
            amplitudes -= fit_paras['V_per_phi0']
        elif np.any((amplitudes < -abs(fit_paras['V_per_phi0']) / 2)):
            amplitudes += fit_paras['V_per_phi0']

        for [qb1, qb2] in [[qbm, qbs], [qbs, qbm]]:
            operation_dict[cz_pulse_name + f' {qb1.name} {qb2.name}'] \
                ['amplitude2'] = amp_s

        freqs += f_offset

        cz_pulse_name += f' {qbm.name} {qbs.name}'

        seq, sweep_points, sweep_points_2D = \
            fsqs.fluxpulse_amplitude_sequence(
                amplitudes=amplitudes, freqs=freqs, qb_name=qbm.name,
                operation_dict=operation_dict,
                cz_pulse_name=cz_pulse_name, cal_points=cp,
                prep_params=prep_params, upload=False)

        MC.set_sweep_function(awg_swf.SegmentHardSweep(
            sequence=seq, upload=upload, parameter_name='Amplitude', unit='V'))

        MC.set_sweep_points(sweep_points)
        MC.set_sweep_function_2D(swf.Offset_Sweep(
            qbm.instr_ge_lo.get_instr().frequency,
            -qbm.ge_mod_freq(),
            name='Drive frequency',
            parameter_name='Drive frequency', unit='Hz'))
        MC.set_sweep_points_2D(sweep_points_2D)
        MC.set_detector_function(qbm.int_avg_det)
        if exp_metadata is None:
            exp_metadata = {}
        exp_metadata.update({'sweep_points_dict': {qbm.name: amplitudes},
                             'sweep_points_dict_2D': {qbm.name: freqs},
                             'use_cal_points': cal_points,
                             'preparation_params': prep_params,
                             'cal_points': repr(cp),
                             'rotate': cal_points,
                             'data_to_fit': {qbm.name: 'pe'},
                             "sweep_name": "Amplitude",
                             "sweep_unit": "V",
                             "global_PCA": True})
        MC.run_2D(label, exp_metadata=exp_metadata)

        if analyze:
            ma.MeasurementAnalysis(TwoD=True)

    def measure_tomography(self, qubits, prep_sequence, state_name,
                           rots_basis=('I', 'X180', 'Y90', 'mY90', 'X90', 'mX90'),
                           use_cal_points=True,
                           preselection=True,
                           rho_target=None,
                           shots=None,
                           ro_spacing=1e-6,
                           ro_slack=10e-9,
                           thresholded=False,
                           liveplot=True,
                           nreps=1, run=True,
                           upload=True,
                           operation_dict=None):
        exp_metadata = {}

        MC = self.instr_mc.get_instr()

        for qb in qubits:
            qb.prepare(drive='timedomain')

        if operation_dict is None:
            operation_dict = self.get_operation_dict()

        qubit_names = [qb.name for qb in qubits]

        if preselection:
            label = '{}_tomography_ssro_preselection_{}'.format(state_name, '-'.join(
                [qb.name for qb in qubits]))
        else:
            label = '{}_tomography_ssro_{}'.format(state_name, '-'.join(
                [qb.name for qb in qubits]))

        seq_tomo, seg_list_tomo = mqs.n_qubit_tomo_seq(qubit_names,
                                                       operation_dict,
                                                       prep_sequence=prep_sequence,
                                                       rots_basis=rots_basis,
                                                       return_seq=True,
                                                       upload=False,
                                                       preselection=preselection,
                                                       ro_spacing=ro_spacing)
        seg_list = seg_list_tomo

        if use_cal_points:
            seq_cal, seg_list_cal = mqs.n_qubit_ref_all_seq(qubit_names,
                                                            operation_dict,
                                                            return_seq=True,
                                                            upload=False,
                                                            preselection=preselection,
                                                            ro_spacing=ro_spacing)
            # seq += seq_cal
            seg_list += seg_list_cal
        seq = sequence.Sequence(label)
        for seg in seg_list:
            seq.add(seg)

            # reuse sequencer memory by repeating readout pattern
        for qbn in qubit_names:
            seq.repeat_ro(f"RO {qbn}", operation_dict)

        n_segments = seq.n_acq_elements()  # len(seg_list)
        print(n_segments)
        # if preselection:
        #     n_segments *= 2

        # from this point on number of segments is fixed
        sf = awg_swf2.n_qubit_seq_sweep(seq_len=n_segments)

        # shots *= n_segments
        if shots > 1048576:
            shots = 1048576 - 1048576 % n_segments
        # if shots is None:
        #     shots = 4094 - 4094 % n_segments
        # # shots = 600000

        if thresholded:
            df = mqm.get_multiplexed_readout_detector_functions(qubits,
                                                                nr_shots=shots)[
                'dig_log_det']
        else:
            df = mqm.get_multiplexed_readout_detector_functions(qubits,
                                                                nr_shots=shots)[
                'int_log_det']

        # make a channel map
        # fixme - channels and qubits are not always in the same order
        channel_map = {}
        for qb, channel_name in zip(qubits, df.value_names):
            channel_map[qb.name] = channel_name

        # todo Calibration point description code should be a reusable function
        if use_cal_points:
            # calibration definition for all combinations
            cal_defs = []
            for i, name in enumerate(itertools.product("ge", repeat=len(qubits))):
                name = ''.join(name)  # tuple to string
                cal_defs.append({})
                for qb in qubits:
                    if preselection:
                        cal_defs[i][channel_map[qb.name]] = \
                            [2 * len(seg_list) + 2 * i + 1]
                    else:
                        cal_defs[i][channel_map[qb.name]] = \
                            [len(seg_list) + i]
        else:
            cal_defs = None

        exp_metadata["n_segments"] = n_segments
        exp_metadata["rots_basis"] = rots_basis
        if rho_target is not None:
            exp_metadata["rho_target"] = rho_target
        exp_metadata["cal_points"] = cal_defs
        exp_metadata["channel_map"] = channel_map
        exp_metadata["use_preselection"] = preselection

        if upload:
            ps.Pulsar.get_instance().program_awgs(seq)

        MC.live_plot_enabled(liveplot)
        MC.soft_avg(1)
        MC.set_sweep_function(sf)
        MC.set_sweep_points(np.arange(n_segments))
        # MC.set_sweep_function_2D(swf.None_Sweep())
        # MC.set_sweep_points_2D(np.arange(nreps))
        MC.set_detector_function(df)
        if run:
            MC.run(label, exp_metadata=exp_metadata)

        return

    def measure_two_qubit_randomized_benchmarking(
            self, qb1, qb2, cliffords,
            nr_seeds, cz_pulse_name,
            character_rb=False, net_clifford=0,
            clifford_decomposition_name='HZ', interleaved_gate=None,
            n_cal_points_per_state=2, cal_states=tuple(),
            label=None, prep_params=None, upload=True, analyze_RB=True,
            classified=False, correlated=False, thresholded=False, averaged=True):

        qb1n = qb1.name
        qb2n = qb2.name
        qubits = [qb1, qb2]

        if label is None:
            if interleaved_gate is None:
                label = 'RB_{}_{}_seeds_{}_cliffords_{}{}'.format(
                    clifford_decomposition_name, nr_seeds, cliffords[-1],
                    qb1n, qb2n)
            else:
                label = 'IRB_{}_{}_{}_seeds_{}_cliffords_{}{}'.format(
                    interleaved_gate, clifford_decomposition_name, nr_seeds,
                    cliffords[-1], qb1n, qb2n)

        MC = self.instr_mc.get_instr()

        for qb in qubits:
            qb.prepare(drive='timedomain')

        if prep_params is None:
            prep_params = self.get_prep_params([qb1, qb2])

        cal_states = CalibrationPoints.guess_cal_states(cal_states)
        cp = CalibrationPoints.multi_qubit([qb1n, qb2n], cal_states,
                                           n_per_state=n_cal_points_per_state)

        operation_dict = self.get_operation_dict()

        sequences, hard_sweep_points, soft_sweep_points = \
            mqs.two_qubit_randomized_benchmarking_seqs(
                qb1n=qb1n, qb2n=qb2n, operation_dict=operation_dict,
                cliffords=cliffords, nr_seeds=np.arange(nr_seeds),
                max_clifford_idx=24 ** 2 if character_rb else 11520,
                cz_pulse_name=cz_pulse_name + f' {qb1n} {qb2n}',
                net_clifford=net_clifford,
                clifford_decomposition_name=clifford_decomposition_name,
                interleaved_gate=interleaved_gate, upload=False,
                cal_points=cp, prep_params=prep_params)

        hard_sweep_func = awg_swf.SegmentHardSweep(
            sequence=sequences[0], upload=upload,
            parameter_name='Nr. Cliffords', unit='')
        MC.set_sweep_function(hard_sweep_func)
        MC.set_sweep_points(hard_sweep_points if classified else
                            hard_sweep_points * max(qb.acq_shots() for qb in qubits))

        MC.set_sweep_function_2D(awg_swf.SegmentSoftSweep(
            hard_sweep_func, sequences, 'Nr. Seeds', ''))
        MC.set_sweep_points_2D(soft_sweep_points)
        det_get_values_kws = {'classified': classified,
                              'correlated': correlated,
                              'thresholded': thresholded,
                              'averaged': averaged}
        if classified:
            det_type = 'int_avg_classif_det'
            nr_shots = max(qb.acq_averages() for qb in qubits)
        else:
            det_type = 'int_log_det'
            nr_shots = max(qb.acq_shots() for qb in qubits)
        det_func = mqm.get_multiplexed_readout_detector_functions(
            qubits, nr_averages=max(qb.acq_averages() for qb in qubits),
            nr_shots=nr_shots, det_get_values_kws=det_get_values_kws)[det_type]
        MC.set_detector_function(det_func)

        # create sweep points
        sp = SweepPoints('nr_seeds', np.arange(nr_seeds), '', 'Nr. Seeds')
        sp.add_sweep_dimension()
        sp.add_sweep_parameter('cliffords', cliffords, '',
                               'Number of applied Cliffords, $m$')

        # create analysis pipeline object
        meas_obj_value_names_map = mqm.get_meas_obj_value_names_map(qubits,
                                                                    det_func)
        mobj_names = list(meas_obj_value_names_map)
        pp = ProcessingPipeline(meas_obj_value_names_map)
        for i, mobjn in enumerate(mobj_names):
            pp.add_node(
                'average_data', keys_in='raw',
                shape=(len(cliffords), nr_seeds), meas_obj_names=[mobjn])
            pp.add_node(
                'get_std_deviation', keys_in='raw',
                shape=(len(cliffords), nr_seeds), meas_obj_names=[mobjn])
            pp.add_node(
                'SingleQubitRBAnalysis', keys_in='previous average_data',
                std_keys='previous get_std_deviation',
                meas_obj_names=[mobjn], plot_T1_lim=False, d=4)
        # create experimental metadata
        exp_metadata = {'preparation_params': prep_params,
                        'cal_points': repr(cp),
                        'sweep_points': sp,
                        'meas_obj_sweep_points_map':
                            {qbn: ['nr_seeds', 'cliffords'] for
                             qbn in mobj_names},
                        'meas_obj_value_names_map': meas_obj_value_names_map,
                        'processing_pipe': pp}
        MC.run_2D(name=label, exp_metadata=exp_metadata)

        if analyze_RB:
            pla.PipelineDataAnalysis()

    def measure_chevron(self, qbc, qbt, hard_sweep_params, soft_sweep_params,
                        cz_pulse_name=None, upload=True, label=None, qbr=None,
                        classified=False, n_cal_points_per_state=2,
                        num_cz_gates=1, cal_states='auto', prep_params=None,
                        exp_metadata=None, analyze=True, return_seq=False):

        if qbr is None:
            qbr = qbt
        elif qbr != qbc and qbr != qbt:
            raise ValueError('Only target or control qubit can be read out!')

        if len(list(soft_sweep_params)) > 1:
            log.warning('There is more than one soft sweep parameter.')
        if label is None:
            label = 'Chevron_{}{}'.format(qbc.name, qbt.name)
        MC = self.find_instrument('MC')
        for qb in [qbc, qbt]:
            qb.prepare(drive='timedomain')

        if cz_pulse_name is None:
            cz_pulse_name = f'FP {qbc.name}'
        else:
            cz_pulse_name += f' {qbc.name} {qbt.name}'

        cal_states = CalibrationPoints.guess_cal_states(cal_states)
        cp = CalibrationPoints.single_qubit(qbr.name, cal_states,
                                            n_per_state=n_cal_points_per_state)

        if prep_params is None:
            prep_params = self.get_prep_params([qbc, qbt])

        operation_dict = self.get_operation_dict()

        sequences, hard_sweep_points, soft_sweep_points = \
            fsqs.chevron_seqs(
                qbc_name=qbc.name, qbt_name=qbt.name,
                hard_sweep_dict=hard_sweep_params,
                soft_sweep_dict=soft_sweep_params,
                operation_dict=operation_dict,
                cz_pulse_name=cz_pulse_name,
                num_cz_gates=num_cz_gates,
                cal_points=cp, upload=False, prep_params=prep_params)

        if return_seq:
            return sequences

        hard_sweep_func = awg_swf.SegmentHardSweep(
            sequence=sequences[0], upload=upload,
            parameter_name=list(hard_sweep_params)[0],
            unit=list(hard_sweep_params.values())[0]['unit'])
        MC.set_sweep_function(hard_sweep_func)
        MC.set_sweep_points(hard_sweep_points)

        # sweep over flux pulse amplitude of qbc
        channels_to_upload = [qbc.flux_pulse_channel()]
        MC.set_sweep_function_2D(awg_swf.SegmentSoftSweep(
            hard_sweep_func, sequences,
            list(soft_sweep_params)[0], list(soft_sweep_params.values())[0]['unit'],
            channels_to_upload=channels_to_upload))
        MC.set_sweep_points_2D(soft_sweep_points)
        MC.set_detector_function(qbr.int_avg_classif_det if classified
                                 else qbr.int_avg_det)
        if exp_metadata is None:
            exp_metadata = {}
        exp_metadata.update({'preparation_params': prep_params,
                             'cal_points': repr(cp),
                             'rotate': len(cal_states) != 0,
                             'data_to_fit': {qbr.name: 'pe'},
                             'hard_sweep_params': hard_sweep_params,
                             'soft_sweep_params': soft_sweep_params})
        MC.run_2D(name=label, exp_metadata=exp_metadata)

        if analyze:
            tda.MultiQubit_TimeDomain_Analysis(qb_names=[qbr.name],
                                               options_dict={'TwoD': True})

    def measure_cphase(self, qbc, qbt, soft_sweep_params, cz_pulse_name=None, prep_params=None,
                       label=None, cal_states='auto', n_cal_points_per_state=1,
                       num_cz_gates=1, hard_sweep_params=None, exp_metadata=None,
                       analyze=True, upload=True, for_ef=True, **kw):

        qbc_name = qbc.name
        qbt_name = qbt.name

        if [qbc_name, qbt_name] not in self.connectivity_graph() and [qbt_name,
                                                                      qbc_name] not in self.connectivity_graph():
            raise ValueError('Qubits are not connected!')

        if cz_pulse_name is None:
            cz_pulse_name = f'FP {qbc.name}'
        else:
            cz_pulse_name += f' {qbc.name} {qbt.name}'

        MC = self.instr_mc.get_instr()

        plot_all_traces = kw.get('plot_all_traces', True)
        plot_all_probs = kw.get('plot_all_probs', True)
        classified = kw.get('classified', False)
        predictive_label = kw.pop('predictive_label', False)

        if prep_params is None:
            prep_params = self.get_prep_params([qbc, qbt])

        if label is None:
            if predictive_label:
                label = 'Predictive_cphase_nz_measurement'
            else:
                label = 'CPhase_nz_measurement'
            if classified:
                label += '_classified'
            if 'active' in prep_params['preparation_type']:
                label += '_reset'
            if num_cz_gates > 1:
                label += f'_{num_cz_gates}_gates'
            label += f'_{qbc.name}_{qbt.name}'

        if hard_sweep_params is None:
            hard_sweep_params = {
                'phase': {'values': np.tile(np.linspace(0, 2 * np.pi, 6) * 180 / np.pi, 2),
                          'unit': 'deg'}
            }

        for qb in [qbc, qbt]:
            qb.prepare(drive='timedomain')

        cal_states = CalibrationPoints.guess_cal_states(cal_states,
                                                        for_ef=for_ef)
        cp = CalibrationPoints.multi_qubit([qbc.name, qbt.name], cal_states,
                                           n_per_state=n_cal_points_per_state)

        operation_dict = self.get_operation_dict()
        sequences, hard_sweep_points, soft_sweep_points = \
            fsqs.cphase_seqs(
                hard_sweep_dict=hard_sweep_params,
                soft_sweep_dict=soft_sweep_params,
                qbc_name=qbc.name, qbt_name=qbt.name,
                cz_pulse_name=cz_pulse_name,
                operation_dict=operation_dict,
                cal_points=cp, upload=False, prep_params=prep_params,
                num_cz_gates=num_cz_gates)

        hard_sweep_func = awg_swf.SegmentHardSweep(
            sequence=sequences[0], upload=upload,
            parameter_name=list(hard_sweep_params)[0],
            unit=list(hard_sweep_params.values())[0]['unit'])
        MC.set_sweep_function(hard_sweep_func)
        MC.set_sweep_points(hard_sweep_points)

        channels_to_upload = [operation_dict[cz_pulse_name]['channel']]
        MC.set_sweep_function_2D(awg_swf.SegmentSoftSweep(
            hard_sweep_func, sequences,
            list(soft_sweep_params)[0], list(soft_sweep_params.values())[0]['unit'],
            channels_to_upload=channels_to_upload))
        MC.set_sweep_points_2D(soft_sweep_points)

        det_get_values_kws = {'classified': classified,
                              'correlated': False,
                              'thresholded': False,
                              'averaged': True}
        det_name = 'int_avg{}_det'.format('_classif' if classified else '')
        det_func = mqm.get_multiplexed_readout_detector_functions(
            [qbc, qbt], nr_averages=max(qb.acq_averages() for qb in [qbc, qbt]),
            det_get_values_kws=det_get_values_kws)[det_name]
        MC.set_detector_function(det_func)

        if exp_metadata is None:
            exp_metadata = {}
        exp_metadata.update({'leakage_qbname': qbc.name,
                             'cphase_qbname': qbt.name,
                             'preparation_params': prep_params,
                             'cal_points': repr(cp),
                             'classified_ro': classified,
                             'rotate': len(cal_states) != 0 and not classified,
                             'cal_states_rotations':
                                 {qbc.name: {'g': 0, 'f': 1},
                                  qbt.name: {'g': 0, 'e': 1}} if
                                 (len(cal_states) != 0 and not classified
                                  and for_ef) else None,
                             'data_to_fit': {qbc.name: 'pf', qbt.name: 'pe'},
                             'hard_sweep_params': hard_sweep_params,
                             'soft_sweep_params': soft_sweep_params})
        MC.run_2D(label, exp_metadata=exp_metadata)
        if analyze:
            if classified:
                channel_map = {qb.name: [vn + ' ' +
                                         qb.instr_uhf() for vn in
                                         qb.int_avg_classif_det.value_names]
                               for qb in [qbc, qbt]}
            else:
                channel_map = {qb.name: [vn + ' ' +
                                         qb.instr_uhf() for vn in
                                         qb.int_avg_det.value_names]
                               for qb in [qbc, qbt]}
            flux_pulse_tdma = tda.CPhaseLeakageAnalysis(
                qb_names=[qbc.name, qbt.name],
                options_dict={'TwoD': True, 'plot_all_traces': plot_all_traces,
                              'plot_all_probs': plot_all_probs,
                              'channel_map': channel_map})
            cphases = flux_pulse_tdma.proc_data_dict[
                'analysis_params_dict']['cphase']['val']
            population_losses = flux_pulse_tdma.proc_data_dict[
                'analysis_params_dict']['population_loss']['val']
            leakage = flux_pulse_tdma.proc_data_dict[
                'analysis_params_dict']['leakage']['val']
            return cphases, population_losses, leakage, flux_pulse_tdma
        else:
            return

    def measure_dynamic_phases(self, qbc, qbt, cz_pulse_name, hard_sweep_params=None,
                               qubits_to_measure=None, cal_points=True,
                               analyze=True, upload=True, n_cal_points_per_state=1,
                               cal_states='auto', prep_params=None,
                               exp_metadata=None, classified=False, update=False,
                               reset_phases_before_measurement=True,
                               prepend_n_cz=0):

        if qubits_to_measure is None:
            qubits_to_measure = [qbc, qbt]

        if reset_phases_before_measurement:
            dyn_phases = {qb.name: 0 for qb in qubits_to_measure}
            self.get_pulse_par(cz_pulse_name,
                               qbc, qbt, 'basis_rotation')(dyn_phases)

        if hard_sweep_params is None:
            hard_sweep_params = {
                'phase': {
                    'values': np.tile(np.linspace(0, 2 * np.pi, 6) * 180 / np.pi, 2),
                    'unit': 'deg'}}
        qbc_name = qbc.name
        qbt_name = qbt.name

        if [qbc_name, qbt_name] not in self.connectivity_graph() and [qbt_name,
                                                                      qbc_name] not in self.connectivity_graph():
            raise ValueError('Qubits are not connected!')

        if prep_params is None:
            prep_params = self.get_prep_params([qbc, qbt])

        for qb in qubits_to_measure:
            label = f'Dynamic_phase_measurement_CZ{qbt.name}{qbc.name}-{qb.name}'
            qb.prepare(drive='timedomain')
            MC = qbc.instr_mc.get_instr()

            if cal_points:
                cal_states = CalibrationPoints.guess_cal_states(cal_states)
                cp = CalibrationPoints.single_qubit(
                    qb.name, cal_states, n_per_state=n_cal_points_per_state)
            else:
                cp = None

            seq, hard_sweep_points = \
                fsqs.dynamic_phase_seq(
                    qb_name=qb.name, hard_sweep_dict=hard_sweep_params,
                    operation_dict=self.get_operation_dict(),
                    cz_pulse_name=cz_pulse_name + f' {qbc.name} {qbt.name}',
                    cal_points=cp,
                    prepend_n_cz=prepend_n_cz,
                    upload=False, prep_params=prep_params)

            MC.set_sweep_function(awg_swf.SegmentHardSweep(
                sequence=seq, upload=upload,
                parameter_name=list(hard_sweep_params)[0],
                unit=list(hard_sweep_params.values())[0]['unit']))
            MC.set_sweep_points(hard_sweep_points)
            MC.set_detector_function(qb.int_avg_classif_det if classified
                                     else qb.int_avg_det)
            if exp_metadata is None:
                exp_metadata = {}
            exp_metadata.update({'use_cal_points': cal_points,
                                 'preparation_params': prep_params,
                                 'cal_points': repr(cp),
                                 'rotate': cal_points,
                                 'data_to_fit': {qb.name: 'pe'},
                                 'cal_states_rotations':
                                     {qb.name: {'g': 0, 'e': 1}},
                                 'hard_sweep_params': hard_sweep_params})
            MC.run(label, exp_metadata=exp_metadata)

            if analyze:
                MA = tda.CZDynamicPhaseAnalysis(qb_names=[qb.name], options_dict={
                    'flux_pulse_length': self.get_pulse_par(cz_pulse_name,
                                                            qbc, qbt,
                                                            'pulse_length')(),
                    'flux_pulse_amp': self.get_pulse_par(cz_pulse_name,
                                                         qbc, qbt,
                                                         'amplitude')(), })
                dyn_phases[qb.name] = \
                    MA.proc_data_dict['analysis_params_dict'][qb.name][
                        'dynamic_phase']['val'] * 180 / np.pi
        if update and reset_phases_before_measurement:
            self.get_pulse_par(cz_pulse_name,
                               qbc, qbt, 'basis_rotation')(dyn_phases)
        return dyn_phases

    def calibrate_device(self, qubits=None, from_ts=None, benchmark=True,
                         repark=True, for_ef=True):

        if repark:
            def freq_func(V, V0, f0, fv):
                return f0 - fv * (V - V0) ** 2

        if from_ts is None:
            timestamp = self.last_calib()

        if qubits is None:
            qubits = [self.get_qb(qb_name) for qb_name in self.qubits()]

        for qubit in qubits:
            gen.load_settings(qubit, timestamp=timestamp)
            # FIXME: also reload flux parameters

        cp = self.calibration_parameters()

        for qubit in qubits:
            qubit.preparation_params(dict(preparation_type='wait'))

            if benchmark:
                # If benchmark is True the values for T1 and T2 are
                # compared before and after calibration

                qubit.find_T1(np.linspace(0, cp['T1_time'], cp['data_points']),
                              update=True, upload=True)
                qubit.find_T2_echo(
                    np.linspace(0, cp['long_ram_freq'], cp['data_points']),
                    artificial_detuning=cp['long_ram_det'], update=True, upload=True)

            # Find new amp180 (rough)
            qubit.find_amplitudes(np.linspace(0, cp['rabi_amp'], cp['data_points']),
                                  upload=True, update=True)

            # Find new qubit frequency
            qubit.find_frequency_T2_ramsey(np.linspace(0, cp['short_ram_freq'], cp['data_points']),
                                           artificial_detunings=cp['short_ram_det'],
                                           update=True, upload=True)

            if repark:
                fluxline = self.fluxline_dict()[qubit.name]
                voltages = fluxline() + np.linspace(
                    -cp['flux_sweep'], cp['flux_sweep'], cp['flux_sweep_points'])
                freqs = []
                for i, volt in enumerate(voltages):
                    fluxline(volt)
                    qubit.find_frequency_T2_ramsey(
                        np.linspace(0, cp['short_ram_freq'], cp['data_points']),
                        artificial_detunings=cp['short_ram_det'],
                        update=True, upload=(i == 0))
                    freqs.append(qubit.ge_freq())

                freq_fit, _ = opti.curve_fit(freq_func, voltages, np.array(freqs) / 1e9)
                self.instr_dc_source.get_instr().set_smooth(
                    {self.fluxline_map()[qubit.name]: freq_fit[0]})

            # Find new qubit frequency
            qubit.find_frequency_T2_ramsey(np.linspace(0, cp['short_ram_freq'], cp['data_points']),
                                           artificial_detunings=cp['short_ram_det'],
                                           update=True, upload=True)

            # Find new amp180 (exact)
            n = cp['rabi_n']
            amp180 = qubit.ge_amp180()
            amps = np.linspace((n - 1) * amp180 / n,
                               min((n + 1) * amp180 / n, cp['rabi_amp']),
                               cp['data_points'])
            qubit.find_amplitudes(amps, upload=True, n=n, update=True)

            # Qscale
            qubit.find_qscale(qscales=np.linspace(-cp['qscale'], cp['qscale'],
                                                  cp['data_points']),
                              update=True, upload=True)

            if for_ef:
                qubit.find_amplitudes(
                    np.linspace(0, cp['rabi_amp'], 21), upload=True, update=True,
                    for_ef=True)
                qubit.find_frequency_T2_ramsey(
                    np.linspace(0, cp['ef_ram_freq'], cp['data_points']),
                    artificial_detunings=cp['ef_ram_det'],
                    update=True, upload=True)
                qubit.find_amplitudes(
                    np.linspace(0, cp['rabi_amp'], 21), upload=True, update=True,
                    for_ef=True)

            qubit.find_T1(np.linspace(0, cp['T1_time'], cp['data_points']),
                          update=True, upload=True)
            qubit.find_T2_echo(
                np.linspace(0, cp['long_ram_freq'], cp['data_points']),
                artificial_detuning=cp['long_ram_det'], update=True, upload=True)

            if benchmark:
                pass
                # FIXME: enable benchmarking improvement of T1 and T2

            return
