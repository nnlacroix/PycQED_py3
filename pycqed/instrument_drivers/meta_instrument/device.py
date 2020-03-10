# general imports
import logging
from pycqed.utilities import general as gen

import pycqed.measurement.pulse_sequences.fluxing_sequences as fsqs
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import scipy.optimize as opti

# PycQED imports
from pycqed.instrument_drivers.meta_instrument.qubit_objects import QuDev_transmon as QuDev_transmon
from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import (
    ManualParameter, InstrumentRefParameter)
from qcodes.utils import validators as vals
from pycqed.measurement.calibration_points import CalibrationPoints
from pycqed.analysis import measurement_analysis as ma
from pycqed.measurement import sweep_functions as swf
import pycqed.measurement.awg_sweep_functions as awg_swf
import pycqed.analysis_v2.timedomain_analysis as tda
import pycqed.measurement.multi_qubit_module as mqm
import pycqed.analysis.fitting_models as fms


log = logging.getLogger(__name__)


class Device(Instrument):
    def __init__(self, name, qubits, connections, **kw):
        super().__init__(name, **kw)

        qb_names = [qb.name for qb in qubits]
        connectivity_graph = [[qb1.name, qb2.name] for [qb1, qb2] in connections]

        self._operations = {}

        self.add_parameter('instr_mc',
                           parameter_class=InstrumentRefParameter)
        self.add_parameter('instr_pulsar',
                           parameter_class=InstrumentRefParameter)
        self.add_parameter('instr_dc_source',
                           parameter_class=InstrumentRefParameter)
        self.add_parameter('instr_trigger',
                           parameter_class=InstrumentRefParameter)

        self.add_parameter('qubits',
                           vals=vals.Lists(),
                           initial_value=qb_names,
                           parameter_class=ManualParameter)
        for qb_name in self.qubits():
            self.add_parameter(f'{qb_name}',
                               parameter_class=InstrumentRefParameter,
                               initial_value=f'{qb_name}')

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
                           parameter_class=ManualParameter)

        # Pulse preparation parameters
        default_prep_params = dict(preparation_type='wait',
                                   post_ro_wait=1e-6, reset_reps=1)

        self.add_parameter('preparation_params', parameter_class=ManualParameter,
                           initial_value=default_prep_params, vals=vals.Dict())

    # General Class Methods

    def add_operation(self, operation_name):
        self._operations[operation_name] = {}

    def _get_operations(self):
        return self._operations

    def get_operation_dict(self, operation_dict=None):
        if operation_dict is None:
            operation_dict = dict()

        # add 2qb operations
        two_qb_operation_dict = {}
        for op_tag, op in self.operations().items():
            # Add both qubit combinations to operations dict
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
        return self.find_instrument(qb_name)

    def get_pulse_par(self, pulse_name, qb1, qb2, param):
        qb1_name = qb1.name
        qb2_name = qb2.name
        try:
            return self.__dict__['parameters'] \
                [f'{pulse_name}_{qb1_name}_{qb2_name}_{param}']
        except KeyError:
            try:
                return self.__dict__['parameters'] \
                    [f'{pulse_name}_{qb2_name}_{qb1_name}_{param}']
            except KeyError:
                raise ValueError(f'Parameter {param} for the gate '
                                 f'{pulse_name} {qb1_name} {qb2_name} '
                                 f'does not exist!')

    def set_pulse_par(self, pulse_name, qb1, qb2, param, value):
        qb1_name = qb1.name
        qb2_name = qb2.name

        try:
            self.__dict__['parameters'] \
                [f'{pulse_name}_{qb1_name}_{qb2_name}_{param}'] = value
        except KeyError:
            try:
                self.__dict__['parameters'] \
                    [f'{pulse_name}_{qb2_name}_{qb1_name}_{param}'] = value
            except KeyError:
                raise ValueError(f'Parameter {param} for the gate '
                                 f'{pulse_name} {qb1_name} {qb2_name} '
                                 f'does not exist!')

    def add_pulse_parameter(self,
                            operation_name,
                            parameter_name,
                            argument_name,
                            initial_value=None,
                            **kw):
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

        return

    def get_prep_params(self, qb_list):
        thresh_map = {}
        for prep_params in [qb.preparation_params() for qb in qb_list]:
            if 'threshold_mapping' in prep_params:
                thresh_map.update(prep_params['threshold_mapping'])

        prep_params = deepcopy(self.preparation_params())
        prep_params['threshold_mapping'] = thresh_map

        return prep_params

    # Two Qubit Gates

    def add_halfway_pulse(self, gate_name):

        self.set('two_qb_gates', self.get('two_qb_gates') + [gate_name])

        for [qb1, qb2] in self.connectivity_graph():
            # op_name = f'{gate_name} {qb1} {qb2}'
            op_name = (gate_name, qb1, qb2)
            par_name = f'{gate_name}_{qb1}_{qb2}'
            self.add_operation(op_name)

            self.add_pulse_parameter(op_name, par_name + '_pulse_type', 'pulse_type',
                                     initial_value='BufferedHalfwayPulse',
                                     vals=vals.Enum('BufferedHalfwayPulse',
                                                    'BufferedNZHalfwayPulse'))

            qb1_obj = self.get_qb(qb1)
            qb2_obj = self.get_qb(qb2)
            if qb1_obj.flux_pulse_channel() == '' or \
                    qb2_obj.flux_pulse_channel() == '':
                raise ValueError(f'No flux pulse channel defined for'
                                 f' {qb1} or {qb2}!')
            self.add_pulse_parameter(op_name, par_name + '_channel', 'channel',
                                     initial_value=qb1_obj.flux_pulse_channel(),
                                     vals=vals.Strings())
            self.add_pulse_parameter(op_name, par_name + '_channel2', 'channel2',
                                     initial_value=qb2_obj.flux_pulse_channel(),
                                     vals=vals.Strings())
            self.add_pulse_parameter(op_name, par_name + '_aux_channels_dict',
                                     'aux_channels_dict',
                                     initial_value={}, vals=vals.Dict())
            self.add_pulse_parameter(op_name, par_name + '_amplitude', 'amplitude',
                                     initial_value=0, vals=vals.Numbers())
            self.add_pulse_parameter(op_name, par_name + '_amplitude2', 'amplitude2',
                                     initial_value=0, vals=vals.Numbers())
            self.add_pulse_parameter(op_name, par_name + '_pulse_length',
                                     'pulse_length',
                                     initial_value=0, vals=vals.Numbers(0))
            self.add_pulse_parameter(op_name, par_name + '_alpha', 'alpha',
                                     initial_value=1, vals=vals.Numbers())
            self.add_pulse_parameter(op_name, par_name + '_alpha2', 'alpha2',
                                     initial_value=1, vals=vals.Numbers())
            self.add_pulse_parameter(op_name, par_name + '_buffer_length_start',
                                     'buffer_length_start', initial_value=30e-9,
                                     vals=vals.Numbers(0))
            self.add_pulse_parameter(op_name, par_name + '_buffer_length_end',
                                     'buffer_length_end', initial_value=30e-9,
                                     vals=vals.Numbers(0))
            self.add_pulse_parameter(op_name, par_name + '_flux_buffer_length',
                                     'flux_buffer_length', initial_value=0,
                                     vals=vals.Numbers(0))
            self.add_pulse_parameter(op_name, par_name + '_flux_buffer_length2',
                                     'flux_buffer_length2', initial_value=0,
                                     vals=vals.Numbers(0))
            self.add_pulse_parameter(op_name, par_name + '_extra_buffer_aux_pulse',
                                     'extra_buffer_aux_pulse', initial_value=5e-9,
                                     vals=vals.Numbers(0))
            self.add_pulse_parameter(op_name, par_name + '_pulse_delay',
                                     'pulse_delay',
                                     initial_value=0, vals=vals.Numbers())
            self.add_pulse_parameter(op_name, par_name + '_channel_relative_delay',
                                     'channel_relative_delay',
                                     initial_value=0, vals=vals.Numbers())
            self.add_pulse_parameter(op_name, par_name + '_gaussian_filter_sigma',
                                     'gaussian_filter_sigma', initial_value=1e-9,
                                     vals=vals.Numbers(0))

    def add_nz_cz_pulse(self, gate_name, symmetric=True):
        raise NotImplementedError('NZ CZ pulse has not been implemented!')

    # Device Algorithms

    def measure_J_coupling(self, qbm, qbs , freqs, cz_pulse_name=None,
                           label=None, cal_points=False, prep_params=None,
                           cal_states='auto', n_cal_points_per_state=1,
                           freq_s=None, exp_metadata=None, upload=True,
                           analyze=True):

        """
        Measure the J coupling between the qubits qbm and qbm at the interaction
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

        if cz_pulse_name is None:
            raise ValueError('Provide a cz_pulse_name!')

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

        for [qb1, qb2] in [[qbm, qbs],[qbs,qbm]]:
            operation_dict[cz_pulse_name+f' {qb1.name} {qb2.name}']\
                ['amplitude2'] = amp_s


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

    def measure_chevron(self, qbc, qbt, hard_sweep_params, soft_sweep_params,
                        cz_pulse_name=None, upload=True, label=None, qbr=None,
                        classified=False, n_cal_points_per_state=2,
                        cal_states='auto', prep_params=None,
                        exp_metadata=None, analyze=True):

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
                cal_points=cp, upload=False, prep_params=prep_params)

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

        if [qbc_name, qbt_name] not in self.connectivity_graph() and [qbt_name, qbc_name] not in self.connectivity_graph():
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
