import numpy as np
from copy import copy
from copy import deepcopy
from itertools import zip_longest
import traceback
from pycqed.utilities.general import temporary_value
from pycqed.measurement.calibration.two_qubit_gates import CalibBuilder
from pycqed.measurement.waveform_control.block import Block, ParametricValue
from pycqed.measurement.waveform_control.segment import UnresolvedPulse
from pycqed.measurement.sweep_points import SweepPoints
from pycqed.analysis import fitting_models as fit_mods
from pycqed.measurement.calibration.calibration_points import CalibrationPoints
import pycqed.measurement.awg_sweep_functions as awg_swf
import pycqed.analysis_v2.timedomain_analysis as tda
from pycqed.instrument_drivers.meta_instrument.qubit_objects.QuDev_transmon \
    import QuDev_transmon
from pycqed.measurement.multi_qubit_module import \
    get_multiplexed_readout_detector_functions
import logging
log = logging.getLogger(__name__)


class T1FrequencySweep(CalibBuilder):
    def __init__(self, dev, task_list=None, sweep_points=None,
                 qubits=None, **kw):
        """

        :param dev:
        :param task_list:
        :param sweep_points:
        :param qubits:
        :param kw:

        Assumptions:
         - assumes there is one task for each qubit. If task_list is None, it
          will internally create it.

        """
        # try:
        if task_list is None:
            if sweep_points is None or qubits is None:
                raise ValueError('Please provide either "sweep_points" '
                                 'and "qubits," or "task_list" containing '
                                 'this information.')
            task_list = [{'qubits_to_measure': qb.name,
                          'sweep_points': sweep_points} for qb in qubits]
        if qubits is None:
            qubits = self.find_qubits_in_tasks(dev.qubits(), task_list)
        super().__init__(dev, qubits=qubits, **kw)

        self.analysis = None
        task_list = self.add_amplitude_sweep_points(task_list)
        self.task_list = task_list
        self.ro_qubits = self.get_ro_qubits()
        self.guess_label(**kw)
        self.data_to_fit = {qb.name: 'pe' for qb in self.ro_qubits}
        for_ef = kw.get('for_ef', False)
        kw['for_ef'] = for_ef
        self.cal_points = self.get_cal_points(**kw)
        sweep_points = SweepPoints(from_dict_list=[{}, {}])
        self.sequences, sp = \
            self.parallel_sweep(sweep_points, self.task_list,
                                self.t1_flux_pulse_block, **kw)
        self.hard_sweep_points, self.soft_sweep_points = sp
        self.exp_metadata.update({
            'data_to_fit': self.data_to_fit,
            'global_PCA': len(self.cal_points.states) == 0
        })

        if self.measure:
            # compress the 2nd sweep dimension completely onto the first
            self.run_measurement(
                compression_seg_lim=np.product([len(s) for s in sp]) +
                                    len(self.cal_points.states), **kw)
        if self.analyze:
            self.run_analysis(**kw)
        # except Exception as x:
        #     self.exception = x
        #     traceback.print_exc()

    def add_amplitude_sweep_points(self, task_list):
        for task in task_list:
            this_qb = self.get_qubits(task['qubits_to_measure'])[0][0]
            sweep_points = task['sweep_points']
            if len(sweep_points) == 1:
                raise NotImplementedError('Sweep points must be two-dimensional'
                                          ' with dim 1 over flux pulse lengths,'
                                          ' and dim 2 over qubit frequencies'
                                          ' or flux pulse amplitudes.')
                # # it must be the 1st sweep dimension, over flux pulse lengths
                # sweep_points = [sweep_points[0], {}]
                # sweep_points = SweepPoints(from_dict_list=sweep_points)

            if next(iter(sweep_points[1].values()))[1].lower() == 'hz':
                fit_paras = deepcopy(this_qb.fit_ge_freq_from_flux_pulse_amp())
                if len(fit_paras) == 0:
                    raise ValueError(f'fit_ge_freq_from_flux_pulse_amp is empty'
                                     f' for {this_qb.name}. Cannot calculate '
                                     f'amplitudes from qubit frequencies.')
                amplitudes = np.array(fit_mods.Qubit_freq_to_dac(
                    next(iter(sweep_points[1].values()))[0], **fit_paras))
                if np.any((amplitudes > abs(fit_paras['V_per_phi0']) / 2)):
                    amplitudes -= fit_paras['V_per_phi0']
                elif np.any((amplitudes < -abs(fit_paras['V_per_phi0']) / 2)):
                    amplitudes += fit_paras['V_per_phi0']
                if np.any(np.isnan(amplitudes)):
                    raise ValueError('Specified frequencies resulted in nan '
                                     'amplitude. Check frequency range!')
                amp_sweep_points = SweepPoints('amplitude', amplitudes,
                                               'V', 'Flux pulse amplitude')
                sweep_points.update([{}] + amp_sweep_points)
            task['sweep_points'] = sweep_points
        return task_list

    def add_default_sweep_points(self, sweep_points, **kw):
        sweep_points = self.add_default_ramsey_sweep_points(sweep_points, **kw)
        nr_phases = len(list(sweep_points[0].values())[0][0]) // 2
        hard_sweep_dict = SweepPoints(
            'pi_pulse_off', [0] * nr_phases + [1] * nr_phases)
        sweep_points.update(hard_sweep_dict + [{}])
        return sweep_points

    def t1_flux_pulse_block(self, qubits_to_measure, sweep_points,
                            prepend_pulse_dicts=None, **kw):
        """
        TODO
        :param cz_pulse_name: task-specific prefix of CZ gates (overwrites
            global choice passed to the class init)
        ...
        """

        qubit_name = qubits_to_measure
        hard_sweep_dict, soft_sweep_dict = sweep_points
        pb = self.prepend_pulses_block(prepend_pulse_dicts)

        pulse_modifs = {'all': {'element_name': 'pi_pulse'}}
        pp = self.block_from_ops('pipulse',
                                 [f'X180 {qubit_name}'] +
                                 kw.get('spectator_op_codes', []),
                                 pulse_modifs=pulse_modifs)

        pulse_modifs = {'all': {'element_name': 'flux_pulse', 'pulse_delay': 0}}
        fp = self.block_from_ops('flux', [f'FP {qubit_name}'],
                                 pulse_modifs=pulse_modifs)
        for k in ['channel', 'channel2']:
            if k in fp.pulses[0]:
                if fp.pulses[0][k] not in self.channels_to_upload:
                    self.channels_to_upload.append(fp.pulses[0][k])
        for k in hard_sweep_dict:
            for p in fp.pulses:
                if k in p:
                    p[k] = ParametricValue(k)
        for k in soft_sweep_dict:
            for p in fp.pulses:
                if k in p:
                    p[k] = ParametricValue(k)

        return self.sequential_blocks(f't1 flux pulse {qubit_name}',
                                      [pb, pp, fp])

    def guess_label(self, **kw):
        if self.label is None:
            self.label = f'T1_frequency_sweep' \
                         f'{self.dev.get_msmt_suffix(self.ro_qubits)}'

    def run_analysis(self, **kw):
        self.all_fits = kw.get('all_fits', True)
        self.analysis = tda.T1FrequencySweepAnalysis(
            qb_names=[qb.name for qb in self.ro_qubits],
            options_dict=dict(TwoD=False, all_fits=self.all_fits))