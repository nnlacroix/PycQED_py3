import numpy as np
from copy import copy
from copy import deepcopy
import traceback
from pycqed.measurement.calibration.two_qubit_gates import CalibBuilder
import pycqed.measurement.sweep_functions as swf
from pycqed.measurement.waveform_control.block import Block, ParametricValue
from pycqed.measurement.sweep_points import SweepPoints
from pycqed.analysis import fitting_models as fit_mods
import pycqed.analysis_v2.timedomain_analysis as tda
from pycqed.analysis import measurement_analysis as ma
from pycqed.utilities.general import temporary_value
import logging

log = logging.getLogger(__name__)


class T1FrequencySweep(CalibBuilder):
    def __init__(self, task_list=None, sweep_points=None, qubits=None, **kw):
        """
        Flux pulse amplitude measurement used to determine the qubits energy in
        dependence of flux pulse amplitude.

        Timings of sequence

       |          ---|X180|  ------------------------------|RO|
       |          --------| --------- fluxpulse ---------- |


        :param task_list: list of dicts; see CalibBuilder docstring
        :param sweep_points: SweepPoints class instance with first sweep
            dimension describing the flux pulse lengths and second dimension
            either the flux pulse amplitudes, qubit frequencies, or both.
            !!! If both amplitudes and frequencies are provided, they must be
            be specified in the order amplitudes, frequencies as shown:
            [{'pulse_length': (lengths, 's', 'Flux pulse length')},
             {'flux_pulse_amp': (amps, 'V', 'Flux pulse amplitude'),
              'qubit_freqs': (freqs, 'Hz', 'Qubit frequency')}]
            If this parameter is provided it will be used for all qubits.
        :param qubits: list of QuDev_transmon class instances
        :param kw: keyword arguments
            for_ef (bool, default: False): passed to get_cal_points; see
                docstring there.
            spectator_op_codes (list, default: []): see t1_flux_pulse_block
            all_fits (bool, default: True) passed to run_analysis; see
                docstring there

        Assumptions:
         - assumes there is one task for each qubit. If task_list is None, it
          will internally create it.
         - the entry "qb" in each task should contain one qubit name.

        """
        try:
            self.experiment_name = 'T1_frequency_sweep'
            if task_list is None:
                if sweep_points is None or qubits is None:
                    raise ValueError('Please provide either "sweep_points" '
                                     'and "qubits," or "task_list" containing '
                                     'this information.')
                task_list = [{'qb': qb.name} for qb in qubits]
            for task in task_list:
                if not isinstance(task['qb'], str):
                    task['qb'] = task['qb'].name

            super().__init__(task_list, qubits=qubits,
                             sweep_points=sweep_points, **kw)

            self.analysis = None
            self.data_to_fit = {qb: 'pe' for qb in self.meas_obj_names}
            self.sweep_points = SweepPoints(
                from_dict_list=[{}, {}] if self.sweep_points is None
                else self.sweep_points)
            self.add_amplitude_sweep_points()

            self.preprocessed_task_list = self.preprocess_task_list(**kw)
            self.sequences, self.mc_points = \
                self.parallel_sweep(self.preprocessed_task_list,
                                    self.t1_flux_pulse_block, **kw)
            self.exp_metadata.update({
                'global_PCA': len(self.cal_points.states) == 0
            })

            if kw.get('compression_seg_lim', None) is None:
                # compress the 2nd sweep dimension completely onto the first
                kw['compression_seg_lim'] = \
                    np.product([len(s) for s in self.mc_points]) \
                    + len(self.cal_points.states)
            self.autorun(**kw)
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def add_amplitude_sweep_points(self, task_list=None):
        """
        If flux pulse amplitudes are not in the sweep_points in each task, but
        qubit frequencies are, then amplitudes will be calculated based on
        the frequencies and the fit_ge_freq_from_flux_pulse_amp qubit parameter.
        sweep_points entry in each task_list will be updated.
        :param task_list: list of dictionaries describing the the measurement
            for each qubit.
        :return: updated task list
        """
        if task_list is None:
            task_list = self.task_list
        # TODO: check combination of sweep points in task and in sweep_points
        for task in task_list:
            sweep_points = task.get('sweep_points', [{}, {}])
            sweep_points = SweepPoints(from_dict_list=sweep_points)
            if len(sweep_points) == 1:
                sweep_points.add_sweep_dimension()
            if 'qubit_freqs' in sweep_points[1]:
                qubit_freqs = sweep_points[1]['qubit_freqs'][0]
            elif len(self.sweep_points) >= 2 and \
                    'qubit_freqs' in self.sweep_points[1]:
                qubit_freqs = self.sweep_points[1]['qubit_freqs'][0]
            else:
                qubit_freqs = None
            if qubit_freqs is not None:
                qubits, _ = self.get_qubits(task['qb'])
                if qubits is None:
                    raise KeyError('qubit_freqs specified in sweep_points, '
                                   'but no qubit objects available, so that '
                                   'the corresponding amplitudes cannot be '
                                   'computed.')
                this_qb = qubits[0]
                fit_paras = deepcopy(this_qb.fit_ge_freq_from_flux_pulse_amp())
                if len(fit_paras) == 0:
                    raise ValueError(
                        f'fit_ge_freq_from_flux_pulse_amp is empty'
                        f' for {this_qb.name}. Cannot calculate '
                        f'amplitudes from qubit frequencies.')
                amplitudes = np.array(fit_mods.Qubit_freq_to_dac(
                    qubit_freqs, **fit_paras))
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

    def t1_flux_pulse_block(self, qb, sweep_points,
                            prepend_pulse_dicts=None, **kw):
        """
        Function that constructs the experiment block for one qubit
        :param qb: name or list with the name of the qubit
            to measure. This function expect only one qubit to measure!
        :param sweep_points: SweepPoints class instance
        :param prepend_pulse_dicts: dictionary of pulses to prepend
        :param kw: keyword arguments
            spectator_op_codes: list of op_codes for spectator qubits
        :return: precompiled block
        """

        qubit_name = qb
        if isinstance(qubit_name, list):
            qubit_name = qubit_name[0]
        hard_sweep_dict, soft_sweep_dict = sweep_points
        pb = self.prepend_pulses_block(prepend_pulse_dicts)

        pulse_modifs = {'all': {'element_name': 'pi_pulse'}}
        pp = self.block_from_ops('pipulse',
                                 [f'X180 {qubit_name}'] +
                                 kw.get('spectator_op_codes', []),
                                 pulse_modifs=pulse_modifs)

        pulse_modifs = {
            'all': {'element_name': 'flux_pulse', 'pulse_delay': 0}}
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

    def run_analysis(self, **kw):
        """
        Runs analysis and stores analysis instance in self.analysis.
        :param kw:
            all_fits (bool, default: True): whether to do all fits
        """

        self.all_fits = kw.get('all_fits', True)
        self.analysis = tda.T1FrequencySweepAnalysis(
            qb_names=self.meas_obj_names,
            options_dict=dict(TwoD=True, all_fits=self.all_fits,
                              global_PCA=not len(self.cal_points.states)))


class ParallelLOSweepExperiment(CalibBuilder):
    def __init__(self, task_list, sweep_points=None, **kw):
        for task in task_list:
            if not isinstance(task['qb'], str):
                task['qb'] = task['qb'].name
            if not 'prefix' in task:
                task['prefix'] = f"{task['qb']}_"

        super().__init__(task_list, sweep_points=sweep_points, **kw)
        self.lo_offsets = {}
        self.lo_sweep_points = []
        self.analysis = {}

        self.preprocessed_task_list = self.preprocess_task_list(**kw)
        self.resolve_lo_sweep_points(**kw)
        self.sequences, self.mc_points = self.parallel_sweep(
            self.preprocessed_task_list, self.sweep_block, **kw)

    def resolve_lo_sweep_points(self, freq_sp_suffix='freq', **kw):
        all_freqs = self.sweep_points.get_sweep_params_property('values', 1,
                                                                'all')
        if np.ndim(all_freqs) == 1:
            all_freqs = [all_freqs]
        all_diffs = [np.diff(freqs) for freqs in all_freqs]
        assert all([np.mean(abs(diff - all_diffs[0]) / all_diffs[0]) < 1e-10
                    for diff in all_diffs]), \
            "The steps between frequency sweep points must be the same for " \
            "all qubits."
        self.lo_sweep_points = all_freqs[0] - all_freqs[0][0]

        if self.qubits is None:
            log.warning('No qubit objects provided. Creating the sequence '
                        'without checking for ge_mod_freq corrections.')
        else:
            temp_vals = []
            for task in self.task_list:
                qb = self.get_qubits(task['qb'])[0][0]
                sp = self.exp_metadata['meas_obj_sweep_points_map'][qb.name]
                freq_sp = [s for s in sp if s.endswith(freq_sp_suffix)][0]
                f_start = self.sweep_points.get_sweep_params_property(
                    'values', 1, freq_sp)[0]
                lo = qb.instr_ge_lo.get_instr()
                if lo not in self.lo_offsets:
                    self.lo_offsets[lo] = f_start - qb.ge_mod_freq()
                else:
                    temp_vals.append(
                        (qb.ge_mod_freq, f_start - self.lo_offsets[lo]))

            with temporary_value(*temp_vals):
                self.update_operation_dict()

    def run_measurement(self, **kw):
        name = 'Drive frequency shift'
        sweep_functions = [swf.Offset_Sweep(
            lo.frequency, offset, name=name, parameter_name=name, unit='Hz')
            for lo, offset in self.lo_offsets.items()]
        self.sweep_functions = [
            self.sweep_functions[0], swf.multi_sweep_function(
                sweep_functions, name=name, parameter_name=name)]
        self.mc_points[1] = self.lo_sweep_points
        super().run_measurement(**kw)

    def get_meas_objs_from_task(self, task):
        return [task['qb']]

    def sweep_block(self, **kw):
        raise NotImplementedError('Child class has to implement sweep_block.')


class FluxPulseScope(ParallelLOSweepExperiment):
    """
        flux pulse scope measurement used to determine the shape of flux pulses
        set up as a 2D measurement (delay and drive pulse frequecy are
        being swept)
        pulse sequence:
                      <- delay ->
           |    -------------    |X180|  ---------------------  |RO|
           |    ---   | ---- fluxpulse ----- |

            sweep_points:
            delay (numpy array): array of amplitudes of the flux pulse
            freq (numpy array): array of drive frequencies

        Returns: None

    """
    kw_for_task_keys = ['ro_pulse_delay']
    kw_for_sweep_points = {
        'freqs': dict(param_name='freq', unit='Hz',
                      label=r'drive frequency, $f_d$',
                      dimension=1),
        'delays': dict(param_name='delay', unit='s',
                       label=r'delay, $\tau$',
                       dimension=0),
    }

    def __init__(self, task_list, sweep_points=None, **kw):
        try:
            self.experiment_name = 'Flux_scope'
            super().__init__(task_list, sweep_points=sweep_points, **kw)
            self.autorun(**kw)
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def sweep_block(self, qb, sweep_points, flux_op_code=None,
                    ro_pulse_delay=None, **kw):
        """
        Performs X180 pulse on top of a fluxpulse
        Timings of sequence
        |          ----------           |X180|  ----------------------------  |RO|
        |        ---      | --------- fluxpulse ---------- |
                         <-  delay  ->

        :param qb: (str) the name of the qubit
        :param sweep_points: the sweep points containing a parameter delay
            in dimension 0
        :param flux_op_code: (optional str) the flux pulse op_code (default
            FP qb)
        :param ro_pulse_delay: Can be 'auto' to start the readout after
            the end of the flux pulse or a delay in seconds to start a fixed
            amount of time after the drive pulse. If not provided or set to
            None, a default fixed delay of 100e-9 is used.

        :param kw:
        """
        if flux_op_code is None:
            flux_op_code = f'FP {qb}'
        if ro_pulse_delay is None:
            ro_pulse_delay = 100e-9
        pulse_modifs = {'attr=name,op_code=X180': f'FPS_Pi',
                        'attr=element_name,op_code=X180': 'FPS_Pi_el'}
        b = self.block_from_ops(f'ge_flux {qb}',
                                [f'X180 {qb}', flux_op_code],
                                pulse_modifs=pulse_modifs)
        fp = b.pulses[1]
        fp['ref_point'] = 'middle'
        offs = fp.get('buffer_length_start', 0)
        fp['pulse_delay'] = ParametricValue(
            'delay', func=lambda x, o=offs: -(x + o))

        if ro_pulse_delay == 'auto':
            delay = \
                fp['pulse_length'] - np.min(
                    sweep_points.get_sweep_params_property(
                        'values', dimension=0, param_names='delay')) + \
                fp.get('buffer_length_end', 0) + fp.get('trans_length', 0)
            b.block_end.update({'ref_pulse': 'FPS_Pi', 'ref_point': 'middle',
                                'pulse_delay': delay})
        else:
            b.block_end.update({'ref_pulse': 'FPS_Pi', 'ref_point': 'end',
                                'pulse_delay': ro_pulse_delay})

        self.data_to_fit.update({qb: 'pe'})

        return b

    def run_analysis(self, **kw):
        """
        Runs analysis and stores analysis instances in self.analysis.
        :param kw:
        """
        self.analysis = tda.FluxPulseScopeAnalysis(
            qb_names=self.meas_obj_names,
            options_dict=dict(TwoD=True, global_PCA=True,))


class Cryoscope(CalibBuilder):
    """
        Delft Cryoscope measurement
        (https://aip.scitation.org/doi/pdf/10.1063/1.5133894)
        used to determine the shape of flux pulses set up as a 2D measurement
        (truncation length and phase of second pi-half pulse are being swept)
        Timings of sequence
        |  --- |Y90| ------------------------------------------- |Y90| -  |RO|
        |  ------- | ------ fluxpulse ------ | separation_buffer | -----
                    <-  truncation_length  ->
    """
    def __init__(self, task_list, sweep_points=None, estimation_window=None,
                 separation_buffer=None, **kw):
        """

        :param task_list:
        :param sweep_points:
        :param estimation_window:
        :param separation_buffer: (float) extra delay between the (truncated)
            flux pulse and the last pi-half pulse
        :param kw:
        """
        try:
            self.experiment_name = 'Cryoscope'
            for task in task_list:
                if not isinstance(task['qb'], str):
                    task['qb'] = task['qb'].name
                if not 'prefix' in task:
                    task['prefix'] = f"{task['qb']}_"

            super().__init__(task_list, sweep_points=sweep_points, **kw)
            self.estimation_window = estimation_window
            if self.estimation_window is None:
                self.estimation_window = 1/2.4e9  # one HDAWG sample
            self.exp_metadata['estimation_window'] = self.estimation_window
            self.separation_buffer = separation_buffer
            if self.separation_buffer is None:
                self.separation_buffer = 100e-9  # one HDAWG sample
            self.exp_metadata['separation_buffer'] = self.separation_buffer

            self.add_default_sweep_points(**kw)
            self.preprocessed_task_list = self.preprocess_task_list(**kw)
            self.sequences, self.mc_points = self.sweep_n_dim(
                self.sweep_points, body_block=None,
                body_block_func=self.sweep_block, cal_points=self.cal_points,
                ro_qubits=self.meas_obj_names, **kw)
            self.autorun(**kw)
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def add_default_sweep_points(self, **kw):
        """
        Adds hard sweep points to self.sweep_points: phases of second pi-half
        pulse and the estimation_window increment to the truncation_length
        :param kw: keyword_arguments
        """
        self.sweep_points = self.add_default_ramsey_sweep_points(
            self.sweep_points, tile=2, repeat=0, **kw)
        nr_phases = self.sweep_points.length(0) // 2
        hard_sweep_dict = SweepPoints(
            'extra_truncation_length', [0] * nr_phases +
                                       [self.estimation_window] * nr_phases,
            's', 'Pulse length')
        self.sweep_points.update(hard_sweep_dict + [{}])

    def sweep_block(self, sp1d_idx, sp2d_idx, **kw):
        """
        Performs a Ramsey phase measurement with a truncated flux pulse between
        the two pi-half pulses.
        Timings of sequence
        |  --- |Y90| ------------------------------------------- |Y90| -  |RO|
        |  ------- | ------ fluxpulse ------ | separation_buffer | -----
                    <-  truncation_length  ->

        :param sp1d_idx: (int) index of sweep point to use from the
            first sweep dimension
        :param sp2d_idx: (int) index of sweep point to use from the
            second sweep dimension
        :param kw: keyword arguments
        """
        from pprint import pprint
        parallel_block_list = []
        for i, task in enumerate(self.preprocessed_task_list):
            sweep_points = task['sweep_points']
            qb = task['qb']
            flux_op_code = task.get('flux_op_code', None)
            if flux_op_code is None:
                flux_op_code = f'FP {qb}'

            # pi half pulses blocks
            pihalf_1_bk = self.block_from_ops(f'pihalf_1_{qb}', [f'Y90 {qb}'])
            pihalf_2_bk = self.block_from_ops(f'pihalf_2_{qb}', [f'Y90 {qb}'])
            # set hard sweep phase and delay of second pi-half pulse
            pihalf_2_bk.pulses[0]['phase'] = \
                sweep_points.get_sweep_params_property(
                    'values', 0, 'phase')[sp1d_idx]
            pihalf_2_bk.pulses[0]['pulse_delay'] = self.separation_buffer

            # extra pulses
            eb = self.prepend_pulses_block(task.get('extra_pulse_dicts', {}))

            main_fpbk = self.block_from_ops(f'fp_main_{qb}', [flux_op_code])
            repark_fpbk = self.block_from_ops(f'fp_repark_{qb}', [f'FP {qb}'])
            repark_fpbk.pulses[0]['amplitude'] = task.get('repark_fp_amp', 0)
            repark_fpbk.pulses[0]['pulse_length'] = task.get(
                'repark_fp_len', main_fpbk.pulses[0]['pulse_length'] +
                                 main_fpbk.pulses[0].get('buffer_length_start', 0) +
                                 main_fpbk.pulses[0].get('buffer_length_end', 0))
            repark_fpbk.pulses[0]['buffer_length_start'] = task.get(
                'repark_fp_buffers',
                repark_fpbk.pulses[0].get('buffer_length_start', 0))
            repark_fpbk.pulses[0]['buffer_length_end'] = task.get(
                'repark_fp_buffers',
                repark_fpbk.pulses[0].get('buffer_length_end', 0))

            # set soft sweep truncation_length
            for k in sweep_points[1]:
                main_fpbk.pulses[0][k] = sweep_points.get_sweep_params_property(
                    'values', 1, k)[sp2d_idx]
            # set hard sweep truncation_length
            main_fpbk.pulses[0]['truncation_length'] += \
                sweep_points.get_sweep_params_property(
                    'values', 0, 'extra_truncation_length')[sp1d_idx]
            if repark_fpbk.pulses[0]['amplitude']:
                # truncate the reparking flux pulse
                repark_fpbk.pulses[0]['truncation_length'] = \
                    main_fpbk.pulses[0]['truncation_length'] + \
                    repark_fpbk.pulses[0].get('buffer_length_start', 0)

            # create final block
            fp_block = self.simultaneous_blocks('flux_pulses_{qb}',
                [main_fpbk, repark_fpbk], block_align='center')
            cryo_blk = self.sequential_blocks(f'cryoscope {qb}',
                [pihalf_1_bk, eb, fp_block, pihalf_2_bk])

            parallel_block_list += [cryo_blk]
            self.data_to_fit.update({qb: 'pe'})

        return self.simultaneous_blocks(
            f'sim_rb_{sp2d_idx}_{sp1d_idx}', parallel_block_list,
            block_align='end')

    def run_analysis(self, **kw):
        """
        Runs analysis and stores analysis instances in self.analysis.
        :param kw:
        """
        qb_names = [task['qb'] for task in self.task_list]
        self.analysis = tda.CryoscopeAnalysis(
            qb_names=qb_names, options_dict={'unwrap_phases': True})


class FluxPulseAmplitudeSweep(ParallelLOSweepExperiment):
    """
        Flux pulse amplitude measurement used to determine the qubits energy in
        dependence of flux pulse amplitude.

        pulse sequence:
           |    -------------    |X180|  ---------------------  |RO|
           |    ---   | ---- fluxpulse ----- |


            sweep_points:
            amplitude (numpy array): array of amplitudes of the flux pulse
            freq (numpy array): array of drive frequencies

        Returns: None

    """
    kw_for_task_keys = ['delay']
    kw_for_sweep_points = {
        'freqs': dict(param_name='freq', unit='Hz',
                      label=r'drive frequency, $f_d$',
                      dimension=1),
        'amps': dict(param_name='amplitude', unit='V',
                       label=r'flux pulse amplitude',
                       dimension=0),
    }

    def __init__(self, task_list, sweep_points=None, **kw):
        try:
            self.experiment_name = 'Flux_amplitude'
            super().__init__(task_list, sweep_points=sweep_points, **kw)
            self.exp_metadata.update({"global_PCA": True})
            self.autorun(**kw)

        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def sweep_block(self, qb, flux_op_code=None, delay=None, **kw):
        """
        Performs X180 pulse on top of a fluxpulse
        :param qb: (str) the name of the qubit
        :param flux_op_code: (optional str) the flux pulse op_code (default:
            FP qb)
        :param delay: (optional float): flux pulse delay (default: centered to
            center of drive pulse)
        :param kw:
        """
        if flux_op_code is None:
            flux_op_code = f'FP {qb}'
        pulse_modifs = {'attr=name,op_code=X180': f'FPS_Pi',
                        'attr=element_name,op_code=X180': 'FPS_Pi_el'}
        b = self.block_from_ops(f'ge_flux {qb}',
                                 [f'X180 {qb}', flux_op_code],
                                 pulse_modifs=pulse_modifs)
        fp = b.pulses[1]
        fp['ref_point'] = 'middle'
        if delay is None:
            delay = fp['pulse_length'] / 2
        fp['pulse_delay'] = -fp.get('buffer_length_start', 0) - delay
        fp['amplitude'] = ParametricValue('amplitude')

        b.set_end_after_all_pulses()

        self.data_to_fit.update({qb: 'pe'})

        return b

    def run_analysis(self, analysis_kwargs=None, **kw):
        """
        Runs analysis and stores analysis instances in self.analysis.
        :param analysis_kwargs: (dict) keyword arguments for analysis
        :param kw: currently ignored
        """
        if analysis_kwargs is None:
            analysis_kwargs = {}

        self.analysis = tda.FluxAmplitudeSweepAnalysis(
            qb_names=self.meas_obj_names, options_dict=dict(TwoD=True),
            t_start=self.timestamp, **analysis_kwargs)

        if self.update:
            for qb in self.meas_obj_names:
                qb.fit_ge_freq_from_flux_pulse_amp(
                    self.analysis.fit_res[f'freq_fit_{qb.name}'].best_values)
