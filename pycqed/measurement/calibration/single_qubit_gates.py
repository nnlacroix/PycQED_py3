import numpy as np
from copy import copy
from copy import deepcopy
import traceback
from pycqed.measurement.calibration.two_qubit_gates import CalibBuilder
import pycqed.measurement.sweep_functions as swf
from pycqed.measurement.waveform_control.block import Block, ParametricValue
from pycqed.measurement.waveform_control import segment as seg_mod
from pycqed.measurement.sweep_points import SweepPoints
from pycqed.analysis import fitting_models as fit_mods
import pycqed.analysis_v2.timedomain_analysis as tda
from pycqed.analysis import measurement_analysis as ma
from pycqed.utilities.general import temporary_value
import logging

from pycqed.utilities.timer import Timer

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
                [{}, {}] if self.sweep_points is None else self.sweep_points)
            self.add_amplitude_sweep_points(**kw)

            self.preprocessed_task_list = self.preprocess_task_list(**kw)
            self.sequences, self.mc_points = \
                self.parallel_sweep(self.preprocessed_task_list,
                                    self.t1_flux_pulse_block, **kw)
            self.exp_metadata.update({
                "rotation_type": 'global_PCA' if
                    len(self.cal_points.states) == 0 else 'cal_states'
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

    def add_amplitude_sweep_points(self, task_list=None, **kw):
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
            sweep_points = SweepPoints(sweep_points)
            if len(sweep_points) == 1:
                sweep_points.add_sweep_dimension()
            if 'qubit_freqs' in sweep_points[1]:
                qubit_freqs = sweep_points[1]['qubit_freqs'][0]
            elif len(self.sweep_points) >= 2 and \
                    'qubit_freqs' in self.sweep_points[1]:
                qubit_freqs = self.sweep_points[1]['qubit_freqs'][0]
            else:
                qubit_freqs = None
            if 'amplitude' in sweep_points[1]:
                amplitudes = sweep_points[1]['amplitude'][0]
            elif len(self.sweep_points) >= 2 and \
                    'amplitude' in self.sweep_points[1]:
                amplitudes = self.sweep_points[1]['amplitude'][0]
            else:
                amplitudes = None
            qubits, _ = self.get_qubits(task['qb'])
            if qubit_freqs is None and qubits is not None:
                qb = qubits[0]
                qubit_freqs = qb.calculate_frequency(
                    amplitude=amplitudes,
                    **kw.get('vfc_kwargs', {})
                )
                freq_sweep_points = SweepPoints('qubit_freqs', qubit_freqs,
                                                'Hz', 'Qubit frequency')
                sweep_points.update([{}] + freq_sweep_points)
            if amplitudes is None:
                if qubits is None:
                    raise KeyError('qubit_freqs specified in sweep_points, '
                                   'but no qubit objects available, so that '
                                   'the corresponding amplitudes cannot be '
                                   'computed.')
                qb = qubits[0]
                amplitudes = qb.calculate_flux_voltage(
                    frequency=qubit_freqs,
                    flux=qb.flux_parking(),
                    **kw.get('vfc_kwargs', {})
                )
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

    @Timer()
    def run_analysis(self, **kw):
        """
        Runs analysis and stores analysis instance in self.analysis.
        :param kw:
            all_fits (bool, default: True): whether to do all fits
        """

        self.all_fits = kw.get('all_fits', True)
        self.do_fitting = kw.get('do_fitting', True)
        self.analysis = tda.T1FrequencySweepAnalysis(
            qb_names=self.meas_obj_names,
            do_fitting=self.do_fitting,
            options_dict=dict(TwoD=True, all_fits=self.all_fits,
                              rotation_type='global_PCA' if not
                                len(self.cal_points.states) else 'cal_states'))


class ParallelLOSweepExperiment(CalibBuilder):
    def __init__(self, task_list, sweep_points=None, allowed_lo_freqs=None,
                 adapt_drive_amp=False, adapt_ro_freq=False, **kw):
        for task in task_list:
            if not isinstance(task['qb'], str):
                task['qb'] = task['qb'].name
            if not 'prefix' in task:
                task['prefix'] = f"{task['qb']}_"

        # Passing keyword arguments to the super class (even if they are not
        # needed there) makes sure that they are stored in the metadata.
        super().__init__(task_list, sweep_points=sweep_points,
                         allowed_lo_freqs=allowed_lo_freqs,
                         adapt_drive_amp=adapt_drive_amp,
                         adapt_ro_freq=adapt_ro_freq, **kw)
        self.lo_offsets = {}
        self.lo_qubits = {}
        self.qb_offsets = {}
        self.lo_sweep_points = []
        self.allowed_lo_freqs = allowed_lo_freqs
        self.adapt_drive_amp = adapt_drive_amp
        self.adapt_ro_freq = adapt_ro_freq
        self.drive_amp_adaptation = {}
        self.ro_freq_adaptation = {}
        self.ro_flux_amp_adaptation = {}
        self.analysis = {}

        self.preprocessed_task_list = self.preprocess_task_list(**kw)
        self.resolve_lo_sweep_points(**kw)
        self.sequences, self.mc_points = self.parallel_sweep(
            self.preprocessed_task_list, self.sweep_block, **kw)

    def resolve_lo_sweep_points(self, freq_sp_suffix='freq', **kw):
        all_freqs = np.array(
            self.sweep_points.get_sweep_params_property('values', 1, 'all'))
        if np.ndim(all_freqs) == 1:
            all_freqs = [all_freqs]
        all_diffs = [np.diff(freqs) for freqs in all_freqs]
        assert all([len(d) == 0 for d in all_diffs]) or \
            all([np.mean(abs(diff - all_diffs[0]) / all_diffs[0]) < 1e-10
                 for diff in all_diffs]), \
            "The steps between frequency sweep points must be the same for " \
            "all qubits."
        self.lo_sweep_points = all_freqs[0] - all_freqs[0][0]
        self.exp_metadata['lo_sweep_points'] = self.lo_sweep_points

        temp_vals = []
        if self.qubits is None:
            log.warning('No qubit objects provided. Creating the sequence '
                        'without checking for ge_mod_freq corrections.')
        else:
            f_start = {}
            for task in self.task_list:
                qb = self.get_qubits(task['qb'])[0][0]
                sp = self.exp_metadata['meas_obj_sweep_points_map'][qb.name]
                freq_sp = [s for s in sp if s.endswith(freq_sp_suffix)][0]
                f_start[qb] = self.sweep_points.get_sweep_params_property(
                    'values', 1, freq_sp)[0]
                self.qb_offsets[qb] = f_start[qb] - self.lo_sweep_points[0]
                lo = qb.instr_ge_lo.get_instr()
                if lo not in self.lo_qubits:
                    self.lo_qubits[lo] = [qb]
                else:
                    self.lo_qubits[lo] += [qb]

            for lo, qbs in self.lo_qubits.items():
                for qb in qbs:
                    if lo not in self.lo_offsets:
                        if kw.get('optimize_mod_freqs', False):
                            fs = [f_start[qb] for qb in self.lo_qubits[lo]]
                            self.lo_offsets[lo] = 1 / 2 * (max(fs) + min(fs))
                        else:
                            self.lo_offsets[lo] = f_start[qb] \
                                                  - qb.ge_mod_freq()
                    temp_vals.append(
                        (qb.ge_mod_freq, f_start[qb] - self.lo_offsets[lo]))
            self.exp_metadata['lo_offsets'] = {
                k.name: v for k, v in self.lo_offsets.items()}

        if self.allowed_lo_freqs is not None:
            for task in self.preprocessed_task_list:
                task['pulse_modifs'] = {'attr=mod_frequency': None}
            self.cal_points.pulse_modifs = {'*.mod_frequency': [None]}

        if self.adapt_drive_amp and self.qubits is None:
            log.warning('No qubit objects provided. Creating the sequence '
                        'without adapting drive amp.')
        elif self.adapt_drive_amp:
            for task in self.task_list:
                qb = self.get_qubits(task['qb'])[0][0]
                sp = self.exp_metadata['meas_obj_sweep_points_map'][qb.name]
                freq_sp = [s for s in sp if s.endswith(freq_sp_suffix)][0]
                f = self.sweep_points.get_sweep_params_property(
                    'values', 1, freq_sp)
                amps = qb.get_ge_amp180_from_ge_freq(np.array(f))
                if amps is None:
                    continue
                max_amp = np.max(amps)
                temp_vals.append((qb.ge_amp180, max_amp))
                self.drive_amp_adaptation[qb] = (
                    lambda x, qb=qb, s=max_amp,
                           o=self.qb_offsets[qb] :
                    qb.get_ge_amp180_from_ge_freq(x + o) / s)
                if not kw.get('adapt_cal_point_drive_amp', False):
                    if self.cal_points.pulse_modifs is None:
                        self.cal_points.pulse_modifs = {}
                    self.cal_points.pulse_modifs.update(
                        {f'e_X180 {qb.name}*.amplitude': [
                            qb.ge_amp180() / (qb.get_ge_amp180_from_ge_freq(
                                qb.ge_freq()) / max_amp)]})
            self.exp_metadata['drive_amp_adaptation'] = {
                qb.name: fnc(self.lo_sweep_points)
                for qb, fnc in self.drive_amp_adaptation.items()}

        if self.adapt_ro_freq and self.qubits is None:
            log.warning('No qubit objects provided. Creating the sequence '
                        'without adapting RO freq.')
        elif self.adapt_ro_freq:
            for task in self.task_list:
                qb = self.get_qubits(task['qb'])[0][0]
                if qb.get_ro_freq_from_ge_freq(qb.ge_freq()) is None:
                    continue
                ro_mwg = qb.instr_ro_lo.get_instr()
                if ro_mwg in self.ro_freq_adaptation:
                    raise NotImplementedError(
                        f'RO adaptation for {qb.name} with LO {ro_mwg.name}: '
                        f'Parallel RO frequency adaptation for qubits '
                        f'sharing an LO is not implemented.')
                self.ro_freq_adaptation[ro_mwg] = (
                    lambda x, mwg=ro_mwg, o=self.qb_offsets[qb],
                           f_mod=qb.ro_mod_freq():
                    qb.get_ro_freq_from_ge_freq(x + o) - f_mod)
            self.exp_metadata['ro_freq_adaptation'] = {
                mwg.name: fnc(self.lo_sweep_points)
                for mwg, fnc in self.ro_freq_adaptation.items()}

        for task in self.task_list:
            if 'fp_assisted_ro_calib_flux' in task and 'fluxline' in task:
                if self.qubits is None:
                    log.warning('No qubit objects provided. Creating the '
                                'sequence without RO flux amplitude.')
                    break
                qb = self.get_qubits(task['qb'])[0][0]
                if qb.ro_pulse_type() != 'GaussFilteredCosIQPulseWithFlux':
                    continue
                sp = self.exp_metadata['meas_obj_sweep_points_map'][qb.name]
                freq_sp = [s for s in sp if s.endswith(freq_sp_suffix)][0]
                f = self.sweep_points.get_sweep_params_property(
                    'values', 1, freq_sp)
                ro_fp_amp = lambda x, qb=qb, cal_flux=task[
                    'fp_assisted_ro_calib_flux'] : qb.ro_flux_amplitude() - (
                        qb.calculate_flux_voltage(x) -
                        qb.calculate_voltage_from_flux(cal_flux)) \
                        * qb.flux_amplitude_bias_ratio()
                amps = ro_fp_amp(f)
                max_amp = np.max(np.abs(amps))
                temp_vals.append((qb.ro_flux_amplitude, max_amp))
                self.ro_flux_amp_adaptation[qb] = (
                    lambda x, fnc=ro_fp_amp, s=max_amp, o=self.qb_offsets[qb]:
                    fnc(x + o) / s)
                if 'ro_flux_amp_adaptation' not in self.exp_metadata:
                    self.exp_metadata['ro_flux_amp_adaptation'] = {}
                self.exp_metadata['ro_flux_amp_adaptation'][qb.name] = \
                    amps / max_amp

        with temporary_value(*temp_vals):
            self.update_operation_dict()

    def run_measurement(self, **kw):
        temp_vals = []
        name = 'Drive frequency shift'
        sweep_functions = [swf.Offset_Sweep(
            lo.frequency, offset, name=name, parameter_name=name, unit='Hz')
            for lo, offset in self.lo_offsets.items()]
        if self.allowed_lo_freqs is not None:
            minor_sweep_functions = []
            for lo, qbs in self.lo_qubits.items():
                qb_sweep_functions = []
                for qb in qbs:
                    mod_freq = self.get_pulse(f"X180 {qb.name}")[
                        'mod_frequency']
                    pulsar = qb.instr_pulsar.get_instr()
                    # Pulsar assumes that the first channel in a pair is the
                    # I component. If this is not the case, the following
                    # workaround finds the correct channel to configure
                    # and swaps the sign of the modulation frequency to get
                    # the correct sideband.
                    iq_swapped = (int(qb.ge_I_channel()[-1:])
                                  > int(qb.ge_Q_channel()[-1:]))
                    param = pulsar.parameters[
                        f'{qb.ge_Q_channel()}_mod_freq' if iq_swapped else
                        f'{qb.ge_I_channel()}_mod_freq']
                    # The following temporary value ensures that HDAWG
                    # modulation is set back to its previous state after the end
                    # of the modulation frequency sweep.
                    temp_vals.append((param, None))
                    qb_sweep_functions.append(
                        swf.Transformed_Sweep(param, transformation=(
                            lambda x, o=mod_freq, s=(-1 if iq_swapped else 1)
                            : s * (x + o))))
                minor_sweep_functions.append(swf.multi_sweep_function(
                    qb_sweep_functions))
            sweep_functions = [
                swf.MajorMinorSweep(majsp, minsp,
                                    np.array(self.allowed_lo_freqs) - offset)
                for majsp, minsp, offset in zip(
                    sweep_functions, minor_sweep_functions,
                    self.lo_offsets.values())]
        for qb, adaptation in self.drive_amp_adaptation.items():
            adapt_name = f'Drive amp adaptation freq {qb.name}'
            pulsar = qb.instr_pulsar.get_instr()
            for quad in ['I', 'Q']:
                ch = qb.get(f'ge_{quad}_channel')
                param = pulsar.parameters[f'{ch}_amplitude_scaling']
                sweep_functions += [swf.Transformed_Sweep(
                    param, transformation=adaptation,
                    name=adapt_name, parameter_name=adapt_name, unit='Hz')]
                # The following temporary value ensures that HDAWG
                # amplitude scaling is set back to its previous state after the
                # end of the sweep.
                temp_vals.append((param, 1.0))
        for mwg, adaptation in self.ro_freq_adaptation.items():
            adapt_name = f'RO freq adaptation freq {mwg.name}'
            param = mwg.frequency
            sweep_functions += [swf.Transformed_Sweep(
                param, transformation=adaptation,
                name=adapt_name, parameter_name=adapt_name, unit='Hz')]
            temp_vals.append((param, param()))
        for qb, adaptation in self.ro_flux_amp_adaptation.items():
            adapt_name = f'RO flux amp adaptation freq {qb.name}'
            pulsar = qb.instr_pulsar.get_instr()
            ch = qb.get(f'ro_flux_channel')
            for seg in self.sequences[0].segments.values():
                for p in seg.unresolved_pulses:
                    if (ch in p.pulse_obj.channels and
                        p.pulse_obj.pulse_type
                            != 'GaussFilteredCosIQPulseWithFlux'):
                        raise NotImplementedError(
                            'RO flux amp adaptation cannot be used when the '
                            'sequence contains other flux pulses.')
            param = pulsar.parameters[f'{ch}_amplitude_scaling']
            sweep_functions += [swf.Transformed_Sweep(
                param, transformation=adaptation,
                name=adapt_name, parameter_name=adapt_name, unit='Hz')]
            temp_vals.append((param, param()))
        for task in self.task_list:
            if 'fluxline' not in task:
                continue
            qb = self.get_qubits(task['qb'])[0][0]
            # offs = self.lo_offsets[[lo for lo, qbs in self.lo_qubits.items()
            #                         if qb in qbs][0]]
            dc_amp = (
                lambda x, o=self.qb_offsets[qb],
                       vfc=qb.fit_ge_freq_from_dc_offset() :
                fit_mods.Qubit_freq_to_dac_res(np.array([x + o]), **vfc)[0])
            sweep_functions += [swf.Transformed_Sweep(
                task['fluxline'], transformation=dc_amp,
                name=f'DC Offset {qb.name}',
                parameter_name=f'Parking freq {qb.name}', unit='Hz')]
        self.sweep_functions = [
            self.sweep_functions[0], swf.multi_sweep_function(
                sweep_functions, name=name, parameter_name=name)]
        self.mc_points[1] = self.lo_sweep_points
        with temporary_value(*temp_vals):
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
    kw_for_task_keys = ['ro_pulse_delay', 'fp_truncation',
                        'fp_truncation_buffer',
                        'fp_compensation',
                        'fp_compensation_amp',
                        'fp_during_ro', 'tau',
                        'fp_during_ro_length',
                        'fp_during_ro_buffer']
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
                    ro_pulse_delay=None,
                    fp_truncation=False, fp_compensation=False,
                    fp_compensation_amp=None, fp_truncation_buffer=None,
                    fp_during_ro=False, tau=None,
                    fp_during_ro_length=None,
                    fp_during_ro_buffer=None, **kw):
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
            the end of the flux pulse (or in the middle of the readout-flux-pulse
            if fp_during_ro is True) or a delay in seconds to start a fixed
            amount of time after the drive pulse. If not provided or set to
            None, a default fixed delay of 100e-9 is used.
        :param fp_truncation: Truncate the flux pulse after the drive pulse
        :param fp_truncation_buffer: Time buffer after the drive pulse, before
            the truncation happens.
        :param fp_compensation: Custom compensation for the charge build-up
            in the bias T.
        :param fp_compensation_amp: Fixed amplitude for the custom compensation
            pulse.
        :param fp_during_ro: Play a flux pulse during the read-out pulse to
            bring the qubit actively to the parking position in the case where
            the flux-pulse is not filtered yet. This assumes a unipolar flux-pulse.
        :param fp_during_ro_length: Length of the fp_during_ro.
        :param fp_during_ro_buffer: Time buffer between the drive pulse and
            the fp_during_ro
        :param tau: Approximate dominant time constant in the flux line, which
            is used to calculate the amplitude of the fp_during_ro.

        :param kw:
        """
        if flux_op_code is None:
            flux_op_code = f'FP {qb}'
        if ro_pulse_delay is None:
            ro_pulse_delay = 100e-9
        if fp_truncation_buffer is None:
            fp_truncation_buffer = 5e-8
        if fp_compensation_amp is None:
            fp_compensation_amp = -2
        if tau is None:
            tau = 20e-6
        if fp_during_ro_length is None:
            fp_during_ro_length = 2e-6
        if fp_during_ro_buffer is None:
            fp_during_ro_buffer = 0.2e-6

        if ro_pulse_delay is 'auto' and (fp_truncation or \
            hasattr(fp_truncation, '__iter__')):
            raise Exception('fp_truncation does currently not work ' + \
                            'with the auto mode of ro_pulse_delay.')

        assert not (fp_compensation and fp_during_ro)

        pulse_modifs = {'attr=name,op_code=X180': f'FPS_Pi',
                        'attr=element_name,op_code=X180': 'FPS_Pi_el'}
        b = self.block_from_ops(f'ge_flux {qb}',
                                [f'X180 {qb}'] + [flux_op_code] * \
                                (2 if fp_compensation else 1) \
                                + ([f'FP {qb}'] if fp_during_ro else []),
                                pulse_modifs=pulse_modifs)

        fp = b.pulses[1]
        fp['ref_point'] = 'middle'
        bl_start = fp.get('buffer_length_start', 0)
        bl_end = fp.get('buffer_length_end', 0)

        def fp_delay(x, o=bl_start):
            return -(x+o)

        fp['pulse_delay'] = ParametricValue(
            'delay', func=fp_delay)

        fp_length_function = lambda x: fp['pulse_length']

        if (fp_truncation or hasattr(fp_truncation, '__iter__')):
            if not hasattr(fp_truncation, '__iter__'):
                fp_truncation = [-np.inf, np.inf]
            original_fp_length = fp['pulse_length']
            max_fp_sweep_length = np.max(
                sweep_points.get_sweep_params_property(
                    'values', dimension=0, param_names='delay'))
            sweep_diff = max(max_fp_sweep_length - original_fp_length, 0)
            fp_length_function = lambda x, opl=original_fp_length, \
                o=bl_start + fp_truncation_buffer, trunc=fp_truncation: \
                max(min((x + o), opl), 0) if (x>np.min(trunc) and x<np.max(trunc)) else opl
            # TODO: check what happens if buffer_length_start and buffer_length_end are zero.

            fp['pulse_length'] = ParametricValue(
                'delay', func=fp_length_function)
            if fp_compensation:
                cp = b.pulses[2]
                cp['amplitude'] = -np.sign(fp['amplitude']) * np.abs(
                    fp_compensation_amp)
                cp['pulse_delay'] = sweep_diff + bl_start
                tau = 200e-9 * 100

                def t_trunc(x, fnc=fp_length_function, tau=tau,
                            fp_amp=fp['amplitude'], cp_amp=cp['amplitude']):
                    fp_length = fnc(x)

                    def v_c(tau, fp_length, fp_amp, v_c_start=0):
                        return fp_amp - (fp_amp - v_c_start) * np.exp(
                            -fp_length / tau)

                    v_c_fp = v_c(tau, fp_length, fp_amp, v_c_start=0)
                    return -np.log(cp_amp / (cp_amp - v_c_fp)) * tau

                cp['pulse_length'] = ParametricValue('delay', func=t_trunc)
                # TODO: implement that the ro_delay is adjusted accordingly!

        # assumes a unipolar flux-pulse for the calculation of the
        # amplitude decay.
        if fp_during_ro:
            rfp = b.pulses[2]

            def rfp_delay(x, fp_delay=fp_delay, fp_length=fp_length_function,\
                fp_bl_start=bl_start, fp_bl_end=bl_end):
                return -(fp_length(x)+fp_bl_end+fp_delay(x))

            def rfp_amp(x, fp_delay=fp_delay, rfp_delay=rfp_delay, tau=tau,
                fp_amp=fp['amplitude'], o=fp_during_ro_buffer-bl_start):
                fp_length=-fp_delay(x)+o
                if fp_length <= 0:
                    return 0
                elif rfp_delay(x) < 0:
                    # in the middle of the fp
                    return -fp_amp * np.exp(-fp_length / tau)
                else:
                    # after the end of the fp
                    return fp_amp * (1 - np.exp(-fp_length / tau))

            rfp['pulse_length'] = fp_during_ro_length
            rfp['pulse_delay'] = ParametricValue('delay', func=rfp_delay)
            rfp['amplitude'] = ParametricValue('delay', func=rfp_amp)
            rfp['buffer_length_start'] = fp_during_ro_buffer

        if ro_pulse_delay == 'auto':
            if fp_during_ro:
                # start the ro pulse in the middle of the fp_during_ro pulse
                delay = fp_during_ro_buffer + fp_during_ro_length/2
                b.block_end.update({'ref_pulse': 'FPS_Pi', 'ref_point': 'end',
                                    'pulse_delay': delay})
            else:
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

    @Timer()
    def run_analysis(self, analysis_kwargs=None, **kw):
        """
        Runs analysis and stores analysis instances in self.analysis.
        :param analysis_kwargs: (dict) keyword arguments for analysis
        :param kw:
        """
        if analysis_kwargs is None:
            analysis_kwargs = {}

        options_dict = {'rotation_type': 'fixed_cal_points' if
                            len(self.cal_points.states) > 0 else 'global_PCA',
                        'TwoD': True}
        if 'options_dict' not in analysis_kwargs:
            analysis_kwargs['options_dict'] = {}
        analysis_kwargs['options_dict'].update(options_dict)

        self.analysis = tda.FluxPulseScopeAnalysis(
            qb_names=self.meas_obj_names, **analysis_kwargs)


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

    :param task_list: list of dicts, where each dict contains the parameters of
        a task (= keyword arguments for the block creation function)
    :param sweep_points: SweepPoints class instance. Can also be specified
        separately in each task.
    :param estimation_window: (float or None) delta_tau in the cryoscope paper.
        The extra bit of flux pulse length before truncation in the second
        Ramsey measurement. If None, only one set of Ramsey measurements are
        done. Can also be specified separately in each task.
    :param separation_buffer: (float) extra delay between the (truncated)
        flux pulse and the last pi-half pulse. Can also be specified separately
        in each task.
    :param awg_sample_length: (float) the length of one sample on the flux
        AWG used by the measurement objects in this experiment. Can also be
        specified separately in each task.
    :param sequential: (bool) whether to apply the cryoscope pulses sequentially
        (True) or simultaneously on n-qubits
    :param kw: keyword arguments: passed down to parent class(es)

    The sweep_points for this measurements must contain
        - 0'th sweep dimension: the Ramsey phases, and, optionally, the
        extra_truncation_lengths, which are 0ns for the first Ramsey (first set
        of phases) and the estimation_window for the second Ramsey. This sweep
        dimension is added automatically in add_default_hard_sweep_points;
        user can specify nr_phases and the estimation_window.
        - 1'st sweep dimension: main truncation lengths that specify the length
        of the pulse(s) at each cryoscope point.


    How to use this class:
        - each task must contain the qubit that is measured under the key "qb"
        - specify sweep_points, estimation_window, separation_buffer,
         awg_sample_length either globally as input parameters to the class
         instantiation, or specify them in each task
        - several flux pulses to measure (n pulses between Ramsey pulses):
            - specify the flux_pulse_dicts in each task. This is a list of
            dicts, where each dict can contain the following:
             {'op_code': flux_op_code,
              'truncation_lengths': numpy array
              'spacing': float, # np.arange(0, tot_pulse_length, spacing)
              'nr_points': int, # np.linspace(0, tot_pulse_length, nr_points,
                                              endpoint=True)}
            If truncation_length is given, it will ignore spacing and nr_points.
            If spacing is given, it will ignore nr_points.
            !!! This entry has priority over Option1 below.
        - only one flux pulse to measure (one pulse between Ramsey pulses):
            Option1 :
                - specify the truncation lengths sweep points globally or in
                    each task.
                - optionally, specify flux_op_code in each task
            Option2:
                - specify the flux_pulse_dicts with one dict in the list
        - for any of the cases described above, the user can specify the
            reparking_flux_pulse entry in each task. This entry is a dict that
            specifies the pulse pars for a reparking flux pulse that will be
            applied on top of the flux pulse(s) that are measured by the
            cryoscope (between the 2 Ramsey pulses). The reparking_flux_pulse
            dict must contain at least the 'op_code' entry.
        - for any of the cases described above, the user can specify the
            prepend_pulse_dicts entry in each task.
            See CalibBuilder.prepend_pulses_block() for details.

    Example of a task with all possible entry recognized by this class.
    See above for details on how they are used and which ones have priority
        {'qb': qb,
        'flux_op_code': flux_op_code,
        'sweep_points': SweepPoints instance,
        'flux_pulse_dicts': [{'op_code': flux_op_code0,
                              'spacing': 2*hdawg_sample_length,
                              'nr_points': 10,
                              'truncation_lengths': array},
                             {'op_code': flux_op_code1,
                              'spacing': 2*hdawg_sample_length,
                              'nr_points': 10,
                              'truncation_lengths': array}],
        'awg_sample_length': hdawg_sample_length,
        'estimation_window': hdawg_sample_length,
        'separation_buffer': 50e-9,
        'reparking_flux_pulse': {'op_code': f'FP {qb.name}',
                                 'amplitude': -0.5}}
    """

    def __init__(self, task_list, sweep_points=None, estimation_window=None,
                 separation_buffer=50e-9, awg_sample_length=None,
                 sequential=False, **kw):
        try:
            self.experiment_name = 'Cryoscope'
            for task in task_list:
                if not isinstance(task['qb'], str):
                    task['qb'] = task['qb'].name
                if 'prefix' not in task:
                    task['prefix'] = f"{task['qb']}_"
                if 'awg_sample_length' not in task:
                    task['awg_sample_length'] = awg_sample_length
                if 'estimation_window' not in task:
                    task['estimation_window'] = estimation_window
                if 'separation_buffer' not in task:
                    task['separation_buffer'] = separation_buffer
            # check estimation window
            none_est_windows = [task['estimation_window'] is None for task in
                                task_list]
            if any(none_est_windows) and not all(none_est_windows):
                raise ValueError('Some tasks have estimation_window == None. '
                                 'You can have different values for '
                                 'estimation_window in different tasks, but '
                                 'none these can be None. To use the same '
                                 'estimation window for all tasks, you can set '
                                 'the class input parameter estimation_window.')

            super().__init__(task_list, sweep_points=sweep_points, **kw)
            self.sequential = sequential
            self.blocks_to_save = {}
            self.add_default_soft_sweep_points(**kw)
            self.add_default_hard_sweep_points(**kw)
            self.preprocessed_task_list = self.preprocess_task_list(**kw)
            self.sequences, self.mc_points = self.sweep_n_dim(
                self.sweep_points, body_block=None,
                body_block_func=self.sweep_block, cal_points=self.cal_points,
                ro_qubits=self.meas_obj_names, **kw)
            self.add_blocks_to_metadata()
            self.update_sweep_points(**kw)
            self.autorun(**kw)
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def add_default_soft_sweep_points(self, **kw):
        """
        Adds soft sweep points (truncation_lengths) to each task in
        self.task_list if flux_pulse_dicts in task. The truncation_lengths
        array is a concatenation of the truncation lengths created between 0 and
        total length of each pulse in flux_pulse_dicts.
        I also adds continuous_truncation_lengths to each task which contains
        the continuous-time version of the truncation_lengths described above.
        :param kw: keyword_arguments (to allow pass through kw even if it
            contains entries that are not needed)
        """
        for task in self.task_list:
            awg_sample_length = task['awg_sample_length']
            if 'flux_pulse_dicts' not in task:
                if 'sweep_points' not in task:
                    raise ValueError(f'Please provide either "sweep_points" '
                                     f'or "flux_pulse_dicts" in the task dict '
                                     f'for {task["qb"]}.')
                continue
            else:
                if awg_sample_length is None:
                    raise ValueError(f'Please provide the length of one sample '
                                     f'for the flux AWG of {task["qb"]}')

            flux_pulse_dicts = task['flux_pulse_dicts']
            trunc_lengths = len(flux_pulse_dicts) * ['']
            continuous_trunc_lengths = len(flux_pulse_dicts) * ['']
            for i, fpd in enumerate(flux_pulse_dicts):
                pd_temp = {'element_name': 'dummy'}
                pd_temp.update(self.get_pulse(fpd['op_code']))
                pulse_length = seg_mod.UnresolvedPulse(pd_temp).pulse_obj.length
                if 'truncation_lengths' in fpd:
                    tr_lens = fpd['truncation_lengths']
                elif 'spacing' in fpd:
                    tr_lens = np.arange(0, pulse_length, fpd['spacing'])
                    if not np.isclose(tr_lens[-1], pulse_length):
                        tr_lens = np.append(tr_lens, pulse_length)
                    tr_lens -= tr_lens % awg_sample_length
                    tr_lens += awg_sample_length/2
                elif 'nr_points' in fpd:
                    tr_lens = np.linspace(0, pulse_length, fpd['nr_points'],
                                          endpoint=True)
                    tr_lens -= tr_lens % awg_sample_length
                    tr_lens += awg_sample_length/2
                elif 'truncation_lengths' in fpd:
                    tr_lens = fpd['truncation_lengths']
                else:
                    raise ValueError(f'Please specify either "delta_tau" or '
                                     f'"nr_points" or "truncation_lengths" '
                                     f'for {task["qb"]}')

                trunc_lengths[i] = tr_lens
                task['flux_pulse_dicts'][i]['nr_points'] = len(tr_lens)
                if i:
                    continuous_trunc_lengths[i] = \
                        tr_lens + continuous_trunc_lengths[i-1][-1]
                else:
                    continuous_trunc_lengths[i] = tr_lens

            sp = task.get('sweep_points', SweepPoints())
            sp.update(SweepPoints('truncation_length',
                                  np.concatenate(trunc_lengths),
                                  's', 'Length', dimension=1))
            task['sweep_points'] = sp
            task['continuous_truncation_lengths'] = np.concatenate(
                continuous_trunc_lengths)

    def add_default_hard_sweep_points(self, **kw):
        """
        Adds hard sweep points to self.sweep_points: phases of second pi-half
        pulse and the estimation_window increment to the truncation_length,
        if provided.
        :param kw: keyword_arguments (to allow pass through kw even if it
            contains entries that are not needed)
        """
        none_est_windows = [task['estimation_window'] is None for task in
                            self.task_list]
        self.sweep_points = self.add_default_ramsey_sweep_points(
            self.sweep_points, tile=0 if any(none_est_windows) else 2,
            repeat=0, **kw)

        for task in self.task_list:
            estimation_window = task['estimation_window']
            if estimation_window is None:
                log.warning(f'estimation_window is missing for {task["qb"]}. '
                            f'The global parameter estimation_window is also '
                            f'missing.\nDoing only one Ramsey per truncation '
                            f'length.')
            else:
                nr_phases = self.sweep_points.length(0) // 2
                task_sp = task.pop('sweep_points', SweepPoints())
                task_sp.update(SweepPoints('extra_truncation_length',
                                           [0] * nr_phases +
                                           [estimation_window] * nr_phases,
                                           's', 'Pulse length', dimension=0))
                task['sweep_points'] = task_sp

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
        :param kw: keyword arguments (to allow pass through kw even if it
            contains entries that are not needed)

        Assumptions:
            - uses the sweep_points entry in each task. If more than one pulse
            between the two Ramsey pulses, then assumes the sweep_points are a
            concatenation of the truncation_lengths array for each pulse,
            defined between 0 and total length of each pulse.
        """
        from pprint import pprint
        parallel_block_list = []
        for i, task in enumerate(self.preprocessed_task_list):
            sweep_points = task['sweep_points']
            qb = task['qb']

            # pi half pulses blocks
            pihalf_1_bk = self.block_from_ops(f'pihalf_1_{qb}', [f'Y90 {qb}'])
            pihalf_2_bk = self.block_from_ops(f'pihalf_2_{qb}', [f'Y90 {qb}'])
            # set hard sweep phase and delay of second pi-half pulse
            pihalf_2_bk.pulses[0]['phase'] = \
                sweep_points.get_sweep_params_property(
                    'values', 0, 'phase')[sp1d_idx]
            pihalf_2_bk.pulses[0]['pulse_delay'] = task['separation_buffer']

            # pulses to prepend
            prep_bk = self.prepend_pulses_block(task.get('prepend_pulse_dicts',
                                                         {}))

            # pulse(s) to measure with cryoscope
            if 'flux_pulse_dicts' in task:
                ops = [fpd['op_code'] for fpd in task['flux_pulse_dicts']]
                main_fpbk = self.block_from_ops(f'fp_main_{qb}', ops)
                n_pts_per_pulse = [fpd['nr_points'] for fpd in
                                   task['flux_pulse_dicts']]
                mask = (np.cumsum(n_pts_per_pulse) <= sp2d_idx)
                meas_pulse_idx = np.count_nonzero(mask)
                # set soft sweep truncation_length
                main_fpbk.pulses[meas_pulse_idx]['truncation_length'] = \
                    sweep_points.get_sweep_params_property(
                        'values', 1, 'truncation_length')[sp2d_idx]
                if task['estimation_window'] is not None:
                    # set hard sweep truncation_length
                    main_fpbk.pulses[meas_pulse_idx]['truncation_length'] += \
                        sweep_points.get_sweep_params_property(
                            'values', 0, 'extra_truncation_length')[sp1d_idx]
                # for the pulses that come after the pulse that is currently
                # truncated, set all their amplitude parameters to 0
                for pidx in range(meas_pulse_idx+1, len(n_pts_per_pulse)):
                    for k in main_fpbk.pulses[pidx]:
                        if 'amp' in k:
                            main_fpbk.pulses[pidx][k] = 0
            else:
                flux_op_code = task.get('flux_op_code', None)
                if flux_op_code is None:
                    flux_op_code = f'FP {qb}'
                ops = [flux_op_code]
                main_fpbk = self.block_from_ops(f'fp_main_{qb}', ops)
                meas_pulse_idx = 0
                # set soft sweep truncation_length
                for k in sweep_points[1]:
                    main_fpbk.pulses[meas_pulse_idx][k] = \
                        sweep_points.get_sweep_params_property('values', 1, k)[
                            sp2d_idx]
                if task['estimation_window'] is not None:
                    # set hard sweep truncation_length
                    main_fpbk.pulses[meas_pulse_idx]['truncation_length'] += \
                        sweep_points.get_sweep_params_property(
                            'values', 0, 'extra_truncation_length')[sp1d_idx]

            # reparking flux pulse
            if 'reparking_flux_pulse' in task:
                reparking_fp_params = task['reparking_flux_pulse']
                if 'pulse_length' not in reparking_fp_params:
                    # set pulse length
                    reparking_fp_params['pulse_length'] = self.get_ops_duration(
                        pulses=main_fpbk.pulses)

                repark_fpbk = self.block_from_ops(
                    f'fp_repark_{qb}', reparking_fp_params['op_code'],
                    pulse_modifs={0: reparking_fp_params})

                # truncate the reparking flux pulse
                repark_fpbk.pulses[0]['truncation_length'] = \
                    main_fpbk.pulses[meas_pulse_idx]['truncation_length'] + \
                    repark_fpbk.pulses[0].get('buffer_length_start', 0)
                if meas_pulse_idx:
                    repark_fpbk.pulses[0]['truncation_length'] += \
                        self.get_ops_duration(pulses=main_fpbk.pulses[
                                                     :meas_pulse_idx])

                main_fpbk = self.simultaneous_blocks(
                    'flux_pulses_{qb}', [main_fpbk, repark_fpbk],
                    block_align='center')

            if sp1d_idx == 0 and sp2d_idx == 0:
                self.blocks_to_save[qb] = deepcopy(main_fpbk)


            cryo_blk = self.sequential_blocks(f'cryoscope {qb}',
                [prep_bk, pihalf_1_bk, main_fpbk, pihalf_2_bk])

            parallel_block_list += [cryo_blk]
            self.data_to_fit.update({qb: 'pe'})

        if self.sequential:
            return self.sequential_blocks(
                f'sim_rb_{sp2d_idx}_{sp1d_idx}', parallel_block_list)
        else:
            return self.simultaneous_blocks(
                f'sim_rb_{sp2d_idx}_{sp1d_idx}', parallel_block_list,
                block_align='end')

    def update_sweep_points(self, **kw):
        """
        Updates the soft sweep points in self.sweep_points with the
        continuous_truncation_lengths from each task in preprocessed_task_list,
        if it exists.
        :param kw: keyword arguments (to allow pass through kw even if it
            contains entries that are not needed)
        """
        sp = SweepPoints()
        for task in self.preprocessed_task_list:
            if 'continuous_truncation_lengths' not in task:
                continue
            param_name = f'{task["prefix"]}truncation_length'
            unit = self.sweep_points.get_sweep_params_property(
                'unit', 1, param_names=param_name)
            label = self.sweep_points.get_sweep_params_property(
                'label', 1, param_names=param_name)
            sp.add_sweep_parameter(param_name,
                                   task['continuous_truncation_lengths'],
                                   unit, label, dimension=1)
        self.sweep_points.update(sp)

    @Timer()
    def run_analysis(self, analysis_kwargs=None, **kw):
        """
        Runs analysis and stores analysis instances in self.analysis.
        :param analysis_kwargs: (dict) keyword arguments for analysis
        :param kw: keyword arguments (to allow pass through kw even if it
            contains entries that are not needed)
        """
        qb_names = [task['qb'] for task in self.task_list]
        if analysis_kwargs is None:
            analysis_kwargs = {}
        self.analysis = tda.CryoscopeAnalysis(
            qb_names=qb_names, **analysis_kwargs)

    def add_blocks_to_metadata(self):
        self.exp_metadata['flux_pulse_blocks'] = {}
        for qb, block in self.blocks_to_save.items():
            self.exp_metadata['flux_pulse_blocks'][qb] = block.build()


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
            self.exp_metadata.update({'rotation_type': 'global_PCA'})
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

    @Timer()
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

    def run_update(self, **kw):
        for qb in self.meas_obj_names:
            qb.fit_ge_freq_from_flux_pulse_amp(
                self.analysis.fit_res[f'freq_fit_{qb.name}'].best_values)

class RabiFrequencySweep(ParallelLOSweepExperiment):
    kw_for_sweep_points = {
        'freqs': dict(param_name='freq', unit='Hz',
                      label=r'drive frequency, $f_d$',
                      dimension=1),
        'amps': dict(param_name='amplitude', unit='V',
                       label=r'drive pulse amplitude',
                       dimension=0),
    }

    def __init__(self, task_list, sweep_points=None, **kw):
        try:
            self.experiment_name = 'RabiFrequencySweep'
            super().__init__(task_list, sweep_points=sweep_points, **kw)
            self.autorun(**kw)

        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def sweep_block(self, qb, **kw):
        b = self.block_from_ops(f'ge {qb}', [f'X180 {qb}'])
        b.pulses[0]['amplitude'] = ParametricValue('amplitude')
        self.data_to_fit.update({qb: 'pe'})
        return b
