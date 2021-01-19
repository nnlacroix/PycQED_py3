import numpy as np
from copy import copy
from copy import deepcopy
import traceback

from pycqed.measurement.calibration.calibration_points import CalibrationPoints
from pycqed.measurement.calibration.two_qubit_gates import CalibBuilder
import pycqed.measurement.sweep_functions as swf
from pycqed.measurement.waveform_control.block import Block, ParametricValue
from pycqed.measurement.sweep_points import SweepPoints
from pycqed.analysis import fitting_models as fit_mods
import pycqed.analysis_v2.timedomain_analysis as tda
from pycqed.analysis import measurement_analysis as ma
from pycqed.utilities.general import temporary_value
from pycqed.measurement import multi_qubit_module as mqm
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
                [{}, {}] if self.sweep_points is None else self.sweep_points)
            self.add_amplitude_sweep_points()

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
        self.do_fitting = kw.get('do_fitting', True)
        self.analysis = tda.T1FrequencySweepAnalysis(
            qb_names=self.meas_obj_names,
            do_fitting=self.do_fitting,
            options_dict=dict(TwoD=True, all_fits=self.all_fits,
                              rotation_type='global_PCA' if not
                                len(self.cal_points.states) else 'cal_states'))


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


class ActiveReset(CalibBuilder):
    def __init__(self, task_list=None, recalibrate_ro=False,
                 prep_states=('g', 'e'), n_shots=10000,
                 reset_reps=10, set_thresholds=True,
                 **kw):
        """
        Characterize active reset with the following sequence:

        |prep-pulses|--|prep_state_i|--(|RO|--|reset_pulse|) x reset_reps --|RO|

        -Prep-pulses are preselection/active_reset pulses, with parameters defined
        in qb.preparation_params().
        - Prep_state_i is "g", "e", "f" as provided by prep_states.
        - the following readout and reset pulses use the "ro_separation" and
        "post_ro_wait" of the qb.preparation_params() but the number of pulses is set
        by reset_reps, such that we can both apply active reset and characterize the
        reset with different number of pulses.
        Args:
            task_list (list): list of task for the reset. Needs the keys
            recalibrate_ro (bool): whether or not to recalibrate the readout
                before characterizing the active reset.
            prep_states (iterable): list of states on which the reset will be
                characterized
            reset_reps (int): number of readouts used to characterize the reset.
                Note that this parameter does NOT correspond to 'reset_reps' in
                qb.preparation_params() (the latter is used for reset pe
            set_thresholds (bool): whether or not to set the thresholds from
                qb.acq_classifier_params() to the corresponding UHF channel
            n_shots (int): number of single shot measurements
            **kw:
        """

        self.experiment_name = kw.get('experiment_name',
                                      f"active_reset_{prep_states}")

        # build default task in which all qubits are measured
        if task_list is None:
            assert kw.get('qubits', None) is not None, \
                    "qubits must be passed to create default task_list " \
                    "if task_list=None."
            task_list = [{"qubit": [qb.name]} for qb in kw.get('qubits')]

        # configure detector function parameters
        if kw.get("classified", False):
            kw['df_kwargs'] = kw.get('df_kwargs', {})
            if 'det_get_values_kws' not in kw['df_kwargs']:
                kw['df_kwargs'] = {'det_get_values_kws':
                                    {'classified': True,
                                     'correlated': False,
                                     'thresholded': False,
                                     'averaged': False}}
            else:
                # ensure still single shot
                kw['df_kwargs']['det_get_values_kws'].update({'averaged':False})
        else:
            kw['df_name'] = kw.get('df_name', "int_log_det")



        self.set_thresholds = set_thresholds
        self.recalibrate_ro = recalibrate_ro
        # force resetting of thresholds if recalibrating readout
        if self.recalibrate_ro:
            self.set_thresholds = True
        self.prep_states = prep_states
        self.reset_reps = reset_reps
        self.n_shots = n_shots

        # init parent
        super().__init__(task_list=task_list, **kw)

        if self.dev is None and self.recalibrate_ro:
            raise NotImplementedError(
                "Device must be past when 'recalibrate_ro' is True"
                " because the mqm.measure_ssro() requires the device "
                "as argument. TODO: transcribe measure_ssro to QExperiment"
                " framework to avoid this constraint.")

        # all tasks must have same init sweep point because this will
        # fix the number of readouts
        # for now sweep points are global. But we could make the second
        # dimension task-dependent when introducing the sweep over
        # thresholds
        default_sp = SweepPoints("initialize", self.prep_states)
        default_sp.add_sweep_dimension()
        # second dimension to have once only readout and once with feedback
        default_sp.add_sweep_parameter("pulse_off", [1, 0])
        self.sweep_points = kw.get('sweep_points',
                                   default_sp)

        # get preparation parameters for all qubits. Note: in the future we could
        # possibly modify prep_params to be different for each uhf, as long as
        # the number of readout is the same for all UHFs in the experiment
        self.prep_params = deepcopy(self.get_prep_params())
        # set explicitly some preparation params so that they can be retrieved
        # in the analysis
        for param in ('ro_separation', 'post_ro_wait'):
            if not param in self.prep_params:
                self.prep_params[param] = self.STD_PREP_PARAMS[param]
        # set temporary values
        qb_in_exp = self.find_qubits_in_tasks(self.qubits, self.task_list)
        self.temporary_values.extend([(qb.acq_shots, self.n_shots)
                                      for qb in qb_in_exp])

        # by default empty cal points
        # FIXME: Ideally these 2 lines should be handled properly by lower level class,
        #  that does not assume calibration points, instead of overwriting
        self.cal_points = kw.get('cal_points',
                                 CalibrationPoints([qb.name for qb in qb_in_exp],
                                                   ()))
        self.cal_states = kw.get('cal_states', ())

    def prepare_measurement(self, **kw):

        if self.recalibrate_ro:
            self.analysis = mqm.measure_ssro(self.dev, self.qubits, self.prep_states,
                                             update=True)
            # reanalyze to get thresholds
            options_dict = dict(classif_method="threshold")
            a = tda.MultiQutrit_Singleshot_Readout_Analysis(qb_names=self.qb_names,
                                                            options_dict=options_dict)
            for qb in self.qubits:
                classifier_params = a.proc_data_dict[
                    'analysis_params']['classifier_params'][qb.name]
                qb.acq_classifier_params().update(classifier_params)
                qb.preparation_params()['threshold_mapping'] = \
                    classifier_params['mapping']
        if self.set_thresholds:
            self._set_thresholds(self.qubits)
        self.exp_metadata.update({"thresholds":
                                      self._get_thresholds(self.qubits)})
        self.preprocessed_task_list = self.preprocess_task_list(**kw)
        self.sequences, self.mc_points = \
            self.parallel_sweep(self.preprocessed_task_list,
                                self.reset_block, block_align="start", **kw)

        # should transform raw voltage to probas in analysis if no cal points
        # and not classified readout already
        predict_proba = len(self.cal_states) == 0 and not self.classified
        self.exp_metadata.update({"n_shots": self.n_shots,
                                  "predict_proba": predict_proba,
                                  "reset_reps": self.reset_reps})

    def reset_block(self, qubit, **kw):
        _ , qubit = self.get_qubits(qubit) # ensure qubit in list format

        prep_params = deepcopy(self.prep_params)

        self.prep_params['ro_separation'] = ro_sep = prep_params.get("ro_separation",
                                 self.STD_PREP_PARAMS['ro_separation'])
        # remove the reset repetition for preparation and use the number
        # of reset reps for characterization (provided in the experiment)
        prep_params.pop('reset_reps', None)
        prep_params.pop('preparation_type', None)

        reset_type = f"active_reset_{'e' if len(self.prep_states) < 3 else 'ef'}"
        reset_block = self.prepare(block_name="reset_ro_and_feedback_pulses",
                                   qb_names=qubit,
                                   preparation_type=reset_type,
                                   reset_reps=self.reset_reps, **prep_params)
        # delay the reset block by appropriate time as self.prepare otherwise adds reset
        # pulses before segment start
        # reset_block.block_start.update({"pulse_delay": ro_sep * self.reset_reps})
        pulse_modifs={"attr=pulse_off, op_code=X180": ParametricValue("pulse_off"),
                      "attr=pulse_off, op_code=X180_ef": ParametricValue("pulse_off")}
        reset_block = Block("Reset_block",
                            reset_block.build(block_delay=ro_sep * self.reset_reps),
                            pulse_modifs=pulse_modifs)

        ro = self.mux_readout(qubit)
        return [reset_block, ro]

    def run_analysis(self, analysis_class=None, **kwargs):

        self.analysis = tda.MultiQutritActiveResetAnalysis(**kwargs)

    @staticmethod
    def _set_thresholds(qubits, clf_params=None):
        """
        Sets the thresholds in clf_params to the corresponding UHF channel(s)
        for each qubit in qubits.
        Args:
            qubits (list, QuDevTransmon): (list of) qubit(s)
            clf_params (dict): dictionary containing the thresholds that must
                be set on the corresponding UHF channel(s).
                If several qubits are passed, then it assumes clf_params if of the form:
                {qbi: clf_params_qbi, ...}, where clf_params_qbi contains at least
                the "threshold" key.
                If a single qubit qbi is passed (not in a list), then expects only
                clf_params_qbi.
                If None, then defaults to qb.acq_classifier_params().

        Returns:

        """

        # check if single qubit provided
        if np.ndim(qubits) == 0:
            clf_params = {qubits.name: deepcopy(clf_params)}
            qubits = [qubits]

        if clf_params is None:
            clf_params = {qb.name: qb.acq_classifier_params() for qb in qubits}

        for qb in qubits:
            # perpare correspondance between integration unit (key)
            # and uhf channel
            channels = {0: qb.acq_I_channel(), 1: qb.acq_Q_channel()}
            # set thresholds
            for unit, thresh in clf_params[qb.name]['thresholds'].items():
                qb.instr_uhf.get_instr().set(
                    f'qas_0_thresholds_{channels[unit]}_level', thresh)

    @staticmethod
    def _get_thresholds(qubits, from_clf_params=False, all_qb_channels=False):
        """
        Gets the UHF channel thresholds for each qubit in qubits.
        Args:
            qubits (list, QuDevTransmon): (list of) qubit(s)
            from_clf_params (bool): whether thresholds should be retrieved
                from the classifier parameters (when True) or from the UHF
                channel directly (when False).
            all_qb_channels (bool): whether all thresholds should be retrieved
                or only the ones in use for the current weight type of the qubit.
        Returns:

        """

        # check if single qubit provided
        if np.ndim(qubits) == 0:
            qubits = [qubits]

        thresholds = {}
        for qb in qubits:
            # perpare correspondance between integration unit (key)
            # and uhf channel; check if only one channel is asked for
            # (not asked for all qb channels and weight type uses only 1)
            if not all_qb_channels and qb.acq_weights_type() \
                    in ('square_root', 'optimal'):
                chs = {0: qb.acq_I_channel()}
            else:
                # other weight types have 2 channels
                chs = {0: qb.acq_I_channel(), 1: qb.acq_Q_channel()}

            #get clf thresholds
            if from_clf_params:
                thresh_qb = deepcopy(
                    qb.acq_classifier_params().get("thresholds", {}))
                thresholds[qb.name] = {u: thr for u, thr in thresh_qb.items()
                                       if u in chs}
            # get UHF thresholds
            else:
                thresholds[qb.name] = \
                    {u: qb.instr_uhf.get_instr()
                          .get(f'qas_0_thresholds_{ch}_level')
                     for u, ch in chs.items()}

        return thresholds
