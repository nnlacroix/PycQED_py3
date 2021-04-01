import numpy as np
import traceback
from copy import deepcopy
import random
from pycqed.analysis_v3.processing_pipeline import ProcessingPipeline
from pycqed.measurement.calibration.two_qubit_gates import MultiTaskingExperiment
from pycqed.measurement.sweep_points import SweepPoints
from pycqed.measurement.randomized_benchmarking import \
    randomized_benchmarking as rb
import pycqed.measurement.randomized_benchmarking.two_qubit_clifford_group as tqc
from pycqed.analysis_v3 import *
import logging
log = logging.getLogger(__name__)


class RandomizedBenchmarking(MultiTaskingExperiment):

    kw_for_sweep_points = {
        'cliffords': dict(param_name='cliffords', unit='',
                          label='Nr. Cliffords',
                          dimension=0),
        'nr_seeds,nr_m': dict(param_name='seeds', unit='',
                         label='Seeds', dimension=1,
                         values_func=lambda ns, nm: np.array(
                             [np.random.randint(0, 1e8, ns)
                              for _ in range(nm)]).T),
    }

    def __init__(self, task_list, sweep_points=None, qubits=None,
                 nr_seeds=None, cliffords=None, sweep_type=None,
                 interleaved_gate=None, gate_decomposition='HZ', **kw):
        """
        Class to run and analyze the randomized benchmarking experiment on
        one or several qubits in parallel, using the single-qubit Clifford group
        :param task_list: list of dicts; see CalibBuilder docstring
        :param sweep_points: SweepPoints class instance
            Ex: [{'cliffords': ([0, 4, 10], '', 'Nr. Cliffords')},
                 {'seeds': (array([0, 1, 2, 3]), '', 'Nr. Seeds')}]
        :param qubits: list of QuDev_transmon class instances
        :param nr_seeds: int specifying the number of times the Clifford
            group should be sampled for each Clifford sequence length.
            If nr_seeds is specified and it does not exist in the SweepPoints
            of each task in task_list, THEN ALL TASKS WILL RECEIVE THE SAME
            PULSES!!!
        :param cliffords: list or array of integers specifying the number of
            cliffords to apply.
        :param sweep_type: dict of the form {'cycles': 0/1, 'seqs': 1/0}, where
            the integers specify which parameter should correspond to the inner
            sweep (0), and which to the outer sweep (1).
        :param interleaved_gate: string specifying the interleaved gate in
            pycqed notation (ex: X90, Y180 etc). Gate must be part of the
            Clifford group.
        :param gate_decomposition: string specifying what decomposition to use
            to translate the Clifford elements into applicable pulses.
            Possible choices are 'HZ' or 'XY'.
            See HZ_gate_decomposition and XY_gate_decomposition in
            pycqed\measurement\randomized_benchmarking\clifford_decompositions.py
        :param kw: keyword arguments
            passed to CalibBuilder; see docstring there

        Assumptions:
         - assumes there is one task for each qubit. If task_list is None, it
          will internally create it.
         - in rb_block, it assumes only one parameter is being swept in the
         second sweep dimension (cliffords)
         - interleaved_gate and gate_decomposition should be the same for
         all qubits since otherwise the segments will have very different
         lengths for different qubits
        """
        try:
            self.sweep_type = sweep_type
            if self.sweep_type is None:
                self.sweep_type = {'cliffords': 0, 'seeds': 1}
            self.kw_for_sweep_points = deepcopy(self.kw_for_sweep_points)
            self.kw_for_sweep_points['nr_seeds,nr_m']['dimension'] = \
                self.sweep_type['seeds']
            self.kw_for_sweep_points['cliffords']['dimension'] = \
                self.sweep_type['cliffords']

            self.interleaved_gate = interleaved_gate
            if self.interleaved_gate is not None:
                self.kw_for_sweep_points['nr_seeds,nr_m'] = [
                    dict(param_name='seeds', unit='',
                         label='Seeds', dimension=self.sweep_type['seeds'],
                         values_func=lambda ns, nm: np.array(
                             [np.random.randint(0, 1e8, ns)
                              for _ in range(nm)]).T),
                    dict(param_name='seeds_irb', unit='',
                         label='Seeds', dimension=self.sweep_type['seeds'],
                         values_func=lambda ns, nm: np.array(
                             [np.random.randint(0, 1e8, ns)
                              for _ in range(nm)]).T)]
            kw['cal_states'] = kw.get('cal_states', '')

            if cliffords is not None:
                nr_m = len(cliffords)
            else:
                nr_m = len(task_list[0].get('cliffords', []))
            for task in task_list:
                task['nr_m'] = nr_m

            super().__init__(task_list, qubits=qubits,
                             sweep_points=sweep_points,
                             nr_seeds=nr_seeds, nr_m=nr_m,
                             cliffords=cliffords, **kw)
            if self.experiment_name is None:
                self.experiment_name = f'RB_{gate_decomposition}' if \
                    interleaved_gate is None else f'IRB_{gate_decomposition}'
            self.identical_pulses = nr_seeds is not None and all([
                task.get('nr_seeds', None) is None for task in task_list])
            self.gate_decomposition = gate_decomposition
            self.preprocessed_task_list = self.preprocess_task_list(**kw)

            # Check if we can apply identical pulses on all qubits in task_list
            # Can only do this if they have identical cliffords array
            if self.identical_pulses:
                unique_clf_sets = np.unique([
                    self.sweep_points.get_sweep_params_property(
                        'values', self.sweep_type['cliffords'], k)
                    for k in self.sweep_points.get_sweep_dimension(
                        self.sweep_type['cliffords']) if
                    k.endswith('cliffords')], axis=0)
                if unique_clf_sets.shape[0] > 1:
                    raise ValueError('Cannot apply identical pulses. '
                                     'Not all qubits have the same Cliffords.'
                                     'To use non-identical pulses, '
                                     'move nr_seeds from keyword arguments '
                                     'into the tasks.')

            # # remove the redundant 'seeds' entry in self.sweep_points which
            # # might have been left over from self.generate_kw_sweep_points
            # dim = self.sweep_points.find_parameter('seeds')
            # if dim is not None:
            #     self.sweep_points.get_sweep_params_description('seeds', dim,
            #                                                    pop=True)
            self.sequences, self.mc_points = self.sweep_n_dim(
                self.sweep_points, body_block=None,
                body_block_func=self.rb_block, cal_points=self.cal_points,
                ro_qubits=self.meas_obj_names, **kw)
            if self.interleaved_gate is not None:
                seqs_irb, _ = self.sweep_n_dim(
                    self.sweep_points, body_block=None,
                    body_block_func_kw={'interleaved_gate':
                                            self.interleaved_gate},
                    body_block_func=self.rb_block, cal_points=self.cal_points,
                    ro_qubits=self.meas_obj_names, **kw)
                # interleave sequences
                self.sequences, self.mc_points = \
                    self.sequences[0].interleave_sequences(
                        [self.sequences, seqs_irb])
                self.exp_metadata['interleaved_gate'] = self.interleaved_gate
            self.exp_metadata['gate_decomposition'] = self.gate_decomposition
            self.exp_metadata['identical_pulses'] = self.identical_pulses

            self.add_processing_pipeline(**kw)
            self.autorun(**kw)

        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def rb_block(self, sp1d_idx, sp2d_idx, **kw):
        pass

    def add_processing_pipeline(self, **kw):
        """
        Creates and adds the analysis processing pipeline to exp_metadata.
        """
        # TODO: needs upgrade
        pass
        # if 'dim_hilbert' not in kw:
        #     raise ValueError('Please specify the dimension of the Hilbert '
        #                      'space "dim_hilbert" for this measurement.')
        # if 'log' in self.df_name:
        #     pp = rb_ana.pipeline_ssro_measurement(
        #         self.meas_obj_names, self.exp_metadata[
        #             'meas_obj_sweep_points_map'], self.sweep_points,
        #         n_shots=max(qb.acq_shots() for qb in self.meas_objs),
        #         cal_points=self.cal_points, sweep_type=self.sweep_type,
        #         interleaved_irb=self.interleaved_gate is not None, **kw)
        #     self.exp_metadata.update({'processing_pipeline': pp})
        # else:
        #     log.debug(f'There is no support for automatic pipeline creation '
        #               f'for the detector type {self.df_name}')

    def run_analysis(self, **kw):
        """
        Runs analysis and stores analysis instance in self.analysis.
        :param kw: keyword_arguments passed to analysis functions;
            see docstrings there

        Currently only works for a single-shot measurement without preselection,
        and if the readout thresholds exist in the acq_classifier_params of
        each qubit at the time of measurement.

        """

        if 'dim_hilbert' not in kw:
            raise ValueError('Please specify the dimension of the Hilbert '
                             'space "dim_hilbert" for this measurement.')
        if 'log' not in self.df_name:
            raise NotImplementedError(
                f'Automatic analysis only works for a single-shot '
                f'measurement without preselection. '
                f'There is no support for the detector type '
                f'{self.df_name}. Use df_name=int_log_det.')

        pp = rb_ana.pipeline_ssro_measurement(
            self.meas_obj_names, self.exp_metadata[
                'meas_obj_sweep_points_map'], self.sweep_points,
            n_shots=max(qb.acq_shots() for qb in self.meas_objs),
            dim_hilbert=kw['dim_hilbert'], cal_points=self.cal_points,
            sweep_type=self.sweep_type,
            interleaved_irb=self.interleaved_gate is not None, **kw)
        params_dict = {f'{qbn}.acq_classifier_params':
                           f'Instrument settings.{qbn}.acq_classifier_params'
                       for qbn in self.meas_obj_names}
        pp.add_node('extract_data_hdf', params_dict=params_dict, at_idx=0)
        # self.exp_metadata.update({'processing_pipeline': pp})
        self.analysis = pp

        data_dict = {'plot_T1_lim': True, 'do_simple_fit': False}
        self.analysis.resolve(self.exp_metadata['meas_obj_value_names_map'])
        self.analysis(data_dict)
        self.analysis.save()


class SingleQubitRandomizedBenchmarking(RandomizedBenchmarking):

    def __init__(self, task_list, sweep_points=None, **kw):
        """
        See docstring for RandomizedBenchmarking.
        """
        self.experiment_name = f'SingleQubitRB' if \
            kw.get('interleaved_gate', None) is None else f'SingleQubitIRB'

        for task in task_list:
            if 'qb' not in task:
                raise ValueError('Please specify "qb" in each task in '
                                 '"task_list."')
            if not isinstance(task['qb'], str):
                task['qb'] = task['qb'].name
            if 'prefix' not in task:
                task['prefix'] = f"{task['qb']}_"

        kw['dim_hilbert'] = 2
        super().__init__(task_list, sweep_points=sweep_points, **kw)

    def rb_block(self, sp1d_idx, sp2d_idx, **kw):
        interleaved_gate = kw.get('interleaved_gate', None)
        pulse_op_codes_list = []
        tl = [self.preprocessed_task_list[0]] if self.identical_pulses else \
            self.preprocessed_task_list
        for i, task in enumerate(tl):
            param_name = 'seeds' if interleaved_gate is None else 'seeds_irb'
            seed_idx = sp1d_idx if self.sweep_type['seeds'] == 0 else sp2d_idx
            clf_idx = sp1d_idx if self.sweep_type['cliffords'] == 0 else sp2d_idx
            seed = task['sweep_points'].get_sweep_params_property(
                'values', self.sweep_type['seeds'], param_name)[
                seed_idx, clf_idx]
            clifford = task['sweep_points'].get_sweep_params_property(
                'values', self.sweep_type['cliffords'], 'cliffords')[clf_idx]
            cl_seq = rb.randomized_benchmarking_sequence(
                clifford, seed=seed,
                interleaved_gate=interleaved_gate)
            pulse_op_codes_list += [rb.decompose_clifford_seq(
                cl_seq, gate_decomp=self.gate_decomposition)]
        rb_block_list = [self.block_from_ops(
            f"rb_{task['qb']}", [f"{p} {task['qb']}" for p in
                                 pulse_op_codes_list[0 if self.identical_pulses
                                 else i]])
            for i, task in enumerate(self.preprocessed_task_list)]

        return self.simultaneous_blocks(f'sim_rb_{sp1d_idx}{sp1d_idx}',
                                        rb_block_list, block_align='end')


class TwoQubitRandomizedBenchmarking(RandomizedBenchmarking):

    def __init__(self, task_list, sweep_points=None,
                 max_clifford_idx=11520, **kw):
        """
        See docstring for RandomizedBenchmarking.

        :param max_clifford_idx: int that allows to restrict the two qubit
            Clifford that is sampled. Set to 24**2 to only sample the tensor
            product of 2 single qubit Clifford groups.
        """
        self.max_clifford_idx = max_clifford_idx
        tqc.gate_decomposition = rb.get_clifford_decomposition(
            kw.get('gate_decomposition', 'HZ'))

        for task in task_list:
            for k in ['qb_1', 'qb_2']:
                if not isinstance(task[k], str):
                    task[k] = task[k].name
            if 'prefix' not in task:
                task['prefix'] = f"{task['qb_1']}{task['qb_2']}_"
        kw['for_ef'] = kw.get('for_ef', True)
        self.experiment_name = 'TwoQubitRB' if \
            kw.get('interleaved_gate', None) is None else 'TwoQubitIRB'

        kw['dim_hilbert'] = 4
        super().__init__(task_list, sweep_points=sweep_points, **kw)

    def guess_label(self, **kw):
        """
        Default measurement label.
        """
        suffix = [''.join([task['qb_1'], task['qb_2']])
                  for task in self.task_list]
        suffix = '_'.join(suffix)
        if self.label is None:
            if self.interleaved_gate is None:
                self.label = f'{self.experiment_name}_' \
                             f'{self.gate_decomposition}_{suffix}'
            else:
                self.label = f'{self.experiment_name}_{self.interleaved_gate}_' \
                             f'{self.gate_decomposition}_{suffix}'

    def rb_block(self, sp1d_idx, sp2d_idx, **kw):
        interleaved_gate = kw.get('interleaved_gate', None)
        rb_block_list = []
        for i, task in enumerate(self.preprocessed_task_list):
            param_name = 'seeds' if interleaved_gate is None else 'seeds_irb'
            seed_idx = sp1d_idx if self.sweep_type['seeds'] == 0 else sp2d_idx
            clf_idx = sp1d_idx if self.sweep_type['cliffords'] == 0 else sp2d_idx
            seed = task['sweep_points'].get_sweep_params_property(
                'values', self.sweep_type['seeds'], param_name)[
                seed_idx, clf_idx]
            clifford = task['sweep_points'].get_sweep_params_property(
                'values', self.sweep_type['cliffords'], 'cliffords')[clf_idx]
            cl_seq = rb.randomized_benchmarking_sequence_new(
                clifford, number_of_qubits=2, seed=seed,
                max_clifford_idx=kw.get('max_clifford_idx',
                                        self.max_clifford_idx),
                interleaving_cl=interleaved_gate)

            qb_1 = task['qb_1']
            qb_2 = task['qb_2']
            seq_blocks = []
            single_qb_gates = {qb_1: [], qb_2: []}
            for k, idx in enumerate(cl_seq):
                self.timer.checkpoint("rb_block.seq.iteration.start")
                pulse_tuples_list = tqc.TwoQubitClifford(
                    idx).gate_decomposition
                for j, pulse_tuple in enumerate(pulse_tuples_list):
                    if isinstance(pulse_tuple[1], list):
                        seq_blocks.append(
                            self.simultaneous_blocks(
                                f'blk{k}_{j}', [
                            self.block_from_ops(f'blk{k}_{j}_{qbn}', gates)
                                    for qbn, gates in single_qb_gates.items()]))
                        single_qb_gates = {qb_1: [], qb_2: []}
                        seq_blocks.append(self.block_from_ops(
                            f'blk{k}_{j}_cz',
                            f'{kw.get("cz_pulse_name", "CZ")} {qb_1} {qb_2}'))
                    else:
                        qb_name = qb_1 if '0' in pulse_tuple[1] else qb_2
                        pulse_name = pulse_tuple[0]
                        single_qb_gates[qb_name].append(
                            pulse_name + ' ' + qb_name)
                self.timer.checkpoint("rb_block.seq.iteration.end")

            seq_blocks.append(
                self.simultaneous_blocks(
                    f'blk{i}', [
                        self.block_from_ops(f'blk{i}{qbn}', gates)
                        for qbn, gates in single_qb_gates.items()]))
            rb_block_list += [self.sequential_blocks(
                f'rb_block{i}', seq_blocks)]

        return self.simultaneous_blocks(
            f'sim_rb_{sp2d_idx}_{sp1d_idx}', rb_block_list, block_align='end')


class SingleQubitXEB(MultiTaskingExperiment):

    kw_for_sweep_points = {
        'cycles': dict(param_name='cycles', unit='',
                          label='Nr. Cycles',
                          dimension=0),
        'nr_seqs,cycles': dict(param_name='z_rots', unit='',
                              label='$R_z$ angles, $\\phi$', dimension=1,
                              values_func=lambda ns, cycles: [[
                                  list(np.random.uniform(0, 2, nc) * 180)
                                  for nc in cycles] for _ in range(ns)]),
    }

    def __init__(self, task_list, sweep_points=None, qubits=None,
                 nr_seqs=None, cycles=None, sweep_type=None, **kw):
        """
        Class to run cross-entropy benchmarking experiment on
        one or several qubits in parallel.
        The experiment consists of applying
        [[Ry - Rz(theta)] * nr_cycles for nr_cycles in cycles] nr_seqs times,
        with random values of theta each time.
        :param task_list: list of dicts; see CalibBuilder docstring
        :param sweep_points: SweepPoints class instance
        :param qubits: list of QuDev_transmon class instances
        :param nr_seqs: int specifying the number of times to apply a random
            iteration of a sequence consisting of nr_cycles cycles.
            If nr_seqs is specified and it does not exist in the task_list,
            THEN ALL QUBITS WILL RECEIVE THE SAME PULSES provided they have the
            same cycles array.
        :param cycles: list or array of integers specifying the number of
            [Ry - Rz(theta)] cycles to apply.
        :param sweep_type: dict of the form {'cycles': 0/1, 'seqs': 1/0}, where
            the integers specify which parameter should correspond to the inner
            sweep (0), and which to the outer sweep (1).
        :param kw: keyword arguments
            passed to CalibBuilder; see docstring there

        Assumptions:
         - assumes there is one task for each qubit. If task_list is None, it
          will internally create it.
        """
        try:

            self.sweep_type = sweep_type
            if self.sweep_type is None:
                self.sweep_type = {'cycles': 0, 'seqs': 1}
            self.kw_for_sweep_points['nr_seqs,cycles']['dimension'] = \
                self.sweep_type['seqs']
            self.kw_for_sweep_points['cycles']['dimension'] = \
                self.sweep_type['cycles']
            kw['cal_states'] = kw.get('cal_states', '')

            for task in task_list:
                if 'qb' not in task:
                    raise ValueError('Please specify "qb" in each task in '
                                     '"task_list."')
                if not isinstance(task['qb'], str):
                    task['qb'] = task['qb'].name
                if 'prefix' not in task:
                    task['prefix'] = f"{task['qb']}_"

            # if cycles are not added to each task,
            # self.preprocess_task_list(**kw) will fail because
            # kw_for_sweep_points['nr_seqs,cycles'] requires cycles
            if cycles is not None:
                for task in task_list:
                    task['cycles'] = cycles

            if nr_seqs is not None:  # identical pulses on all qubits
                cycles_list = [''] * len(task_list)
                for i, task in enumerate(task_list):
                    if 'cycles' not in task:
                        raise KeyError('Please specify "cycles" either in '
                                       'the task_list or as input '
                                       'parameter to class init.')
                    cycles_list[i] = task['cycles']
                if np.unique(cycles_list, axis=0).shape[0] > 1:
                    # different qubits have different nr cycles; cannot be if
                    # user wants identical pulses on all qubits
                    raise ValueError('Identical pulses on all qubits requires '
                                     'identical cycles arrays. '
                                     'To use non-identical pulses, '
                                     'move nr_seeds from keyword arguments '
                                     'into the tasks.')
                else:
                    cycles = cycles_list[0]

            super().__init__(task_list, qubits=qubits,
                             sweep_points=sweep_points,
                             nr_seqs=nr_seqs, cycles=cycles, **kw)
            if self.experiment_name is None:
                self.experiment_name = f'XEB'
            self.identical_pulses = nr_seqs is not None and all([
                task.get('nr_seqs', None) is None for task in task_list])
            self.preprocessed_task_list = self.preprocess_task_list(**kw)
            self.sequences, self.mc_points = self.sweep_n_dim(
                self.sweep_points, body_block=None,
                body_block_func=self.xeb_block, cal_points=self.cal_points,
                ro_qubits=self.meas_obj_names, **kw)

            self.exp_metadata['identical_pulses'] = self.identical_pulses

            self.add_processing_pipeline(**kw)
            self.autorun(**kw)

        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def xeb_block(self, sp1d_idx, sp2d_idx, **kw):
        pulse_op_codes_list = []
        tl = [self.preprocessed_task_list[0]] if self.identical_pulses else \
            self.preprocessed_task_list
        for i, task in enumerate(tl):
            seq_idx = sp1d_idx if self.sweep_type['seqs'] == 0 else sp2d_idx
            nrcyc_idx = sp1d_idx if self.sweep_type['cycles'] == 0 else sp2d_idx
            z_angles = task['sweep_points'].get_sweep_params_property(
                'values', self.sweep_type['seqs'], 'z_rots')[seq_idx][nrcyc_idx]
            l = [['Y90', f'Z{zang}'] for zang in z_angles]
            # flatten l, prepend init pulse Y90, append to pulse_op_codes_list
            pulse_op_codes_list += [['Y90'] + [e1 for e2 in l for e1 in e2]]

        rb_block_list = [self.block_from_ops(
            f"rb_{task['qb']}", [f"{p} {task['qb']}" for p in
                                 pulse_op_codes_list[0 if
                                 self.identical_pulses else i]])
            for i, task in enumerate(self.preprocessed_task_list)]

        return self.simultaneous_blocks(f'sim_rb_{sp1d_idx}{sp1d_idx}',
                                        rb_block_list, block_align='end')

    def add_processing_pipeline(self, **kw):
        """
        Creates and adds the analysis processing pipeline to exp_metadata.
        """
        pass

    def run_analysis(self, **kw):
        """
        Runs analysis and stores analysis instance in self.analysis.
        :param kw: keyword_arguments passed to analysis functions;
            see docstrings there
        """
        pass


class TwoQubitXEB(MultiTaskingExperiment):
    kw_for_sweep_points = {'cycles': dict(param_name='cycles', unit='',
                                          label='Nr. Cycles', dimension=0),
                           'nr_seqs,cycles': dict(
                               param_name='gateschoice', unit='',
                               label='cycles gates', dimension=1,
                               values_func='paulis_gen_func')
                           }

    def __init__(self, task_list, sweep_points=None, qubits=None,
                 nr_seqs=None, cycles=None, sweep_type=None, **kw):
        """
        Class to run cross-entropy benchmarking experiment on
        one or several qubits in parallel.
        The experiment consists of applying
        [[Ry - Rz(theta)] * nr_cycles for nr_cycles in cycles] nr_seqs times,
        with random values of theta each time.
        :param task_list: list of dicts; see CalibBuilder docstring
        :param sweep_points: SweepPoints class instance
        :param qubits: list of QuDev_transmon class instances
        :param nr_seqs: int specifying the number of times to apply a random
            iteration of a sequence consisting of nr_cycles cycles.
            If nr_seqs is specified and it does not exist in the task_list,
            THEN ALL QUBITS WILL RECEIVE THE SAME PULSES provided they have the
            same cycles array.
        :param cycles: list or array of integers specifying the number of
            [Ry - Rz(theta)] cycles to apply.
        :param sweep_type: dict of the form {'cycles': 0/1, 'seqs': 1/0}, where
            the integers specify which parameter should correspond to the inner
            sweep (0), and which to the outer sweep (1).
        :param kw: keyword arguments
            passed to CalibBuilder; see docstring there
        Assumptions:
         - assumes there is one task for each qubit. If task_list is None, it
          will internally create it.
        """
        try:

            self.sweep_type = sweep_type
            if self.sweep_type is None:
                self.sweep_type = {'cycles': 0, 'seqs': 1}
            self.kw_for_sweep_points['nr_seqs,cycles']['dimension'] = \
                self.sweep_type['seqs']
            self.kw_for_sweep_points['cycles']['dimension'] = \
                self.sweep_type['cycles']
            kw['cal_states'] = kw.get('cal_states', '')

            for task in task_list:
                for k in ['qb_1', 'qb_2']:
                    if k not in task:
                        raise ValueError('Please specify "{k}" in each task in '
                                         '"task_list."')
                    if not isinstance(task[k], str):
                        task[k] = task[k].name
                if 'prefix' not in task:
                    task['prefix'] = f"{task['qb_1']}{task['qb_2']}_"

            # if cycles are not added to each task,
            # self.preprocess_task_list(**kw) will fail because
            # kw_for_sweep_points['nr_seqs,cycles'] requires cycles
            if cycles is not None:
                for task in task_list:
                    task['cycles'] = cycles

            if nr_seqs is not None:  # identical pulses on all qubits
                cycles_list = [''] * len(task_list)
                for i, task in enumerate(task_list):
                    if 'cycles' not in task:
                        raise KeyError('Please specify "cycles" either in '
                                       'the task_list or as input '
                                       'parameter to class init.')
                    cycles_list[i] = task['cycles']
                if np.unique(cycles_list, axis=0).shape[0] > 1:
                    # different qubits have different nr cycles; cannot be if
                    # user wants identical pulses on all qubits
                    raise ValueError('Identical pulses on all qubits requires '
                                     'identical cycles arrays. '
                                     'To use non-identical pulses, '
                                     'move nr_seeds from keyword arguments '
                                     'into the tasks.')
                else:
                    cycles = cycles_list[0]

            super().__init__(task_list, qubits=qubits,
                             sweep_points=sweep_points,
                             nr_seqs=nr_seqs, cycles=cycles, **kw)
            if self.experiment_name is None:
                self.experiment_name = f'TwoQubitXEB'
            self.identical_pulses = nr_seqs is not None and all([
                task.get('nr_seqs', None) is None for task in task_list])
            self.preprocessed_task_list = self.preprocess_task_list(**kw)
            self.sequences, self.mc_points = self.sweep_n_dim(
                self.sweep_points, body_block=None,
                body_block_func=self.xeb_block, cal_points=self.cal_points,
                ro_qubits=self.meas_obj_names, **kw)

            self.exp_metadata['identical_pulses'] = self.identical_pulses

            self.add_processing_pipeline(**kw)
            self.autorun(**kw)

        except Exception as x:
            self.exception = x
            traceback.print_exc()

    @staticmethod
    def paulis_gen_func(nr_seqs, cycles):
        def gen_random(cycles):
            s_gates = ["X90 ", "Y90 ", "Z45 "]
            lis = []
            for length in cycles:
                i = 0
                gates = []
                gates.append(s_gates[1] + "qb_1")
                sim_str = ' ' if 'Z' in s_gates[1][0:3] else 's '
                gates.append(s_gates[1][0:3] + sim_str + "qb_2")
                gates.append(s_gates[2] + "qb_1")
                sim_str = ' ' if 'Z' in s_gates[2][0:3] else 's '
                gates.append(s_gates[2][0:3] + sim_str + "qb_2")
                gates.append("CZ " + "qb_1 qb_2")
                while i < length:
                    last_1_gate1 = gates[-3][0:4]

                    choice1 = []
                    for gate in s_gates:
                        choice1.append(gate)
                    choice1.remove(last_1_gate1)
                    gate1 = random.choice(choice1)
                    gates.append(gate1 + 'qb_1')

                    last_1_gate2 = gates[-3][0:3] + ' '
                    choice2 = []
                    for gate in s_gates:
                        choice2.append(gate)
                    choice2.remove(last_1_gate2)
                    gate2 = random.choice(choice2)
                    sim_str = ' ' if 'Z' in gate2[:3] else 's '
                    gates.append(gate2[:3] + sim_str + 'qb_2')
                    gates.append("CZ " + 'qb_1 qb_2')
                    i += 1
                lis.append(gates)
            return lis
        return [gen_random(cycles) for _ in range(nr_seqs)]

    def xeb_block(self, sp1d_idx, sp2d_idx, **kw):
        pulse_op_codes_list = []
        tl = [self.preprocessed_task_list[0]] if self.identical_pulses else \
            self.preprocessed_task_list

        for i, task in enumerate(tl):
            seq_idx = sp1d_idx if self.sweep_type['seqs'] == 0 else sp2d_idx
            nrcyc_idx = sp1d_idx if self.sweep_type['cycles'] == 0 else sp2d_idx
            gates_qb_info = task['sweep_points'].get_sweep_params_property(
                'values', self.sweep_type['seqs'], 'gateschoice')[seq_idx][nrcyc_idx]
            l = [gate_qb for gate_qb in gates_qb_info]
            sub_list = []
            for ope in l:
                if len(ope) < 11:
                    op = ope[0:-4] + task[ope[-4::]]
                else:
                    op = 'CZ ' + task[ope[3:7]] + ' ' +  task[ope[8::]]
                sub_list.append(op)
            pulse_op_codes_list += [sub_list]

        rb_block_list = [
            self.block_from_ops(
                f"rb_{task['qb_1']}{task['qb_2']}",
                [op_list for op_list in pulse_op_codes_list[0 if
                self.identical_pulses else i]])
            for i, task in enumerate(self.preprocessed_task_list)]

        return self.simultaneous_blocks(f'sim_rb_{sp1d_idx}{sp1d_idx}',
                                        rb_block_list, block_align='end')

    def add_processing_pipeline(self, **kw):
        """
        Creates and adds the analysis processing pipeline to exp_metadata.
        """
        pass

    def run_analysis(self, **kw):
        """
        Runs analysis and stores analysis instance in self.analysis.
        :param kw: keyword_arguments passed to analysis functions;
            see docstrings there
        """
        pass
