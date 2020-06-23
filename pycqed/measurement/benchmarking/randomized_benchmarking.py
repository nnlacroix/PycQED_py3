import numpy as np
import traceback
from pycqed.analysis_v3.processing_pipeline import ProcessingPipeline
from pycqed.measurement.calibration.two_qubit_gates import CalibBuilder
from pycqed.measurement.sweep_points import SweepPoints
from pycqed.measurement.randomized_benchmarking import \
    randomized_benchmarking as rb
from pycqed.analysis_v3 import pipeline_analysis as pla
import logging
log = logging.getLogger(__name__)


class SingleQubitRandomizedBenchmarking(CalibBuilder):
    def __init__(self, dev, task_list=None, sweep_points=None, qubits=None,
                 nr_seeds=None, interleaved_gate=None, gate_decomposition='HZ',
                 identical_pulses=False, **kw):
        """
        Class to run and analyze the randomized benchmarking experiment on
        one or several qubits in parallel, using the single-qubit Clifford group
        :param dev: instance of Device class; see CalibBuilder docstring
        :param task_list: list of dicts; see CalibBuilder docstring
        :param sweep_points: SweepPoints class instance with first sweep
            dimension describing the seeds and second dimension the cliffords
            Ex: [{'seeds': (array([0, 1, 2, 3]), '', 'Nr. Seeds')},
                 {'cliffords': ([0, 4, 10], '', 'Nr. Cliffords')}]
            If it contains only one sweep dimension, this must be the
            cliffords. The seeds will be added automatically.
            If this parameter is provided it will be used for all qubits.
        :param qubits: list of QuDev_transmon class instances
        :param nr_seeds: int specifying the number of times the Clifford
            group should be sampled for each Clifford sequence length.
            If nr_seeds is specified and it does not exist in the SweepPoints
            of each task in task_list, then it will be the same for all qubits
        :param interleaved_gate: string specifying the interleaved gate in
            pycqed notation (ex: X90, Y180 etc). Gate must be part of the
            Clifford group.
        :param gate_decomposition: string specifying what decomposition to use
            to translate the Clifford elements into applicable pulses.
            Possible choices are 'HZ' or 'XY'.
            See HZ_gate_decomposition and XY_gate_decomposition in
            pycqed\measurement\randomized_benchmarking\clifford_decompositions.py
        :param identical_pulses: bool that indicates whether to always apply
            identical pulses on all qubits ie identical Clifford sequence for
            each qubit (True), or to produce a random Clifford sequence for
            each qubit (False)
        :param kw: keyword arguments
            passed to CalibBuilder; see docstring there.

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
            self.nr_seeds = nr_seeds
            self.interleaved_gate = interleaved_gate
            self.gate_decomposition = gate_decomposition

            task_list = self.add_seeds_sweep_points(task_list, self.nr_seeds)
            self.task_list = task_list
            self.identical_pulses = identical_pulses
            # Check if we can apply identical pulses on all qubits in task_list
            # Can only do this if they have identical cliffords array
            one_clf_set = self.task_list[0][
                'sweep_points'].get_sweep_params_property('values', 2)
            unique_clf_sets = np.unique([
                task['sweep_points'].get_sweep_params_property('values', 2)
                for task in self.task_list])
            if len(unique_clf_sets) != len(one_clf_set):
                self.identical_pulses = False

            self.ro_qubits = self.get_ro_qubits()
            self.guess_label()
            # TODO: there is currently no analysis for RB with cal_points
            for_ef = kw.get('for_ef', False)
            kw['for_ef'] = for_ef
            cal_states = kw.get('cal_states', '')
            kw['cal_states'] = cal_states
            self.create_cal_points(**kw)
            self.sweep_points = SweepPoints(from_dict_list=[{}, {}])
            for task in task_list:
                self.preprocess_task(task, self.sweep_points)
            kw.update({'ro_qubits': [qb.name for qb in self.ro_qubits]})
            self.sequences, sp = self.sweep_n_dim(
                self.sweep_points, body_block=None,
                body_block_func=self.rb_block, cal_points=self.cal_points,
                **kw)
            self.hard_sweep_points, self.soft_sweep_points = sp
            self.add_processing_pipeline()

            if self.measure:
                self.run_measurement(**kw)
            if self.analyze:
                self.run_analysis(**kw)

        except Exception as x:
            self.exception = x
            traceback.print_exc()

    @staticmethod
    def add_seeds_sweep_points(task_list, nr_seeds):
        """
        If seeds are not in the sweep_points in each task, but cliffords are,
        then seeds will be added to the sweep_points entry in each task_list.
        :param task_list: list of dictionaries describing the the measurement
            for each qubit.
        :param nr_seeds: int specifying the number of repetitions of each
            Clifford sequence. This function will add np.arange(nr_seeds).
        :return: updated task list

        """
        for task in task_list:
            sweep_points = task['sweep_points']
            if len(sweep_points) == 1:
                # it must be the 2nd sweep dimension, over cliffords
                sweep_points = [{}, sweep_points[0]]
            sweep_points = SweepPoints(from_dict_list=sweep_points)
            hard_sweep_dict = SweepPoints()
            if 'seeds' not in sweep_points[0]:
                if nr_seeds is None:
                    raise ValueError('Please specify nr_seeds or add it to '
                                     'the sweep points.')
                hard_sweep_dict.add_sweep_parameter(
                    'seeds', np.arange(nr_seeds), '', 'Nr. Seeds')
            sweep_points.update(hard_sweep_dict + [{}])
            task['sweep_points'] = sweep_points
        return task_list

    def rb_block(self, sp1d_idx, sp2d_idx, sweep_points, **kw):
        if self.identical_pulses:
            # all qubits have the same cliffords array
            clifford = self.task_list[0][
                'sweep_points'].get_sweep_params_property('values', 2)[sp2d_idx]
            cl_seq = rb.randomized_benchmarking_sequence(
                clifford, interleaved_gate=self.interleaved_gate)
            pulse_keys = rb.decompose_clifford_seq(
                cl_seq, gate_decomp=self.gate_decomposition)
            rb_block_list = [self.block_from_ops(
                f'rb_{qb.name}', [f'{p} {qb.name}' for p in pulse_keys])
                for qb in self.ro_qubits]
        else:
            rb_block_list = []
            for task in self.task_list:
                qbn = task['qubits_to_measure']
                clifford = task['sweep_points'].get_sweep_params_property(
                    'values', 2)[sp2d_idx]
                cl_seq = rb.randomized_benchmarking_sequence(
                    clifford, interleaved_gate=self.interleaved_gate)
                pulse_keys = rb.decompose_clifford_seq(
                    cl_seq, gate_decomp=self.gate_decomposition)
                rb_block_list += [self.block_from_ops(
                    f'rb_{qbn}', [f'{p} {qbn}' for p in pulse_keys])]
        return self.simultaneous_blocks_align_end(f'sim_rb_{clifford}{sp1d_idx}',
                                        rb_block_list)

    def guess_label(self):
        """
        Default measurement label.
        """
        if self.label is None:
            if self.interleaved_gate is None:
                self.label = f'RB_{self.gate_decomposition}' \
                             f'{self.dev.get_msmt_suffix(self.ro_qubits)}'
            else:
                self.label = f'IRB_{self.interleaved_gate}_' \
                             f'{self.gate_decomposition}' \
                             f'{self.dev.get_msmt_suffix(self.ro_qubits)}'

    def add_processing_pipeline(self):
        """
        Creates and adds the analysis processing pipeline to exp_metadata.
        """
        pp = ProcessingPipeline()
        for task in self.task_list:
            cliffords = task['sweep_points'].get_sweep_params_property(
                'values', 2)
            seeds = task['sweep_points'].get_sweep_params_property(
                'values', 1)
            if not self.classified:
                pp.add_node('rotate_iq', keys_in='raw',
                            meas_obj_names=task['qubits_to_measure'],
                            num_keys_out=1)
            pp.add_node('average_data',
                        keys_in='raw' if not self.classified else
                            'previous rotate_iq',
                        shape=(len(cliffords), len(seeds)),
                        meas_obj_names=task['qubits_to_measure'])
            pp.add_node('get_std_deviation',
                        keys_in='raw' if not self.classified else
                            'previous rotate_iq',
                        shape=(len(cliffords), len(seeds)),
                        meas_obj_names=task['qubits_to_measure'])
            pp.add_node('rb_analysis', meas_obj_names=task['qubits_to_measure'],
                        keys_out=None, d=2,
                        keys_in=f'previous average_data',
                        keys_in_std=f'previous get_std_deviation')
        self.exp_metadata.update({'processing_pipe': pp})

    def run_analysis(self, **kw):
        """
        Runs analysis and stores analysis instance in self.analysis.
        :param kw: keyword_arguments passed to analysis functions;
            see docstrings there
        """
        self.analysis = pla.extract_data_hdf(**kw) # returns a dict
        pla.process_pipeline(self.analysis, **kw)