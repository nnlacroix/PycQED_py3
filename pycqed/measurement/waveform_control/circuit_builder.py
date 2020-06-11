import numpy as np
from copy import copy
from copy import deepcopy
from pycqed.measurement.waveform_control.block import Block
from pycqed.measurement.waveform_control.sequence import Sequence
from pycqed.measurement.waveform_control.segment import Segment
from pycqed.measurement import multi_qubit_module as mqm


class CircuitBuilder:
    """
    A class that helps to build blocks, segments, or sequences, e.g.,
    when implementing quantum algorithms.

    :param dev: the device on which the algorithm will be executed
    :param qubits: a list of qubit objects or names if the builder should
        act only on a subset of qubits (default: all qubits of the device)
    :param kw: keyword arguments
         cz_pulse_name: (str) the prefix of CZ gates (default: upCZ)
         prep_params: (dict) custom preparation params (default: from
            instrument settings)
    """

    STD_INIT = {'0': 'I', '1': 'X180', '+': 'Y90', '-': 'mY90'}

    def __init__(self, dev=None, qubits=None, **kw):
        assert (dev is not None or qubits is not None)
        self.dev = dev
        if qubits is None:
            self.qubits = dev.qubits()
        else:
            self.qubits = [dev.get_qb(qb) if isinstance(qb, str) else qb
                           for qb in qubits]
        if dev is None:  # this is only allowed for backwards compatibility
            self.operation_dict = deepcopy(mqm.get_operation_dict(qubits))
        else:
            self.operation_dict = dev.get_operation_dict()
        self.cz_pulse_name = kw.get('cz_pulse_name', 'upCZ')
        self.prep_params = kw.get('prep_params', None)

    def get_qubits(self, qb_names='all'):
        """
        Wrapper to get 'all' qubits, single qubit specified as string
        or list of qubits, checking they are in self.qubits
        :param qb_names: 'all', single qubit name (eg. 'qb1') or list of
            qb names
        :return: list of qb names
        """
        stored_qb_names = [qb.name for qb in self.qubits]
        if qb_names == 'all':
            return self.qubits, stored_qb_names
        elif not isinstance(qb_names, list):
            qb_names = [qb_names]

        # test if qubit indices were provided instead of names
        try:
            ind = [int(i) for i in qb_names]
            qb_names = [stored_qb_names[i] for i in ind]
        except ValueError:
            pass

        for qb in qb_names:
            assert qb in stored_qb_names, f"{qb} not found in {stored_qb_names}"
        return [self.qubits[stored_qb_names.index(qb)] for qb in qb_names], \
               qb_names

    def get_prep_params(self, qb_names='all'):
        qubits, qb_names = self.get_qubits(qb_names)
        if self.prep_params is not None:
            return self.prep_params
        elif self.dev is not None:
            return self.dev.get_prep_params(qubits)
        else:
            return mqm.get_multi_qubit_prep_params(
                [qb.preparation_params() for qb in qubits])

    def get_cz_operation_name(self, qb1, qb2):
        """
        Finds the name of the CZ gate between qubit1-qubit2 that exists in
        self.operation_dict.
        :param qb1: QuDev_transmon instance of one of the gate qubits
        :param qb2: QuDev_transmon instance of the other gate qubit
        :return: the CZ gate name
        """
        if not isinstance(qb1, str):
            qb1 = qb1.name
        if not isinstance(qb2, str):
            qb2 = qb2.name

        if f"{self.cz_pulse_name} {qb1} {qb2}" in self.operation_dict:
            return f"{self.cz_pulse_name} {qb1} {qb2}"
        elif f"{self.cz_pulse_name} {qb2} {qb1}" in self.operation_dict:
            return f"{self.cz_pulse_name} {qb2} {qb1}"
        else:
            raise KeyError(f'CZ gate "{self.cz_pulse_name} {qb1} {qb2}" '
                           f'not found.')

    def get_pulse(self, op, parse_z_gate=False):
        """
        Gets a pulse from the operation dictionary, and possibly parses
        logical indexing as well as arbitrary angle from Z gate operation.
        Examples:
             >>> get_pulse('Z100 qb1', parse_z_gate=True)
             will perform a 100 degree Z rotation
        Args:
            op: operation
            parse_z_gate: whether or not to look for Zgates with arbitrary
            angles.

        Returns: deepcopy of the pulse dictionary

        """
        op_info = op.split(" ")
        # the call to get_qubits resolves qubits indices if needed
        _, op_info[1:] = self.get_qubits(op_info[1:])
        op = op_info[0] + ' ' + ' '.join(op_info[1:])
        if parse_z_gate and op.startswith("Z"):
            # assumes operation format of f"Z{angle} qbname"
            # FIXME: This parsing is format dependent and is far from ideal but
            #  until we can get parametrized pulses it is helpful to be able to
            #  parse Z gates
            angle, qbn = op_info[0][1:], op_info[1]
            p = self.get_pulse(f"Z180 {qbn}", parse_z_gate=False)
            p['basis_rotation'] = {qbn: float(angle)}
        elif op_info[0].startswith('CZ'):
            operation = self.get_cz_operation_name(op_info[1], op_info[2])
            p = deepcopy(self.operation_dict[operation])
        else:
            p = deepcopy(self.operation_dict[op])
        p['op_code'] = op
        return p

    def swap_qubit_indices(self, i, j=None):
        """
        Swaps logical qubit indices by swapping the entries in self.qubits.
        :param i: (int or iterable): index of the first qubit to be swapped or
            indices of the two qubits to be swapped (as two ints given in the
            first two elements of the iterable)
        :param j: index of the second qubit (if it is not set via param i)
        """
        if j is None:
            i, j = i[0], i[1]
        self.qubits[i], self.qubits[j] = self.qubits[j], self.qubits[i]

    def initialize(self, init_state='0', qb_names='all', prep_params=None,
                   simultaneous=True, block_name=None):
        """
        Initializes the specified qubits with the corresponding init_state
        :param init_state (String or list): Can be one of the following
            - one of the standard initializations: '0', '1', '+', '-'.
              In that case the same init_state is done on all qubits
            - list of standard init. Must then be of same length as 'qubits' and
              in the same order.
            - list of arbitrary pulses (which are in the operation_dict). Must
              be of the same lengths as 'qubits' and in the same order. Should
              not include space and qubit name (those are added internally).
        :param qb_names (list or 'all'): list of qubits on which init should be
            applied. Defaults to all qubits.
        :param prep_params: preparation parameters
        :return: init segment
        """
        if block_name is None:
            block_name = f"Initialization_{qb_names}"
        qubits, qb_names = self.get_qubits(qb_names)
        if prep_params is None:
            prep_params = self.get_prep_params(qb_names)
        if len(init_state) == 1:
            init_state = [init_state] * len(qb_names)
        else:
            assert len(init_state) == len(qb_names), \
                f"There must be a one to one mapping between initializations " \
                f"and qubits. Got {len(init_state)} init and {len(qb_names)} " \
                f"qubits"

        pulses = []
        pulses.extend(self.prepare(qb_names, ref_pulse="start",
                                   **prep_params).build())
        for i, (qbn, init) in enumerate(zip(qb_names, init_state)):
            # add qb name and "s" for reference to start of previous pulse
            op = self.STD_INIT.get(init, init)
            if op != 'I':
                op += f"{'s' if len(pulses) != 0 and simultaneous else ''} " + \
                      qbn
                pulse = self.get_pulse(op)
                pulses.append(pulse)
        return Block(block_name, pulses)

    def prepare(self, qb_names='all', ref_pulse='start',
                preparation_type='wait', post_ro_wait=1e-6,
                ro_separation=1.5e-6, reset_reps=3, final_reset_pulse=False,
                threshold_mapping=None, block_name=None):
        """
        Prepares specified qb for an experiment by creating preparation pulse
        for preselection or active reset.
        Args:
            qb_names: which qubits to prepare. Defaults to all.
            ref_pulse: reference pulse of the first pulse in the pulse list.
                reset pulse will be added in front of this.
                If the pulse list is empty,
                reset pulses will simply be before the block_start.
            preparation_type:
                for nothing: 'wait'
                for preselection: 'preselection'
                for active reset on |e>: 'active_reset_e'
                for active reset on |e> and |f>: 'active_reset_ef'
            post_ro_wait: wait time after a readout pulse before applying reset
            ro_separation: spacing between two consecutive readouts
            reset_reps: number of reset repetitions
            final_reset_pulse: Note: NOT used in this function.
            threshold_mapping (dict): thresholds mapping for each qb

        Returns:

        """
        if block_name is None:
            block_name = f"Preparation_{qb_names}"
        qubits, qb_names = self.get_qubits(qb_names)

        if threshold_mapping is None or len(threshold_mapping) == 0:
            threshold_mapping = {qbn: {0: 'g', 1: 'e'} for qbn in qb_names}

        # Calculate the length of a ge pulse, assumed the same for all qubits
        state_ops = dict(g=["I "], e=["X180 "], f=["X180_ef ", "X180 "])

        # no preparation pulses
        if preparation_type == 'wait':
            return Block(block_name, [])

        # active reset
        elif 'active_reset' in preparation_type:
            reset_ro_pulses = []
            ops_and_codewords = {}
            for i, qbn in enumerate(qb_names):
                reset_ro_pulses.append(self.get_pulse('RO ' + qbn))
                reset_ro_pulses[-1]['ref_point'] = 'start' if i != 0 else 'end'

                if preparation_type == 'active_reset_e':
                    ops_and_codewords[qbn] = [
                        (state_ops[threshold_mapping[qbn][0]], 0),
                        (state_ops[threshold_mapping[qbn][1]], 1)]
                elif preparation_type == 'active_reset_ef':
                    assert len(threshold_mapping[qbn]) == 4, \
                        f"Active reset for the f-level requires a mapping of " \
                        f"length 4 but only {len(threshold_mapping)} were " \
                        f"given: {threshold_mapping}"
                    ops_and_codewords[qbn] = [
                        (state_ops[threshold_mapping[qbn][0]], 0),
                        (state_ops[threshold_mapping[qbn][1]], 1),
                        (state_ops[threshold_mapping[qbn][2]], 2),
                        (state_ops[threshold_mapping[qbn][3]], 3)]
                else:
                    raise ValueError(f'Invalid preparation type '
                                     f'{preparation_type}')

            reset_pulses = []
            for i, qbn in enumerate(qb_names):
                for ops, codeword in ops_and_codewords[qbn]:
                    for j, op in enumerate(ops):
                        reset_pulses.append(self.get_pulse(op + qbn))
                        reset_pulses[-1]['codeword'] = codeword
                        if j == 0:
                            reset_pulses[-1]['ref_point'] = 'start'
                            reset_pulses[-1]['pulse_delay'] = post_ro_wait
                        else:
                            reset_pulses[-1]['ref_point'] = 'start'
                            pulse_length = 0
                            for jj in range(1, j + 1):
                                if 'pulse_length' in reset_pulses[-1 - jj]:
                                    pulse_length += reset_pulses[-1 - jj][
                                        'pulse_length']
                                else:
                                    pulse_length += \
                                        reset_pulses[-1 - jj]['sigma'] * \
                                        reset_pulses[-1 - jj]['nr_sigma']
                            reset_pulses[-1]['pulse_delay'] = post_ro_wait + \
                                                              pulse_length

            prep_pulse_list = []
            for rep in range(reset_reps):
                ro_list = deepcopy(reset_ro_pulses)
                ro_list[0]['name'] = 'refpulse_reset_element_{}'.format(rep)

                for pulse in ro_list:
                    pulse['element_name'] = 'reset_ro_element_{}'.format(rep)
                if rep == 0:
                    ro_list[0]['ref_pulse'] = ref_pulse
                    ro_list[0]['pulse_delay'] = -reset_reps * ro_separation
                else:
                    ro_list[0]['ref_pulse'] = \
                        'refpulse_reset_element_{}'.format(rep - 1)
                    ro_list[0]['pulse_delay'] = ro_separation
                    ro_list[0]['ref_point'] = 'start'

                rp_list = deepcopy(reset_pulses)
                for j, pulse in enumerate(rp_list):
                    pulse['element_name'] = 'reset_pulse_element_{}'.format(rep)
                    pulse['ref_pulse'] = 'refpulse_reset_element_{}'.format(rep)
                prep_pulse_list += ro_list
                prep_pulse_list += rp_list

            # manually add block_end with delay referenced to last readout
            # as if it was an additional readout pulse
            # otherwise next pulse will overlap with codeword padding.
            block_end = dict(name='end', pulse_type="VirtualPulse",
                             ref_pulse=f'refpulse_reset_element_{reset_reps-1}',
                             pulse_delay=ro_separation)
            prep_pulse_list += [block_end]
            return Block(block_name, prep_pulse_list)

        # preselection
        elif preparation_type == 'preselection':
            preparation_pulses = []
            for i, qbn in enumerate(qb_names):
                preparation_pulses.append(self.get_pulse('RO ' + qbn))
                preparation_pulses[-1]['ref_point'] = 'start'
                preparation_pulses[-1]['element_name'] = 'preselection_element'
            preparation_pulses[0]['ref_pulse'] = ref_pulse
            preparation_pulses[0]['name'] = 'preselection_RO'
            preparation_pulses[0]['pulse_delay'] = -ro_separation
            block_end = dict(name='end', pulse_type="VirtualPulse",
                             ref_pulse='preselection_RO',
                             pulse_delay=ro_separation)
            preparation_pulses += [block_end]
            return Block(block_name, preparation_pulses)

    def mux_readout(self, qb_names='all', element_name='RO', **pulse_pars):
        block_name = "Readout"
        qubits, qb_names = self.get_qubits(qb_names)
        ro_pulses = []
        for j, qb_name in enumerate(qb_names):
            ro_pulse = deepcopy(self.operation_dict['RO ' + qb_name])
            ro_pulse['name'] = '{}_{}'.format(element_name, j)
            ro_pulse['element_name'] = element_name
            if j == 0:
                ro_pulse.update(pulse_pars)
            else:
                ro_pulse['ref_point'] = 'start'
            ro_pulses.append(ro_pulse)
        return Block(block_name, ro_pulses)

    def Z_gate(self, theta=0, qb_names='all'):

        """
        Software Z-gate of arbitrary rotation.

        :param theta:           rotation angle, in degrees
        :param qb_names:      pulse parameters (dict)

        :return: Pulse dict of the Z-gate
        """

        # if qb_names is the name of a single qb, expects single pulse output
        single_qb_given = not isinstance(qb_names, list)
        qubits, qb_names = self.get_qubits(qb_names)
        pulses = [self.get_pulse(f'Z{theta} {qbn}', True) for qbn in qb_names]
        return pulses[0] if single_qb_given else pulses

    def block_from_ops(self, block_name, operations, fill_values=None,
                       pulse_modifs=None):
        """
        Returns a block with the given operations.
        Eg.
        >>> ops = ['X180 {qbt:}', 'X90 {qbc:}']
        >>> builder.block_from_ops("MyAwesomeBlock",
        >>>                                ops,
        >>>                                {'qbt': qb1, 'qbc': qb2})
        :param block_name: Name of the block
        :param operations: list of operations (str), which can be preformatted
            and later filled with values in the dictionary fill_values
        :param fill_values (dict): optional fill values for operations.
        :param pulse_modifs (dict): Modification of pulses parameters.
            keys:
             -indices of the pulses on  which the pulse modifications should be
             made (backwards compatible)
             -
             values: dictionaries of modifications
            E.g. ops = ["X180 qb1", "Y90 qb2"],
            pulse_modifs = {1: {"ref_point": "start"}}
            This will modify the pulse "Y90 qb2" and reference it to the start
            of the first one.
        :return: The created block
        """
        if fill_values is None:
            fill_values = {}
        if pulse_modifs is None:
            pulse_modifs = {}
        if isinstance(operations, str):
            operations = [operations]

        pulses = [self.get_pulse(op.format(**fill_values), True)
                  for op in operations]

        return Block(block_name, pulses, pulse_modifs)

    def seg_from_ops(self, operations, fill_values=None, pulse_modifs=None,
                     init_state='0', seg_name='Segment1', ro_kwargs=None):
        """
        Returns a segment with the given operations using the function
        block_from_ops().
        :param operations: list of operations (str), which can be preformatted
            and later filled with values in the dictionary fill_values
        :param fill_values: optional fill values for operations (dict),
            see documentation of block_from_ops().
        :param pulse_modifs: Modification of pulses parameters (dict),
            see documentation of block_from_ops().
        :param init_state: initialization state (string or list),
            see documentation of initialize().
        :param seg_name: Name (str) of the segment (default: "Segment1")
        :param ro_kwargs: Keyword arguments (dict) for the function
            mux_readout().
        :return: The created segment
        """
        if ro_kwargs is None:
            ro_kwargs = {}
        pulses = self.initialize(init_state=init_state).build()
        pulses += self.block_from_ops("Block1", operations,
                                      fill_values=fill_values,
                                      pulse_modifs=pulse_modifs).build()
        pulses += self.mux_readout(**ro_kwargs).build()
        return Segment(seg_name, pulses)

    def seq_from_ops(self, operations, fill_values=None,  pulse_modifs=None,
                     init_state='0', seq_name='Sequence', ro_kwargs=None):
        """
        Returns a sequence with the given operations using the function
        block_from_ops().
        :param operations: list of operations (str), which can be preformatted
            and later filled with values in the dictionary fill_values
        :param fill_values: optional fill values for operations (dict),
            see documentation of block_from_ops().
        :param pulse_modifs: Modification of pulses parameters (dict),
            see documentation of block_from_ops().
        :param init_state: initialization state (string or list),
            see documentation of initialize().
        :param seq_name: Name (str) of the sequence (default: "Sequence")
        :param ro_kwargs: Keyword arguments (dict) for the function
            mux_readout().
        :return: The created sequence
        """
        seq = Sequence(seq_name)
        seq.add(self.seg_from_ops(operations=operations,
                                  fill_values=fill_values,
                                  pulse_modifs=pulse_modifs,
                                  init_state=init_state,
                                  ro_kwargs=ro_kwargs))
        return seq

    def simultaneous_blocks(self, block_name, blocks):
        """
        Creates a block with name :block_name: that consists of the parallel
        execution of the given :blocks:. Ensures that any pulse or block
        following the created block will occur after the longest given block.

        Note that within each of the given blocks, it is assumed that the
        pulse listed last in the block is the one that occurs last.
        TODO: We might want to relax this assumption in a future version!

        Args:
            block_name (string): name of the block that is created
            blocks (iterable): iterable where each element is a block that has
            to be executed in parallel to the others.
        """

        simultaneous = Block(block_name, [])
        simultaneous_end_pulses = []
        for block in blocks:
            simultaneous.extend(block.build(ref_pulse=f"start"))
            simultaneous_end_pulses.append(simultaneous.pulses[-1]['name'])
        simultaneous.extend([{"name": f"simultaneous_end_pulse",
                              "pulse_type": "VirtualPulse",
                              "pulse_delay": 0,
                              "ref_pulse": simultaneous_end_pulses,
                              "ref_point": 'end',
                              "ref_function": 'max'
                              }])
        return simultaneous

    def sequential_blocks(self, block_name, blocks):
        """
        Creates a block with name :block_name: that consists of the serial
        execution of the given :blocks:.

        Args:
            block_name (string): name of the block that is created
            blocks (iterable): iterable where each element is a block that has
            to be executed one after another.
        """

        sequential = Block(block_name, [])
        for block in blocks:
            sequential.extend(block.build())
        return sequential

    def sweep_n_dim(self, body_block, sweep_points, cal_points=None,
                 init_state='0', seq_name='Sequence', ro_kwargs=None,
                 return_segments=False, **kw):
        """
        Creates a sequence or a list of segments by doing an N-dim sweep
        over the given operations based on the sweep_points.
        Currently, only 1D and 2D sweeps are implemented.

        :param body_block: block containing the pulses to be swept (excluding
            initialization and readout)
        :param sweep_points: SweepPoints object
        :param cal_points: CalibrationPoints object
        :param init_state: initialization state (string or list),
            see documentation of initialize().
        :param seq_name: Name (str) of the sequence (default: "Sequence")
        :param ro_kwargs: Keyword arguments (dict) for the function
            mux_readout().
        :param return_segments: whether to return segments or the sequence
        :param kw: keyword arguments (to allow pass through kw even if it
            contains entries that are not needed)
        :return:
            - if return_segments==True:
                1D: list of segments, number of 1d sweep points or
                2D: list of list of segments, list of numbers of sweep points
            - else:
                1D: sequence, number acquisition elements
                2D: list of sequences, number of acquisition elements, number
                    of sequences
        """
        sweep_dims = len(sweep_points)
        if sweep_dims > 2:
            raise NotImplementedError('Only 1D and 2D sweeps are implemented.')

        if ro_kwargs is None:
            ro_kwargs = {}

        nr_sp_list = [len(list(d.values())[0][0]) for d in sweep_points]

        if sweep_dims == 1:
            sweep_points = copy(sweep_points)
            sweep_points.add_sweep_dimension()
            nr_sp_list.append(1)

        prep = self.initialize(init_state=init_state)
        ro = self.mux_readout(**ro_kwargs)
        segblock = self.sequential_blocks('segblock', [prep, body_block, ro])

        seqs = []
        for i in range(nr_sp_list[1]):
            this_seq_name = seq_name + (f'_{i}' if sweep_dims == 2 else '')
            seq = Sequence(this_seq_name)
            for j in range(nr_sp_list[0]):
                seq.add(Segment(f'seg{j}', segblock.build(
                    sweep_dicts_list=sweep_points, sweep_index_list=[j, i])))
            if cal_points is not None:
                seq.extend(cal_points.create_segments(self.operation_dict,
                                                      **self.get_prep_params()))
            seqs.append(seq)

        if return_segments:
            segs = [list(seq.segments.values()) for seq in seqs]
            if sweep_dims == 1:
                return segs[0], nr_sp_list[0]
            else:
                return segs, nr_sp_list

        # repeat UHF seqZ code
        for s in seqs:
            for ro_op in [p['op_code'] for p in ro.pulses]:
                s.repeat_ro(ro_op, self.operation_dict)

        if sweep_dims == 1:
            return seqs, [np.arange(seqs[0].n_acq_elements())]
        else:
            return seqs, [np.arange(seqs[0].n_acq_elements()),
                          np.arange(nr_sp_list[1])]

