from pprint import pprint

import numpy as np
from numpy import array # Do not remove, used in eval(str_with_array)
from copy import deepcopy
import pycqed.measurement.waveform_control.sequence as sequence
import pycqed.analysis_v2.tomography_qudev as tomo
from pycqed.measurement.pulse_sequences.multi_qubit_tek_seq_elts import get_tomography_pulses
from pycqed.measurement.pulse_sequences.single_qubit_tek_seq_elts import \
    get_pulse_dict_from_pars, add_preparation_pulses, pulse_list_list_seq, prepend_pulses
import pycqed.measurement.waveform_control.segment as segment
from pycqed.measurement.waveform_control.block import Block
from pycqed.measurement.waveform_control import pulsar as ps
import itertools

# TODO: Move this function to more meaningfull place where others can use it
def correlate_qubits(qubit_states, correlations='all', correlator='z',
                     average=True):
    """
    Returns correlations on the given qubit_states.
    Args:
        qubit_states (array): (n_shots, n_qubits) where each digit is the
            assigned state of a qubit (0 or 1).
        coorelations (list): list of tuples indicating which qubits have to be
            correlated, where each tuple indicates the column index of the qubit.
            Eg. [(0,1),(1,2)] will correlate qb0 and qb1, and then qb1 and qb2
            for qubit_states where qi is the ith column of qubit_states.
            defaults to "all" which takes all two qubit correlators
        correlator: 'z' corresponding to sigma_z pauli matrix. Function can
            easily be extended to support other correlators.


    Returns:
        correlations_output (array): (n_shots, n_correlations) if average == True
            else (n_correlations,)
    """
    if correlator == 'z':
        pauli_mtx = np.array([[1, 0], [0, -1]])  # sigma_z matrix
        # correlator matrix for 2 qubits correlator. Can be extended to more
        # qubits by recursively taking tensor product
        corr_mtx = np.kron(pauli_mtx, pauli_mtx)  # sigma_z tensorproduct sigma_z
    else:
        raise NotImplementedError("non 'z' correlators are not yet supported.")

    n_shots, n_qubits = qubit_states.shape
    if correlations == "all":
        correlations = list(itertools.combinations(np.arange(n_qubits), 2))

    correlated_output = np.zeros((n_shots, len(correlations)))
    for j, corr in enumerate(correlations):
        qb_states_to_correlate = []
        for i in corr:
            qb_states_to_correlate.append(qubit_states[:, i])
        # transform to 2 qb basis ie 00 --> 1000, 10 = 0100,...
        qb_states_to_correlate = basis_transformation(np.array(qb_states_to_correlate))
        # < phi | corr_mtx | phi> for all phis simultaneously
        correlated_shots = (qb_states_to_correlate.T @ corr_mtx) @ qb_states_to_correlate
        # only elements on diagonal are relevant
        correlated_shots = correlated_shots[np.eye(n_shots, dtype=bool)]
        correlated_output[:, j] = correlated_shots

    return np.mean(correlated_output, axis=0) if average else correlated_output

def average_sigmaz(qubit_states):
    """
     Returns average sigmaz on the given qubit_states,
     i.e average state of a qubit
    Args:
        qubit_states (array): (n_shots, n_qubits) where each digit is the
            assigned state of a qubit (0 or 1).
    """
    return np.mean(qubit_states, axis=0)

class ProblemHamiltonians:

    @staticmethod
    def ising(avg_sigma_z_sigma_z_corr, C):
        """
        $H = \sum_{i,j} C_{i,j} (I - \sigma_{z_i} \otimes \sigma_{z_j})$
        Args:
            avg_sigma_z_sigma_z_corr:
            C:

        Returns:

        """
        return np.sum([Ci * (1 - corr)
                       for Ci, corr in zip(C, avg_sigma_z_sigma_z_corr)])
    @staticmethod

    def ising_with_field(avg_sigma_z_sigma_z_corr, avg_sigmaz, C, h):
        """
        $H = \sum_{i<j}^{M} C_{i,j} \langle\sigma_{z_i} \sigma_{z_j}\rangle +
        \sum_{i}^N h_i \langle\sigma_{z_i}\rangle$
        Sum of M two qubit terms and N single qubit terms weighted by Cs and hs.
        Args:
            avg_sigma_z_sigma_z_corr (array/list): shape (M,) average correlations
            C (array/list): corresponding weighting factor for each of M correlations
            avg_sigmaz (array/list): shape (N,) average single qubit term
            h (array/list): corresponding weighting factor for each of the  N qubits
        Returns:

        """
        assert len(C) == len(avg_sigma_z_sigma_z_corr), \
            f"Inconsistent number of correlations and weights (C):" \
                f" {avg_sigma_z_sigma_z_corr} vs {len(C)} "
        assert len(h) == len(avg_sigmaz), \
            f"Inconsistent number of single qubit terms and weights (h):" \
                f" {avg_sigmaz} vs {len(h)} "
        two_qb_terms = np.sum([Ci * corr
                               for Ci, corr in zip(C, avg_sigma_z_sigma_z_corr)])
        single_qb_terms = np.sum([hi * qbi for hi, qbi in zip(h, avg_sigmaz)])
        return two_qb_terms + single_qb_terms

# TODO: move this function to more meaningful place
def basis_transformation(qb_array_in_01_basis):
    """
    Transforms qubit string to qubits encoded in 2^n_qubits basis. Eg for 2 qubits:
    [0,0]^T --> [1,0,0,0]^T
    [0,1]^T --> [0,1,0,0]^T
    [1,0]^T --> [0,0,1,0]^T
    [1,1]^T --> [0,0,0,1]^T
    qb_array_in_01_basis (array): (n_qubits, n_shots)
    Returns:
        array (2^n_qubits, n_shots)
    """
    n_qubits, n_shots = qb_array_in_01_basis.shape
    inversions = np.logical_not(
        list(itertools.product((0, 1), repeat=n_qubits)))  # (2^n_qubits, n_qubits)

    # repeat the inversion for all shots
    inversions = \
        np.tile(inversions, (n_shots, 1)).reshape(n_shots, 2 ** n_qubits, n_qubits)

    # expand qubits to new basis space (n_shots, basis_length, n_qubits)
    trans = np.tile(qb_array_in_01_basis, 2 ** n_qubits).reshape(n_qubits,
                                                                 2 ** n_qubits,
                                                                 n_shots).T
    # inverse
    trans = np.logical_not(trans, out=trans, where=inversions)

    # and
    trans = np.all(trans, axis=-1)
    return trans.astype(np.int).T

def qaoa_sequence(qb_names, betas, gammas, two_qb_gates_info, operation_dict,
                  init_state='0', cphase_implementation='hardware',
                  single_qb_terms=None,
                  tomography=False, tomo_basis=tomo.DEFAULT_BASIS_ROTS,
                  cal_points=None, prep_params=None, upload=True):

    # create sequence, segment and builder
    seq_name = f'QAOA_{cphase_implementation}_cphase_{qb_names}'

    seq = sequence.Sequence(seq_name)

    builder = QAOAHelper(qb_names, operation_dict)

    prep_params = {} if prep_params is None else prep_params

    # tomography pulses
    tomography_segments = (None,)
    if tomography:
        tomography_segments = \
            get_tomography_pulses(*qb_names, basis_pulses=tomo_basis)

    for i, ts in enumerate(tomography_segments):
        seg_name = f'segment_{i}' if ts is None else  f'segment_{i}_tomo_{i}'
        seg = segment.Segment(seg_name)

        # initialize qubits
        seg.extend(builder.initialize(init_state, prep_params=prep_params).build())

        # QAOA Unitaries
        for k, (gamma, beta) in enumerate(zip(gammas, betas)):
            # # Uk
            seg.extend(builder.U(f"U_{k}", two_qb_gates_info,
                       gamma, cphase_implementation, single_qb_terms).build())
            # # Dk
            seg.extend(builder.D(f"D_{k}", beta).build())

        # add tomography pulses if required
        if ts is not None:
            seg.extend(builder.block_from_ops(f"tomography_{i}", ts).build())

        # readout qubits
        seg.extend(builder.mux_readout().build())

        seq.add(seg)

    # add calibration points
    if cal_points is not None:
        seq.extend(cal_points.create_segments(operation_dict, **prep_params))

    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)

    return seq, np.arange(seq.n_acq_elements())

class HelperBase:

    STD_INIT = {'0': 'I', '1': 'X180', '+': 'Y90', '-': 'mY90'}

    def __init__(self, qb_names, operation_dict):
        self.qb_names = qb_names
        self.operation_dict = operation_dict

    def get_qubits(self, qubits='all'):
        """
        Wrapper to get 'all' qubits, single qubit specified as string
        or list of qubits, checking they are in self.qb_names
        :param qubits: 'all', single qubit name (eg. 'qb1') or list of qb names
        :return: list of qb names
        """
        if qubits == 'all':
            return self.qb_names
        elif qubits in self.qb_names:  # qubits == single qb name eg. 'qb1'
             qubits = [qubits]
        for qb in qubits:
            assert qb in self.qb_names, f"{qb} not found in {self.qb_names}"
        return qubits

    def get_pulse(self, op, parse_z_gate=False):
        """
        Gets a pulse from the operation dictionary, and possibly parses
        arbitrary angle from Z gate operation.
        Examples:
             >>> get_pulse(['Z100 qb1'], parse_z_gate=True)
             will perform a 100 degree Z rotation
        Args:
            op: operation
            parse_z_gate: whether or not to look for Zgates with arbitrary angles.

        Returns: deepcopy of the pulse dictionary

        """
        if parse_z_gate and op.startswith("Z"):
            # assumes operation format of f"Z{angle} qbname"
            # FIXME: This parsing is format dependent and is far from ideal but
            #  until we can get parametrized pulses it is helpful to be able to
            #  parse Z gates
            angle, qbn = op.split(" ")[0][1:], op.split(" ")[1]
            p = self.get_pulse(f"Z180 {qbn}", parse_z_gate=False)
            p['basis_rotation'] = {qbn: float(angle)}
            return p

        return deepcopy(self.operation_dict[op])

    def initialize(self, init_state='0', qubits='all', prep_params=None,
                   simultaneous=True, block_name=None):
        """
        Initializes the specified qubits with the corresponding init_state
        :param init_state (String or list): Can be one of the following
            - one of the standard initializations: '0', '1', '+', '-'.
              In that case the same init_state is done on all qubits
            - list of standard init. Must then be of same length as 'qubits' and
              in the same order.
            - list of arbitrary pulses (which are in the operation_dict). Must be
              of the same lengths as 'qubits' and in the same order. Should not
              include space and qubit name (those are added internally).
        :param qubits (list or 'all'): list of qubits on which init should be
            applied. Defaults to all qubits.
        :param prep_params: preparation parameters
        :return: init segment
        """
        if block_name is None:
            block_name = f"Initialization_{qubits}"
        qubits = self.get_qubits(qubits)
        if prep_params is None:
            prep_params = {}
        if np.ndim(init_state) == 0:
            init_state = [init_state] * len(qubits)
        else:
            assert len(init_state) == len(qubits), \
                "There must be a one to one mapping between initializations and " \
                f"qubits. Got {len(init_state)} init and {len(qubits)} qubits"

        pulses = []
        pulses.extend(self.prepare(qubits, ref_pulse="start",
                                   **prep_params).build())
        for i, (qbn, init) in enumerate(zip(qubits, init_state)):
            # add qb name and "s" for reference to start of previous pulse
            op = self.STD_INIT.get(init, init) + \
                 f"{'s' if len(pulses) != 0 and simultaneous else ''} " + qbn
            pulse = self.get_pulse(op)
            # if i == 0:
            #     pulse['ref_pulse'] = 'segment_start'
            pulses.append(pulse)
        return Block(block_name, pulses)

    def prepare(self, qubits='all', ref_pulse='start', preparation_type='wait',
                post_ro_wait=1e-6, ro_separation=1.5e-6, reset_reps=3,
                final_reset_pulse=False, threshold_mapping=None, block_name=None):
        """
        Prepares specified qb for an experiment by creating preparation pulse for
        preselection or active reset.
        Args:
            qubits: which qubits to prepare. Defaults to all.
            ref_pulse: reference pulse of the first pulse in the pulse list.
                reset pulse will be added in front of this. If the pulse list is empty,
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
            block_name = f"Preparation_{qubits}"
        qb_names = self.get_qubits(qubits)


        if threshold_mapping is None:
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
                        "Active reset for the f-level requires a mapping of length 4" \
                            f" but only {len(threshold_mapping)} were given: " \
                            f"{threshold_mapping}"
                    ops_and_codewords[qbn] = [
                        (state_ops[threshold_mapping[qbn][0]], 0),
                        (state_ops[threshold_mapping[qbn][1]], 1),
                        (state_ops[threshold_mapping[qbn][2]], 2),
                        (state_ops[threshold_mapping[qbn][3]], 3)]
                else:
                    raise ValueError(f'Invalid preparation type {preparation_type}')

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
                                    pulse_length += reset_pulses[-1 - jj]['pulse_length']
                                else:
                                    pulse_length += reset_pulses[-1 - jj]['sigma'] * \
                                                    reset_pulses[-1 - jj]['nr_sigma']
                            reset_pulses[-1]['pulse_delay'] = post_ro_wait + pulse_length

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
                    ro_list[0]['ref_pulse'] = 'refpulse_reset_element_{}'.format(
                        rep - 1)
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

    def mux_readout(self, qubits='all', element_name='RO',ref_point='end',
                    pulse_delay=0.0):
        block_name = "Readout"
        qubits = self.get_qubits(qubits)
        ro_pulses = []
        for j, qb_name in enumerate(qubits):
            ro_pulse = deepcopy(self.operation_dict['RO ' + qb_name])
            ro_pulse['name'] = '{}_{}'.format(element_name, j)
            ro_pulse['element_name'] = element_name
            if j == 0:
                ro_pulse['pulse_delay'] = pulse_delay
                ro_pulse['ref_point'] = ref_point
            else:
                ro_pulse['ref_point'] = 'start'
            ro_pulses.append(ro_pulse)
        return Block(block_name, ro_pulses)

    def Z_gate(self, theta=0, qubits='all'):

        """
        Software Z-gate of arbitrary rotation.

        :param theta:           rotation angle, in degrees
        :param qubits:      pulse parameters (dict)

        :return: Pulse dict of the Z-gate
        """
        # if qubits is the name of a qb, expects single pulse output
        single_qb_given = False
        if qubits in self.qb_names:
            single_qb_given = True
        qubits = self.get_qubits(qubits)

        pulses = []
        zgate_base_name = 'Z180'

        for qbn in qubits:
            zgate = deepcopy(self.operation_dict[zgate_base_name + f" {qbn}"])
            zgate['basis_rotation'] = {qbn: theta}
            pulses.append(zgate)

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
        :param pulse_modifs (dict): keys are the index of the pulses on which the pulse
            modifications should be made, values are dictionaries of modifications
            Eg. ops = ["X180 qb1", "Y90 qb2"],
            pulse_modifs = {1: {"ref_point": "start"}}
            This will modify the pulse "Y90 qb2" and reference it to the start
            of the first one.
        :return:
        """
        if fill_values is None:
            fill_values = {}
        if pulse_modifs is None:
            pulse_modifs = {}

        pulses = [self.get_pulse(op.format(**fill_values), True)
                  for op in operations]

        # modify pulses given the modifications
        [pulses[i].update(pm) for i, pm in pulse_modifs.items()]
        return Block(block_name, pulses)

class QAOAHelper(HelperBase):

    def U(self, name, gate_sequence_info, gamma, cphase_implementation,
          single_qb_terms=None):
        """
        Returns Unitary propagator pulse sequence (as a Block).
        :param name: name of the block
        :param gate_sequence_info (list) : list of list of information
            dictionaries. Dictionaries contain information about a two QB gate:
            assumes the following keys:
            - qbc: control qubit
            - qbt: target qubit
            - gate_name: name of the 2 qb gate
            - C: coupling btw the two qubits
            - (phase_func) (str): String representation of function predicting
                amplitude and dynamic phase for given target conditional phase.
                Only required when using hardware implementation
               of arbitrary phase gate.
            - (zero_angle_strategy):
                'skip_gate': skips the two qb gate
                'zero_amplitude': forces flux amplitude to zero
                 dict with keys "amplitude", "dynamic_phase": overwrite ampl and dynphase
                 not specified: treated as any other angle with phase_func
            All dictionaries within the same sub list are executed simultaneously
            Example:
            >>> [
            >>>     # first set of 2qb gates to run together
            >>>     [dict(qbc='qb1', qbt='qb2', gate_name='upCZ qb2 qb1', C=1,
            >>>           phase_func=func_qb1_qb2),
            >>>      dict(qbc='qb4', qbt='qb3', gate_name='upCZ qb4 qb3', C=1,
            >>>           phase_func=func_qb4_qb3)],
            >>>     # second set of 2qb gates
            >>>     [dict(qbc='qb3', qbt='qb2', gate_name='upCZ qb2 qb3', C=1,
            >>>        phase_func=func_qb3_qb2)]
            >>> ]
        :param gamma: rotation angle (in rad)
        :param cphase_implementation: implementation of arbitrary phase gate.
            "software" --> gate is decomposed into single qb gates and 2x CZ gate
            "hardware" --> hardware arbitrary phase gate
        :param single_qb_terms (dict): keys are all qubits of experiment
            and values are the h weighting factor for that qubit.
        :return: Unitary U (Block)
        """

        assert cphase_implementation in ("software", "hardware")

        U = Block(name, [])
        for i, two_qb_gates_same_timing in enumerate(gate_sequence_info):
            simult_bname = f"simultanenous_{i}"
            simultaneous = Block(simult_bname, [])
            for two_qb_gates_info in two_qb_gates_same_timing:
                #gate info
                qbc = two_qb_gates_info["qbc"]
                qbt = two_qb_gates_info["qbt"]
                gate_name = two_qb_gates_info['gate_name']
                C = two_qb_gates_info["C"]

                if cphase_implementation == "software":
                    two_qb_block = \
                        self._U_qb_pair_software_decomposition(
                            qbc, qbt, gamma, C, gate_name,
                            f"software qbc:{qbc} qbt:{qbt}")
                elif cphase_implementation == "hardware":
                    # TODO: clean up in function just as above
                    #virtual gate on qb 0
                    z_qbc = self.Z_gate(-2 * gamma * C * 180 / np.pi, qbc)

                    # virtual gate on qb 1
                    z_qbt = self.Z_gate(-2 * gamma * C * 180 / np.pi, qbt)

                    #arbitrary phase gate
                    c_arb_pulse = deepcopy(self.operation_dict[gate_name])
                    #get amplitude and dynamic phase from model
                    angle = -4 * gamma * C
                    angle = angle % (2*np.pi)
                    ampl, dyn_phase = eval(two_qb_gates_info['phase_func'])(angle)

                    # overwrite angles for angle % 2 pi  == 0
                    if angle == 0:
                        strategy = two_qb_gates_info.get("zero_angle_strategy", None)
                        if strategy == "zero_amplitude":
                            ampl, dyn_phase = 0, 0
                        elif strategy == "skip_gate":
                            two_qb_block = Block(f"qbc:{qbc} qbt:{qbt}",
                                                 [z_qbc, z_qbt])
                            simultaneous.extend(two_qb_block.build(ref_pulse=f"start"))
                            continue
                        elif isinstance(strategy, dict):
                            ampl = strategy.get("amplitude", ampl)
                            dyn_phase = strategy.get("dynamic_phase", dyn_phase)
                        elif strategy is None:
                            pass
                        else:
                            raise ValueError(f"Zero angle strategy {strategy} not "
                                             f"understood")
                    print(f"{name}:\nphase angle: {angle}\nAmpl: {ampl}\ndyn_phase: {dyn_phase}")
                    c_arb_pulse['amplitude'] = ampl
                    c_arb_pulse['element_name'] = "flux_arb_gate"
                    c_arb_pulse['basis_rotation'].update(
                        {two_qb_gates_info['qbc']: dyn_phase})

                    two_qb_block = Block(f"qbc:{qbc} qbt:{qbt}",
                                         [z_qbc, z_qbt, c_arb_pulse])

                simultaneous.extend(
                    two_qb_block.build(ref_pulse=f"start"))
            # add block referenced to start of U_k
            U.extend(simultaneous.build())

        # add single qb z rotation for single qb terms of hamiltonian
        if single_qb_terms is not None:
            for qb, h in single_qb_terms.items():
                U.extend([self.Z_gate(2 * gamma * h * 180 / np.pi, qb)])

        return U

    def _U_qb_pair_software_decomposition(self, qbc, qbt, gamma, C, cz_gate_name,
                                          block_name):
        """
        Performs the software decomposition of the QAOA two qubit unitary:
        diag({i phi, -i phi, -i phi, i phi}) where phi = C * gamma.

        Efficient decomposition by Christian :

        H_qbt---CZ---H_qbt---RZ_qbt(2*phi)---H_qbt---CZ---H_qbt
        where:
            H_qbt is a Hadamard gate on qbt (implemented using Y90 + Z180)
            CZ is the control pi-phase gate between qbc and qbt
            RZ_qb(x) is a z rotation of angle x on qb

        :param qbc:
        :param qbt:
        :param gamma:
        :param C:
        :param cz_gate_name:
        :return:
        """
        ops = [ "Z180 {qbt:}","Y90 {qbt:}", cz_gate_name,
               "Z180 {qbt:}", "Y90 {qbt:}", "Z{two_phi:} {qbt:}", "Z180 {qbt:}",
                "Y90 {qbt:}", cz_gate_name, "Z180 {qbt:}", "Y90 {qbt:}"]
        fill_values = dict(qbt=qbt, two_phi= -2*gamma * C * 180/np.pi)

        # put flux pulses in same element
        pulse_modifs = {2: dict(element_name="flux_cz_gate"),
                        8: dict(element_name="flux_cz_gate")}
        return self.block_from_ops(block_name, ops, fill_values, pulse_modifs)

    def D(self, name, beta, qubits='all'):
        if qubits == 'all':
            qubits = self.qb_names

        pulses = []
        ops = ["mY90 {qbn:}", "Z{angle:} {qbn:}", "Y90 {qbn:}"]
        for qbn in qubits:
            D_qbn = self.block_from_ops(f"{qbn}", ops,
                                        dict(qbn=qbn, angle=beta * 180 / np.pi))
            # reference block to beginning of D_k block
            pulses.extend(D_qbn.build(ref_pulse=f"start"))
        return Block(name, pulses)

    @staticmethod
    def get_corr_and_coupl_info(two_qb_gates_info, qb_names=None):
        """
        Helper function to get correlations and couplings used in the sequence
        Correlations are defined as tuples of zero-indexed of qubits: eg.
        (0,1) indicates a correlation will be made on qb1 and qb2
        a coupling is the C between two qubits
        Args:
            two_qb_gates_info: list of list of information
            dictionaries. Dictionaries contain information about a two QB gate:
            assumes the following keys:
            - qbc: control qubit
            - qbt: target qubit
            - gate_name: name of the 2 qb gate
            - C: coupling btw the two qubits
            - (phase_func) (str): String representation of function predicting
                amplitude and dynamic phase for given target conditional phase.
                Only required when using hardware implementation
               of arbitrary phase gate.
            qb_names (list): list of qubit names. if given will return
            correlations based on indices instead of qubit names. Useful to
            apply correlations directly onto an array of states where qubits
            are ordered in the same way
        Returns:
            corr_info (list): list of tuples indicating qubits to correlate:
                by name if qubits not given else by index.
            couplings (list): corresponding coupling for each correlation

        """
        flattened_info = deepcopy([i for info in two_qb_gates_info for i in info])

        if qb_names is not None:
            print(qb_names)
            corr_info = [(qb_names.index(i['qbc']), qb_names.index(i['qbt']))
                         for i in flattened_info]
        else:
            corr_info = [(i['qbc'], i['qbt']) for i in flattened_info]
        couplings = [i['C'] for i in flattened_info]
        return corr_info, couplings