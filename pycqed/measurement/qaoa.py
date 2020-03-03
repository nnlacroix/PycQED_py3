from pprint import pprint

import numpy as np
from numpy import array # Do not remove, used in eval(str_with_array)
from scipy.interpolate import interp1d # Do not remove, used in eval(str_with_interp1d)
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

# TODO: Move this function to more meaningful place where others can use it
def correlate_qubits(qubit_states, correlations='all', correlator='z',
                     average=True):
    """
    Returns correlations on the given qubit_states.
    Args:
        qubit_states (array): (n_shots, n_qubits) where each digit is the
            assigned state of a qubit (0 or 1).
        coorelations (list): list of tuples indicating which qubits have to be
            correlated, where each tuple indicates the column index of the qubit.
            Eg. [(0,1),(1,2,3)] will correlate logical qubits 0 and 1, and then
            calculate the 3-body correlation between logical qubits 1, 2, and 3
            (assuming that the ith column of qubit_states corresponds to the
            ith logical qubit).
            defaults to "all" which takes all two qubit correlators
        correlator: 'z' corresponding to sigma_z pauli matrix. Function could
            later be extended to support other correlators.

    Returns:
        correlations_output (array): (n_shots, n_correlations) if average == True
            else (n_correlations,)
    """
    if correlator == 'z':
        pass
    else:
        raise NotImplementedError("non 'z' correlators are not yet supported.")

    n_shots, n_qubits = qubit_states.shape
    if correlations == "all":
        correlations = list(itertools.combinations(np.arange(n_qubits), 2))

    correlated_output = np.zeros((n_shots, len(correlations)))
    for j, corr in enumerate(correlations):
        qb_states_to_correlate = []
        if type(corr) == int:
            corr = (corr,)
        for i in corr:
            qb_states_to_correlate.append(qubit_states[:, i])
        correlated_output[:, j] = np.prod(1 - np.array(qb_states_to_correlate) * 2, axis=0)

    return np.mean(correlated_output, axis=0) if average else correlated_output

def average_sigmaz(qubit_states):
    """
     Returns average sigmaz on the given qubit_states,
     i.e average state of a qubit
    Args:
        qubit_states (array): (n_shots, n_qubits) where each digit is the
            assigned state of a qubit (0 or 1).
    """
    return np.mean(1 - np.array(qubit_states)*2, axis=0)

class ProblemHamiltonians:

    @staticmethod
    def nbody_zterms(qubit_states, gates_info):
        """
        $H = \sum_{i} J_i \prod_{j \in Q_i} \sigma_{z_j}$ where each $Q_i$ is a subset of qubits
        Args:
            qubit_states:
            gates_info:

        Returns:

        """
        return np.sum(list(ProblemHamiltonians.nbody_zterms_individual(qubit_states,
                                                 gates_info).values()))

    @staticmethod
    def nbody_zterms_individual(qubit_states, gates_info):
        c_info, coupl = QAOAHelper.get_corr_and_coupl_info(gates_info)
        correlations = correlate_qubits(qubit_states, c_info)
        return {info: Ci * corr for info, Ci, corr in zip(c_info, coupl,
                                                          correlations)}

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

def qaoa_sequence(qb_names, betas, gammas, gates_info, operation_dict,
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

    if np.ndim(gammas) < 2:
        gammas = [gammas]
    if np.ndim(betas) < 2:
        betas = [betas]
    for ind_array, (gamma_array, beta_array) in enumerate(zip(gammas,betas)):
        for i, ts in enumerate(tomography_segments):
            seg_name = f'segment_{i}_{ind_array}' if ts is None else  \
                f'segment_{i}_{ind_array}_tomo_{i}'
            seg = segment.Segment(seg_name)

            # initialize qubits
            seg.extend(builder.initialize(init_state, prep_params=prep_params).build())

            # QAOA Unitaries
            gates_info_all = deepcopy(gates_info)
            if 'gate_order' not in gates_info_all:
                gates_info_all['gate_order'] = [[i] for i in range(len(gates_info_all['gate_list']))]
            gates_info_p = deepcopy(gates_info_all)
            for k, (gamma, beta) in enumerate(zip(gamma_array, beta_array)):
                # # Uk
                if isinstance(gates_info_all['gate_order'][0][0],list):
                    gates_info_p['gate_order'] = deepcopy(
                        gates_info_all['gate_order'][k % (len(gates_info_all['gate_order']))])
                seg.extend(builder.U(f"U_{k}", gates_info_p,
                           gamma, cphase_implementation, single_qb_terms,
                                     first_layer=(k==0)).build())
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
        if len(init_state) == 1:
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
        :return:
        """
        if fill_values is None:
            fill_values = {}
        if pulse_modifs is None:
            pulse_modifs = {}

        pulses = [self.get_pulse(op.format(**fill_values), True)
                  for op in operations]

        # modify pulses
        [pulses[i].update(pm) for i, pm in pulse_modifs.items()]
        return Block(block_name, pulses)

class QAOAHelper(HelperBase):

    def U(self, name, gate_sequence_info, gamma, cphase_implementation,
          single_qb_terms=None, first_layer=False):
        """
        Returns Unitary propagator pulse sequence (as a Block).
        :param name: name of the block
        :param gate_sequence_info (dict): has to contain the following keys
            - gate_list: a list of dictionaries, each containing
                information about a two QB gate:
                - qbs: 2-tuple of logical qubit indices
                - gate_name: name of the 2 qb gate type
                - C: coupling btw the two qubits
                - (zero_angle_strategy):
                    'skip_gate': skips the two qb gate
                    'zero_amplitude': forces flux amplitude to zero
                     dict with keys "amplitude", "dynamic_phase": overwrite ampl and dynphase
                     not specified: treated as any other angle with phase_func
                - (zero_angle_threshold): threshold for considering an angle to be zero  (in rad)
                    (default: use global value)
            - gate_order: list of lists of indices from the gate_list
                All gates in the same sublist are executed simultaneously.
            - (phase_func): Dictionary of string representations of functions predicting
                amplitude and dynamic phase for given target conditional phase.
                Only required when using hardware implementation
                of arbitrary phase gate.
            - (zero_angle_threshold): global threshold for considering an angle to be zero (in rad)
                default: 1e-10
            Example:
            >>> dict(
            >>>     phase_func=arb_phase_func_dict,
            >>>     gate_list = [
            >>>      dict(qbs=(0,1), gate_name='upCZ', C=1),
            >>>      dict(qbs=(2,1), gate_name='upCZ', C=1),
            >>>      dict(qbs=(2,3), gate_name='upCZ', C=1)]
            >>>     gate_order = [
            >>>     # first set of 2qb gates to run together
            >>>         [0,2],
            >>>     # second set of 2qb gates
            >>>         [1]
            >>>     ]
            >>> )
        :param gamma: rotation angle (in rad)
        :param cphase_implementation: implementation of arbitrary phase gate.
            "software" --> gate is decomposed into single qb gates and 2x CZ gate
            "hardware" --> hardware arbitrary phase gate
        :param single_qb_terms (dict): keys are all logical qubit indices of experiment
            and values are the h weighting factor for that qubit.
        :param first_layer (bool): only if this is True, remove_1stCZ in
            gates_info will remove the first CZ gate of the software decomposition
        :return: Unitary U (Block)
        """

        assert cphase_implementation in ("software", "hardware")
        global_zero_angle_threshold = gate_sequence_info.get("zero_angle_threshold", 1e-10)

        if single_qb_terms is not None:
            tmp_single_qb_terms = [0]*len(self.qb_names)
            for qb, J in single_qb_terms.items():
                tmp_single_qb_terms[qb] = J
            single_qb_terms = tmp_single_qb_terms
        else:
            single_qb_terms = [0]*len(self.qb_names)
        U = Block(name, [])
        for i, gates_same_timing in enumerate(gate_sequence_info['gate_order']):
            simult_bname = f"simultanenous_{i}"
            simultaneous = Block(simult_bname, [])
            for gates_info in [gate_sequence_info['gate_list'][i]
                               for i in gates_same_timing]:
                #gate info
                C = gates_info['J'] if 'J' in gates_info else gates_info['C'] \
                    if 'C' in gates_info else 0
                doswap = gates_info.get("swap", False)
                if type(gates_info['qbs']) == int:
                    gates_info['qbs'] = (gates_info['qbs'],)
                if len(gates_info['qbs']) == 1:
                    single_qb_terms[gates_info['qbs'][0]] += C
                    continue
                gates_info['gate_name'] = \
                    gates_info['gate_name'] if 'gate_name' in gates_info else 'upCZ'
                remove_1stCZ = gates_info.get('remove_1stCZ', '')
                strategy = gates_info.get("zero_angle_strategy", None)
                nbody = (len(gates_info['qbs'])>2)
                assert not (nbody and doswap), \
                    f"Combination of n-body interaction and swap is not implemented!"
                zero_angle_threshold = gates_info.get("zero_angle_threshold",
                                                      global_zero_angle_threshold)
                if abs((2 * gamma * C) % (2*np.pi))<zero_angle_threshold \
                        and strategy == "skip_gate" and not doswap:
                    continue
                for qbx in [self.qb_names[qb_ind] for qb_ind in gates_info['qbs']]:
                    for qby_tmp in [self.qb_names[qb_ind] for qb_ind in gates_info['qbs']]:
                        if qby_tmp == qbx:
                            continue
                        qby = qby_tmp
                        qbt, qbc = qbx, qby
                        gate_name = f"{gates_info['gate_name']} {qbt} {qbc}"
                        if gate_name not in self.operation_dict:
                            qbt,qbc = qby,qbx
                            gate_name = f"{gates_info['gate_name']} {qbt} {qbc}"
                            if gate_name not in self.operation_dict:
                                break
                    else:
                        break
                else:
                    assert False, \
                    f"The logical qubits {gates_info['qbs']} are currently " \
                        f"not connected by a {gates_info['gate_name']} gate!"
                if nbody:
                    opsH = ["Z180 {qbx:}", "Y90 {qbx:}"] # Hadamard gate
                    nbody_start = self.block_from_ops(f"Had", opsH, dict(qbx=qbx), {}).build()
                    nbody_end = []
                    if cphase_implementation != "software":
                        nbody_end = self.block_from_ops(f"Had", opsH, dict(qbx=qbx), {}).build()
                    for qbz in [self.qb_names[qb_ind] for qb_ind in gates_info['qbs']]:
                        if qbz==qbx or qbz==qby:
                            continue
                        qbz_gate_name = f"{gates_info['gate_name']} {qbx} {qbz}";
                        if gate_name not in self.operation_dict:
                            qbz_gate_name = f"{gates_info['gate_name']} {qbz} {qbx}";
                        nbody_cz = self.block_from_ops(f"CZ {qbz}", [qbz_gate_name],
                            {}, {0: dict(element_name="flux_arb_gate")}).build();
                        nbody_start.extend(nbody_cz)
                        nbody_end.extend(nbody_cz)
                    if cphase_implementation != "software":
                        nbody_start.extend(self.block_from_ops(f"Had2", opsH, dict(qbx=qbx), {}).build())
                    nbody_end.extend(self.block_from_ops(f"Had2", opsH, dict(qbx=qbx), {}).build())

                #virtual gate on qb 0
                z_qbc = self.Z_gate(2 * gamma * C * 180 / np.pi, qbc)
                # virtual gate on qb 1
                z_qbt = self.Z_gate(2 * gamma * C * 180 / np.pi, qbt)

                if cphase_implementation == "software":
                    if doswap:
                        two_qb_block = Block(f"qbc:{qbc} qbt:{qbt}", [z_qbc, z_qbt])
                        two_qb_block.extend(
                            self._U_qb_pair_fermionic_simulation(
                                qbc, qbt, np.pi - 4 * gamma * C, gate_name,
                                f"FSIM").build())
                    else:
                        two_qb_block = \
                            self._U_qb_pair_software_decomposition(
                                qbc, qbt, gamma, C, gate_name,
                                f"software qbc:{qbc} qbt:{qbt}",
                                remove_had=nbody,
                                remove_1stCZ=(remove_1stCZ if first_layer else ''))
                elif cphase_implementation == "hardware":
                    # TODO: clean up in function just as above

                    #arbitrary phase gate
                    c_arb_pulse = deepcopy(self.operation_dict[gate_name])
                    #get amplitude and dynamic phase from model
                    angle = 4 * gamma * C
                    if doswap:
                        angle+= np.pi # correct phase since a fermionic swap gate is used instead of a swap gate
                    angle = angle % (2*np.pi)
                    ampl, dyn_phase = eval(gate_sequence_info['phase_func'][qbt+qbc])(angle)

                    # overwrite angles for angle % 2 pi  == 0
                    if abs(angle) < zero_angle_threshold:
                        if strategy == "zero_amplitude":
                            ampl, dyn_phase = 0, {qb:0 for qb in dyn_phase.keys()}
                        elif strategy == "skip_gate":
                            two_qb_block = Block(f"qbc:{qbc} qbt:{qbt}",
                                                 [z_qbc, z_qbt])
                            if nbody:
                                simultaneous.extend(Block(f"{qbx} nbody_start", nbody_start).build(ref_pulse=f"start"))
                                simultaneous.extend(two_qb_block.build())
                                simultaneous.extend(Block(f"{qbx} nbody_end", nbody_end).build())
                            else:
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
                    # print(f"{name}:\nphase angle: {angle}\nAmpl: {ampl}\ndyn_phase: {dyn_phase}")
                    c_arb_pulse['amplitude'] = ampl
                    c_arb_pulse['element_name'] = "flux_arb_gate"
                    c_arb_pulse['basis_rotation'].update(dyn_phase)

                    two_qb_block = Block(f"qbc:{qbc} qbt:{qbt}",
                                         [z_qbc, z_qbt, c_arb_pulse])
                    if doswap:
                        two_qb_block.extend(self._U_qb_pair_fermionic_swap(qbc, qbt, gate_name, f"FSWAP").build())

                if nbody:
                    simultaneous.extend(Block(f"{qbx} nbody_start", nbody_start).build(ref_pulse=f"start"))
                    simultaneous.extend(two_qb_block.build())
                    simultaneous.extend(Block(f"{qbx} nbody_end", nbody_end).build())
                else:
                    simultaneous.extend(two_qb_block.build(ref_pulse=f"start"))
            # add block referenced to start of U_k
            U.extend(simultaneous.build())
            if doswap:
                self.qb_names[gates_info['qbs'][0]],self.qb_names[gates_info['qbs'][1]] \
                    = self.qb_names[gates_info['qbs'][1]],self.qb_names[gates_info['qbs'][0]]
            #print(self.qb_names)

        # add single qb z rotation for single qb terms of hamiltonian
        for qb, h in enumerate(single_qb_terms):
            U.extend([self.Z_gate(2 * gamma * h * 180 / np.pi, self.qb_names[qb])])

        return U

    def _U_qb_pair_software_decomposition(self, qbc, qbt, gamma, J, cz_gate_name,
                                          block_name, remove_had=False,
                                          remove_1stCZ='', echo=()):
        """
        Performs the software decomposition of the QAOA two qubit unitary:
        diag({i phi, -i phi, -i phi, i phi}) where phi = J * gamma.

        Efficient decomposition by Christian :
        (X180)--------(X180)-------------------------------- (echo pulses)
        H_qbt---CZ---H_qbt---RZ_qbt(2*phi)---H_qbt---CZ---H_qbt
        where:
            H_qbt is a Hadamard gate on qbt (implemented using Y90 + Z180)
            CZ is the control pi-phase gate between qbc and qbt
            RZ_qb(x) is a z rotation of angle x on qb

        :param qbc:
        :param qbt:
        :param gamma:
        :param J:
        :param cz_gate_name:
        :param remove_had: optional. If true, the outermost Hadamard gates
            are removed (default: false)
        :param remove_1stCZ: optional. If 'late_init', the first CZ gate and
            the first Hadamard are removed. If 'early_init', the first CZ gate
            and both surrounding Hadamard gates are removed. (default '')
        :param echo (list): optional list of logical qubits on which echo pulses
            will be applied. Cannot be used with 'early_init' or 'late_init'.
        :return:
        """
        assert remove_1stCZ == '' or not remove_had, \
            "The combination of remove_1stCZ and remove_had is not supported."
        assert remove_1stCZ in ['', 'early_init', 'late_init'], \
            f"remove_1stCZ=\'{remove_1stCZ}\' is not supported."

        ops = [] if remove_1stCZ != '' else [cz_gate_name]
        if remove_1stCZ != 'early_init':
            ops += ["Z180 {qbt:}", "Y90 {qbt:}"]
        ops += ["Z{two_phi:} {qbt:}", "Z180 {qbt:}",
                "Y90 {qbt:}", cz_gate_name]
        if remove_had:
            # put flux pulses in same element
            pulse_modifs = {0: dict(element_name="flux_arb_gate"),
                            6: dict(element_name="flux_arb_gate")}
        elif remove_1stCZ != '':
            ops = ops + ["Z180 {qbt:}", "Y90 {qbt:}"]
            # put flux pulses in same element
            if remove_1stCZ == 'early_init':
                pulse_modifs = {3: dict(element_name="flux_arb_gate")}
            else:
                pulse_modifs = {5: dict(element_name="flux_arb_gate")}
        else:
            ops = ["Z180 {qbt:}", "Y90 {qbt:}"] + ops + ["Z180 {qbt:}", "Y90 {qbt:}"]
            # put flux pulses in same element
            pulse_modifs = {2: dict(element_name="flux_arb_gate"),
                            8: dict(element_name="flux_arb_gate")}
        fill_values = dict(qbt=qbt, two_phi=2 * gamma * J * 180 / np.pi)
        return self.block_from_ops(block_name, ops, fill_values, pulse_modifs)

    def _U_qb_pair_fermionic_simulation(self, qbc, qbt, phi, cz_gate_name,
                                          block_name):
        """
        Performs the software decomposition of the fermionic simulation gate:
        [[1,0,0,0] , [0,0,1,0] , [0,1,0,0] , [0,0,0,-exp(-i phi)]].
        (decomposition by Christoph)

        :param qbc:
        :param qbt:
        :param phi:
        :param cz_gate_name:
        :return:
        """
        ops = ["Z180 {qbt:}", "Z{angle:} {qbt:}", "Y90 {qbt:}", "Z180 {qbt:}", 
                cz_gate_name, "Z90 {qbc:}", "Z{angle:} {qbc:}", "Y90 {qbc:}",
                "mY90 {qbt:}", "Z{angle:} {qbt:}", "Y90 {qbt:}", "Z90 {qbt:}",
                cz_gate_name, "Z180 {qbc:}", "Z180 {qbt:}", "Y90 {qbc:}", "Y90 {qbt:}", "Z90 {qbc:}",
                cz_gate_name, "Z180 {qbc:}", "Z180 {qbt:}", "Y90 {qbc:}", "Y90 {qbt:}"]

        # fermionic simulation gate:
        # @(angle) kron (H, H) * CZ * kron (RZ (pi / 2), I) * kron (RY (pi / 2), RY (pi / 2)) * kron (Z, Z) * CZ *
        # kron (RY (pi / 2), RY (angle)) * kron (RZ (pi / 2) * RZ (angle), RZ (pi / 2)) * CZ *
        # kron (I, Z) * kron (I, RY (pi / 2)) * kron (I, RZ (pi) * RZ (angle))
        # with angle = pi+phi/2
        # where RY(angle) has to be decomposed into RZ(pi/2)*RY(pi/2)*RZ(angle)*RY(-pi/2)*RZ(-pi/2)

        fill_values = dict(qbc=qbc, qbt=qbt, angle=180 + 1/2 * (phi * 180/np.pi) )

        # put flux pulses in same element, simultaneous Y gates
        pulse_modifs = {4: dict(element_name="flux_arb_gate"),
                        12: dict(element_name="flux_arb_gate"),
                        18: dict(element_name="flux_arb_gate"),
                        8: dict(ref_point="start"),
                        16: dict(ref_point="start"),
                        22: dict(ref_point="start")}
        return self.block_from_ops(block_name, ops, fill_values, pulse_modifs)

    def _U_qb_pair_fermionic_swap(self, qbc, qbt, cz_gate_name, block_name):
        """
        Performs a fermionic swap:
        [[1,0,0,0] , [0,0,1,0] , [0,1,0,0] , [0,0,0,-1]]

        Decomposition:

        (H_qbt, H_qbc)---CZ---(H_qbt, H_qbc)---CZ---(H_qbt, H_qbc)
        where:
            H_qbt/H_qbc is a Hadamard gate on qbt/qbc (implemented using Z180 + Y90)
            CZ is the control pi-phase gate between qbc and qbt

        :param qbc:
        :param qbt:
        :param cz_gate_name:
        :return:
        """
        pulses = []
        opsH = ["Z180 {qbc:}", "Z180 {qbt:}", "Y90 {qbc:}", "Y90 {qbt:}"] # 2 Hadamard gates
        for i in range(3):
            pulses.extend(self.block_from_ops(f"Had{i}", opsH, dict(qbc=qbc, qbt=qbt), {3: dict(ref_point="start")}).build())
            if i < 2:
                pulses.extend(self.block_from_ops(f"CZ{i}", [cz_gate_name],
                    {}, {0: dict(element_name="flux_arb_gate")}).build())
        return Block(block_name, pulses)

    def D(self, name, beta, qubits='all'):
        if qubits == 'all':
            qubits = self.qb_names

        pulses = []
        ops = ["mY90 {qbn:}", "Z{angle:} {qbn:}", "Y90 {qbn:}"]
        for qbn in qubits:
            D_qbn = self.block_from_ops(f"{qbn}", ops,
                                        dict(qbn=qbn, angle=2 * beta * 180 /
                                                            np.pi))
            # reference block to beginning of D_k block
            pulses.extend(D_qbn.build(ref_pulse=f"start"))
        return Block(name, pulses)

    @staticmethod
    def get_corr_and_coupl_info(gates_info):
        """
        Helper function to get correlations and couplings used in the sequence
        Correlations are defined as tuples of zero-indexed of qubits: eg.
        (0,1) indicates a correlation will be made on qb1 and qb2
        a coupling is the C between two qubits
        Args:
            gates_info: list of list of information
            dictionaries. Dictionaries contain information about a two QB gate:
            assumes the following keys:
            - qbs: 2-tuple of logical qubit indices
            - gate_name: name of the 2 qb gate type
            - C: coupling btw the two qubits
            - (phase_func): Dictionary of string representations of functions predicting
                amplitude and dynamic phase for given target conditional phase.
                Only required when using hardware implementation
               of arbitrary phase gate.
        Returns:
            corr_info (list): list of tuples indicating qubits to correlate:
                by logical qubit index.
            couplings (list): corresponding coupling for each correlation

        """
        flattened_info = deepcopy(gates_info['gate_list'])

        corr_info = [i['qbs'] for i in flattened_info]
        couplings = [i['J'] if 'J' in i else i['C'] if 'C' in i else 0 for i in flattened_info]
        return corr_info, couplings