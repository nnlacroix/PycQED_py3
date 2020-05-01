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
from pycqed.measurement.waveform_control.circuit_builder import CircuitBuilder
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

def qaoa_sequence(qubits, betas, gammas, gates_info,
                  init_state='0', cphase_implementation='hardware',
                  single_qb_terms=None,
                  tomography=False, tomo_basis=tomo.DEFAULT_BASIS_ROTS,
                  cal_points=None, prep_params=None, upload=True):

    # create sequence, segment and builder
    qb_names = [qb.name for qb in qubits]
    seq_name = f'QAOA_{cphase_implementation}_cphase_{qb_names}'

    seq = sequence.Sequence(seq_name)

    qubits_orig = [qb for qb in qubits]
    builder = QAOAHelper(qubits, prep_params=prep_params)

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
            seg.extend(builder.initialize(init_state).build())

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
            builder.qubits = [qb for qb in qubits_orig]

    # add calibration points
    if cal_points is not None:
        seq.extend(cal_points.create_segments(builder.operation_dict,
                                              builder.get_prep_params()))

    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)

    return seq, np.arange(seq.n_acq_elements())


class QAOAHelper(CircuitBuilder):

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
                     not specified: treated as any other angle
                - (zero_angle_threshold): threshold for considering an angle to be zero  (in rad)
                    (default: use global value)
            - gate_order: list of lists of indices from the gate_list
                All gates in the same sublist are executed simultaneously.
            - (zero_angle_threshold): global threshold for considering an angle to be zero (in rad)
                default: 1e-10
            Example:
            >>> dict(
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
            tmp_single_qb_terms = [0]*len(self.qubits)
            for qb, J in single_qb_terms.items():
                tmp_single_qb_terms[qb] = J
            single_qb_terms = tmp_single_qb_terms
        else:
            single_qb_terms = [0]*len(self.qubits)
        U = Block(name, [])
        for i, gates_same_timing in enumerate(gate_sequence_info['gate_order']):
            simult_bname = f"simultaneous_{i}"
            simultaneous = Block(simult_bname, [])
            simultaneous_end_pulses = []
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
                    add_start = gates_info.get('add_start', '')
                    if add_start != '':
                        qb = self.qubits[gates_info['qbs'][0]].name
                        if add_start == 'Had':
                            add_start = ["Z180 {qb:}", "Y90 {qb:}"]
                        fill_values = dict(qb=qb)
                        add_start_block = self.block_from_ops(f"add_start {qb}", add_start, fill_values, {})
                        simultaneous.extend(add_start_block.build(ref_pulse=f"start"))
                        simultaneous_end_pulses.append(
                            simultaneous.pulses[-1]['name'])
                    continue
                gates_info['gate_name'] = \
                    gates_info['gate_name'] if 'gate_name' in gates_info else 'upCZ'
                remove_1stCZ = gates_info.get('remove_1stCZ', '')
                strategy = gates_info.get("zero_angle_strategy", None)
                nbody = (len(gates_info['qbs'])>2)
                assert not (nbody and doswap), \
                    f"Combination of n-body interaction and swap is not implemented!"
                remove_had = gates_info.get('remove_had', False) or nbody
                zero_angle_threshold = gates_info.get("zero_angle_threshold",
                                                      global_zero_angle_threshold)
                if abs((2 * gamma * C) % (2*np.pi))<zero_angle_threshold \
                        and strategy == "skip_gate" and not doswap:
                    continue
                for qbx in [self.qubits[qb_ind].name for qb_ind in gates_info['qbs']]:
                    for qby_tmp in [self.qubits[qb_ind].name for qb_ind in gates_info['qbs']]:
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
                    f"The logical qubits {gates_info['qbs']} ({[self.qubits[qbi].name for qbi in gates_info['qbs']]}) are currently " \
                        f"not connected by a {gates_info['gate_name']} gate! ({[qb.name for qb in self.qubits]})"
                if nbody:
                    opsH = ["Z180 {qbx:}", "Y90 {qbx:}"] # Hadamard gate
                    nbody_start = self.block_from_ops(f"Had", opsH, dict(qbx=qbx), {}).build()
                    nbody_end = []
                    if cphase_implementation != "software":
                        nbody_end = self.block_from_ops(f"Had", opsH, dict(qbx=qbx), {}).build()
                    for qbz in [self.qubits[qb_ind].name for qb_ind in gates_info['qbs']]:
                        if qbz==qbx or qbz==qby:
                            continue
                        qbz_gate_name = f"{gates_info['gate_name']} {qbx} {qbz}"
                        if gate_name not in self.operation_dict:
                            qbz_gate_name = f"{gates_info['gate_name']} {qbz} {qbx}"
                        nbody_cz = self.block_from_ops(f"CZ {qbz}", [qbz_gate_name],
                            {}, {0: dict(element_name="flux_arb_gate")}).build()
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
                                (self.qubits[gates_info['qbs'][-1]].name if not
                                    nbody else qbt),
                                gamma, C, gate_name,
                                f"software qbc:{qbc} qbt:{qbt}",
                                remove_had=remove_had,
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
                    c_arb_pulse['cphase'] = angle

                    # overwrite angles for angle % 2 pi  == 0
                    if abs(angle) < zero_angle_threshold:
                        # FIXME: check whether all the zero_angle_code still makes sense
                        if strategy == "zero_amplitude":
                            c_arb_pulse['amplitude'], c_arb_pulse['basis_rotation'] = 0, {}
                            c_arb_pulse['cphase'] = None
                            # FIXME: check whether setting to {} is correct
                        elif strategy == "skip_gate":
                            two_qb_block = Block(f"qbc:{qbc} qbt:{qbt}",
                                                 [z_qbc, z_qbt])
                            if nbody:
                                simultaneous.extend(Block(f"{qbx} nbody_start", nbody_start).build(ref_pulse=f"start"))
                                simultaneous.extend(two_qb_block.build())
                                simultaneous.extend(Block(f"{qbx} nbody_end", nbody_end).build())
                            else:
                                simultaneous.extend(two_qb_block.build(ref_pulse=f"start"))
                            simultaneous_end_pulses.append(simultaneous.pulses[-1]['name'])
                            continue
                        elif isinstance(strategy, dict):
                            c_arb_pulse['amplitude'] = strategy.get("amplitude", 0)
                            c_arb_pulse['basis_rotation'] = strategy.get("dynamic_phase", {})
                            # FIXME: check whether setting to {} is correct
                            c_arb_pulse['cphase'] = None
                        elif strategy is None:
                            pass
                        else:
                            raise ValueError(f"Zero angle strategy {strategy} not "
                                             f"understood")
                    # print(f"{name}:\nphase angle: {angle}\nAmpl: {ampl}\ndyn_phase: {dyn_phase}")
                    c_arb_pulse['element_name'] = "flux_arb_gate"
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
                if doswap:
                    # print(f"swapping {(self.qubits[gates_info['qbs'][0]].name, self.qubits[gates_info['qbs'][1]].name)}")
                    self.swap_qubit_indices(gates_info['qbs'])
                simultaneous_end_pulses.append(simultaneous.pulses[-1]['name'])

            if isinstance(simultaneous_end_pulses, list) and len(simultaneous_end_pulses) > 1:
                simultaneous.extend([{"name": f"simultaneous_end_pulse",
                                      "pulse_type": "VirtualPulse",
                                      "pulse_delay": 0,
                                      "ref_pulse": simultaneous_end_pulses,
                                      "ref_point": 'end',
                                      "ref_function": 'max'
                                     }])

            # add block referenced to start of U_k
            U.extend(simultaneous.build())
            #print([qb.name for qb in self.qubits])

        # add single qb z rotation for single qb terms of hamiltonian
        for qb, h in enumerate(single_qb_terms):
            if abs(h) > global_zero_angle_threshold: # TODO: we should use the local value?
                U.extend([self.Z_gate(2 * gamma * h * 180 / np.pi, self.qubits[qb].name)])

        return U

    def _U_qb_pair_software_decomposition(self, qbt, gamma, J, cz_gate_name,
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
            CZ is the control pi-phase gate between the qubits
            RZ_qb(x) is a z rotation of angle x on qb

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
        assert remove_1stCZ in ['', 'early_init', 'late_init'], \
            f"remove_1stCZ=\'{remove_1stCZ}\' is not supported."

        ops = [] if remove_1stCZ != '' else [cz_gate_name]
        if remove_1stCZ != 'early_init':
            ops += ["Z180 {qbt:}", "Y90 {qbt:}"]
        ops += ["Z{two_phi:} {qbt:}", "Z180 {qbt:}",
                "Y90 {qbt:}", cz_gate_name]
        if remove_had and remove_1stCZ == '':
            # put flux pulses in same element
            pulse_modifs = {0: dict(element_name="flux_arb_gate"),
                            6: dict(element_name="flux_arb_gate")}
        elif remove_1stCZ != '':
            if not remove_had:
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
            qubits = [qb.name for qb in self.qubits]

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
        Returns:
            corr_info (list): list of tuples indicating qubits to correlate:
                by logical qubit index.
            couplings (list): corresponding coupling for each correlation

        """
        flattened_info = deepcopy(gates_info['gate_list'])

        corr_info = [i['qbs'] for i in flattened_info]
        couplings = [i['J_for_analysis'] if 'J_for_analysis' in i else i['J'] if 'J' in i else i['C'] if 'C' in i else 0 for i in flattened_info]
        return corr_info, couplings