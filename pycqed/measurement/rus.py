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


def q (theta):
    return np.arctan(np.tan(theta)**2)


def rus_sequence (qbs, thetas, operation_dict,
                  init_state="000", theta_threshold = 0.05,
                  tomography=False, tomography_options={},
                  cal_points=None, prep_params=None, upload=True):

    # Get Qubits & Ancilla
    qb_in_name = qbs[0].name
    qb_out_name = qbs[1].name
    ancilla_name = qbs[2].name
    qb_names = [qb_in_name, qb_out_name, ancilla_name]

    if init_state[2] == '1':
        raise ValueError(f'Ancilla has to be in |0> state')

    # Create Sequence
    seq_name = f'QuantumNeuron_RUS_{qb_in_name}_{qb_out_name}_{ancilla_name}'
    seq = sequence.Sequence(seq_name)

    prep_params = {} if prep_params is None else prep_params

    builder = RUSHelper(qb_names, deepcopy(operation_dict))

    # Tomography Pulses
    tomo_basis = tomography_options.get("basis_rots", tomo.DEFAULT_BASIS_ROTS)
    tomo_qbs = tomography_options.get("tomo_qbs", [qb_out_name])
    tomography_segments = (None,)
    if tomography:
        tomography_segments = \
            get_tomography_pulses(*tomo_qbs, basis_pulses=tomo_basis)

    # Define Sequence Elements
    c_ry_theta_p = [deepcopy(operation_dict['X90 ' + ancilla_name]),
                    deepcopy(operation_dict['upCZ ' + ancilla_name + ' ' + qb_in_name]),
                    deepcopy(operation_dict['mX90 ' + ancilla_name])]
    c_ry_theta_p[0]['name'] = 'ancilla_X90_1'
    c_ry_theta_p[0]['ref_pulse'] = 'Initialization_all-|-start'
    c_ry_theta_p[1]['name'] = 'ancilla_C_Rz_theta_p_1'
    c_ry_theta_p[1]['pulse_type'] = 'BufferedCZPulse'
    c_ry_theta_p[2]['name'] = 'ancilla_mX90_1'

    cy = [deepcopy(operation_dict['X90 ' + qb_out_name]),
#          deepcopy(operation_dict['mZ90 ' + ancilla_name]), #TODO: WHY?
          deepcopy(operation_dict['upCZ ' + qb_out_name + ' ' + ancilla_name]),
          deepcopy(operation_dict['mX90 ' + qb_out_name])]
    cy[0]['name'] = 'output_X90_1'
    cy[0]['ref_pulse'] = 'ancilla_C_Rz_theta_p_1'
#    cy[2]['name'] = 'ancilla_mZ90_1'
    cy[1]['name'] = 'output_CZ_1'
    cy[2]['name'] = 'output_mX90_1'

    c_ry_theta_m = [deepcopy(operation_dict['X90 ' + ancilla_name]),
                    deepcopy(operation_dict['upCZ ' + ancilla_name + ' ' + qb_in_name]),
                    deepcopy(operation_dict['mX90 ' + ancilla_name])]
    c_ry_theta_m[0]['name'] = 'ancilla_X90_2'
    c_ry_theta_m[0]['ref_pulse'] = 'output_CZ_1'
    c_ry_theta_m[1]['name'] = 'ancilla_C_Rz_theta_m_1'
    c_ry_theta_m[1]['pulse_type'] = 'BufferedCZPulse'
    c_ry_theta_m[2]['name'] = 'ancilla_mX90_2'

    if np.ndim(thetas) < 1:
        thetas = [thetas]

    for ind_array, theta in enumerate(thetas):
        for i, ts in enumerate(tomography_segments):
            seg_name = f'segment_{i}_{ind_array}' if ts is None else  \
                f'segment_{i}_{ind_array}_tomo_{i}'
            seg = segment.Segment(seg_name)

            # calculate & update flux pulse for C-ARB
            theta_p = 2*np.pi - (2*theta % (2 * np.pi))
            theta_m = +(2*theta % (2 * np.pi))

            if theta < theta_threshold:
                c_ry_theta_p[1]['amplitude'] = 0
                c_ry_theta_p[1]['force_adapt_pulse_length'] = None
                c_ry_theta_m[1]['amplitude'] = 0
                c_ry_theta_m[1]['force_adapt_pulse_length'] = None
            else:
                c_ry_theta_p[1]['cphase'] = theta_p
                c_ry_theta_p[1]['force_adapt_pulse_length'] = 'absolute'
                c_ry_theta_m[1]['cphase'] = theta_m
                c_ry_theta_m[1]['force_adapt_pulse_length'] = 'absolute'


            # initialize qubits
            seg.extend(builder.initialize(init_state, prep_params=prep_params).build())

            # compile RUS Sequence
            seg.extend(c_ry_theta_p)
            seg.extend(cy)
            seg.extend(c_ry_theta_m)

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


def rus_sequence_new (qbs, thetas, operation_dict,
                  init_state="000", theta_threshold = 0.05,
                  tomography=False, tomography_options={},
                  cal_points=None, prep_params=None, upload=True):

    # Get Qubits & Ancilla
    qb_in_name = qbs[0].name
    qb_out_name = qbs[1].name
    ancilla_name = qbs[2].name
    qb_names = [qb_in_name, qb_out_name, ancilla_name]

    if init_state[2] == '1':
        raise ValueError(f'Ancilla has to be in |0> state')

    # Create Sequence
    seq_name = f'QuantumNeuron_RUS_{qb_in_name}_{qb_out_name}_{ancilla_name}'
    seq = sequence.Sequence(seq_name)

    prep_params = {} if prep_params is None else prep_params

    builder = RUSHelper(qb_names, deepcopy(operation_dict))

    # Tomography Pulses
    tomo_basis = tomography_options.get("basis_rots", tomo.DEFAULT_BASIS_ROTS)
    tomo_qbs = tomography_options.get("tomo_qbs", [qb_out_name])
    tomography_segments = (None,)
    if tomography:
        tomography_segments = \
            get_tomography_pulses(*tomo_qbs, basis_pulses=tomo_basis)

    # Define Sequence Elements
    c_ry_theta_p = [deepcopy(operation_dict['X90 ' + ancilla_name]),
                    deepcopy(operation_dict['upCZ ' + ancilla_name + ' ' + qb_in_name]),
                    deepcopy(operation_dict['mX90 ' + ancilla_name])]
    c_ry_theta_p[0]['name'] = 'ancilla_X90_1'
    c_ry_theta_p[0]['ref_pulse'] = 'Initialization_all-|-start'
    c_ry_theta_p[1]['name'] = 'ancilla_C_Rz_theta_p_1'
    c_ry_theta_p[1]['pulse_type'] = 'BufferedCZPulse'
    c_ry_theta_p[2]['name'] = 'ancilla_mX90_1'

    cy = [deepcopy(operation_dict['X90 ' + qb_out_name]),
#          deepcopy(operation_dict['mZ90 ' + ancilla_name]), #TODO: WHY?
          deepcopy(operation_dict['upCZ ' + qb_out_name + ' ' + ancilla_name]),
          deepcopy(operation_dict['mX90 ' + qb_out_name])]
    cy[0]['name'] = 'output_X90_1'
    cy[0]['ref_pulse'] = 'ancilla_C_Rz_theta_p_1'
#    cy[2]['name'] = 'ancilla_mZ90_1'
    cy[1]['name'] = 'output_CZ_1'
    cy[2]['name'] = 'output_mX90_1'

    c_ry_theta_m = [deepcopy(operation_dict['X90 ' + ancilla_name]),
                    deepcopy(operation_dict['upCZ ' + ancilla_name + ' ' + qb_in_name]),
                    deepcopy(operation_dict['mX90 ' + ancilla_name])]
    c_ry_theta_m[0]['name'] = 'ancilla_X90_2'
    c_ry_theta_m[0]['ref_pulse'] = 'output_CZ_1'
    c_ry_theta_m[1]['name'] = 'ancilla_C_Rz_theta_m_1'
    c_ry_theta_m[1]['pulse_type'] = 'BufferedCZPulse'
    c_ry_theta_m[2]['name'] = 'ancilla_mX90_2'

    if np.ndim(thetas) != 2 or thetas.shape[0] != 2:
        raise ValueError("Now theta must be a (2,num_thetas)-dimensional array")

    for ind_array, theta in enumerate(thetas.T): #taking slices of thetas for each run
        for i, ts in enumerate(tomography_segments):
            seg_name = f'segment_{i}_{ind_array}' if ts is None else  \
                f'segment_{i}_{ind_array}_tomo_{i}'
            seg = segment.Segment(seg_name)

            # calculate & update flux pulse for C-ARB
            theta_0_p = 2*np.pi - (2*theta[0] % (2 * np.pi))
            theta_0_m = +(2*theta[0] % (2 * np.pi))
            theta_1_p = 2*np.pi - (2*(theta[1]-theta[0]) % (2 * np.pi))
            theta_1_m = +(2*(theta[1]-theta[0]) % (2 * np.pi))

            if theta_1_p < theta_threshold:
                c_ry_theta_p[1]['amplitude'] = 0
                c_ry_theta_p[1]['force_adapt_pulse_length'] = None
                c_ry_theta_m[1]['amplitude'] = 0
                c_ry_theta_m[1]['force_adapt_pulse_length'] = None
            else:
                c_ry_theta_p[1]['cphase'] = theta_1_p
                c_ry_theta_p[1]['force_adapt_pulse_length'] = 'absolute'
                c_ry_theta_m[1]['cphase'] = theta_1_m
                c_ry_theta_m[1]['force_adapt_pulse_length'] = 'absolute'


            # initialize qubits
            seg.extend(builder.initialize(init_state, prep_params=prep_params).build())

            # compile RUS Sequence
            seg.extend(c_ry_theta_p)
            seg.extend(cy)
            seg.extend(c_ry_theta_m)

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

class RUSHelper(HelperBase):

    def RUS(self, name, gate_sequence_info, theta):
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
        :param theta: Rz rotation angle (in rad)
        :return: RUS (Block)
        """


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