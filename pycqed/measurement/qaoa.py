from pprint import pprint

import numpy as np
from copy import deepcopy
import pycqed.measurement.waveform_control.sequence as sequence
from pycqed.measurement.pulse_sequences.single_qubit_tek_seq_elts import \
    get_pulse_dict_from_pars, add_preparation_pulses, pulse_list_list_seq, prepend_pulses
import pycqed.measurement.waveform_control.segment as segment
from pycqed.measurement.waveform_control.block import Block
from pycqed.measurement.waveform_control import pulsar as ps



def qaoa_sequence(qb_names, betas, gammas, two_qb_gates_info, operation_dict,
                  init_state='0', cphase_implementation='hardware',
                  prep_params=None, upload=True):

    # create sequence, segment and builder
    seq_name = f'QAOA_{cphase_implementation}_cphase_{qb_names}'
    seg_name = f'QAOA_{cphase_implementation}_cphase_{qb_names}'
    seq = sequence.Sequence(seq_name)
    seg = segment.Segment(seg_name)
    builder = QAOAHelper(qb_names, operation_dict)

    prep_params = {} if prep_params is None else prep_params

    # initialize qubits
    seg.extend(builder.initialize(init_state, prep_params=prep_params).build())

    # QAOA Unitaries
    for k, (gamma, beta) in enumerate(zip(gammas, betas)):
        # Uk
        seg.extend(builder.U(f"U_{k}", two_qb_gates_info,
                   gamma, cphase_implementation).build())
        # Dk
        seg.extend(builder.D(f"D_{k}", beta).build())

    # readout qubits
    seg.extend(builder.mux_readout().build())

    seq.add(seg)

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

    def initialize(self, init_state='0', qubits='all', prep_params=None,
                   simultaneous=True):
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

        for qbn, init in zip(qubits, init_state):
            # add qb name and "s" for reference to start
            op = self.STD_INIT.get(init, init) + \
                 f"{'s' if len(pulses) != 0 and simultaneous else ''} " + qbn
            pulses.append(deepcopy(self.operation_dict[op]))

        # TODO: note, add prep pulses could be in this class also.
        pulses_with_prep = add_preparation_pulses(pulses, self.operation_dict,
                                                  qubits, **prep_params)
        return Block(block_name, pulses_with_prep)

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

    def block_from_operations(self, block_name, operations, fill_values=None):
        """
        Returns a block with the given operations.
        Eg.
        >>> ops = ['X180 {qbt:}', 'X90 {qbc:}']
        >>> builder.block_from_operations("MyAwesomeBlock",
        >>>                                ops,
        >>>                                {'qbt': qb1, 'qbc': qb2})
        :param block_name: Name of the block
        :param operations: list of operations (str), which can be preformatted
            and later filled with values in the dictionary fill_values
        :param fill_values (dict): optional fill values for operations.
        :return:
        """
        return Block(block_name,
                     [deepcopy(self.operation_dict[op.format(**fill_values)])
                      for op in operations])

class QAOAHelper(HelperBase):

    def U(self, name, gate_sequence_info, gamma, cphase_implementation):
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
            - (arb_phase_func): only required when using hardware implementation
               of arbitrary phase gate.
            All dictionaries within the same sub list are executed simultaneously
            Example:
            >>> [
            >>>     # first set of 2qb gates to run together
            >>>     [dict(qbc='qb1', qbt='qb2', gate_name='upCZ qb2 qb1', C=1,
            >>>           arb_phase_func=func_qb1_qb2),
            >>>      dict(qbc='qb4', qbt='qb3', gate_name='upCZ qb4 qb3', C=1,
            >>>           arb_phase_func=func_qb4_qb3)],
            >>>     # second set of 2qb gates
            >>>     [dict(qbc='qb3', qbt='qb2', gate_name='upCZ qb2 qb3', C=1,
            >>>        arb_phase_func=func_qb3_qb2)]
            >>> ]
        :param gamma: rotation angle (in rad)
        :param cphase_implementation: implementation of arbitrary phase gate.
            "software" --> gate is decomposed into single qb gates and 2x CZ gate
            "hardware" --> hardware arbitrary phase gate
        :return: Unitary U (Block)
        """

        assert cphase_implementation in ("software", "hardware")

        if cphase_implementation == "software":
            raise NotImplementedError()

        U = Block(name, [])
        for i, two_qb_gates_same_timing in enumerate(gate_sequence_info):
            simultaneous = Block(f"{name}_simultanenous_{i}", [])
            for two_qb_gates_info in two_qb_gates_same_timing:
                #gate info
                qbc = two_qb_gates_info["qbc"]
                qbt = two_qb_gates_info["qbt"]
                gate_name = two_qb_gates_info['gate_name']
                C = two_qb_gates_info["C"]

                #virtual gate on qb 0
                z_qbc = self.Z_gate(gamma * C * 180 / np.pi, qbc)

                # virtual gate on qb 1
                z_qbt = self.Z_gate(gamma * C * 180 / np.pi, qbt)

                #arbitrary phase gate
                c_arb_pulse = self.operation_dict[gate_name]
                #get amplitude and dynamic phase from model
                ampl, dyn_phase = two_qb_gates_info['arb_phase_func'](2 * gamma * C)
                c_arb_pulse['amplitude'] = ampl
                c_arb_pulse['element_name'] = "flux_arb_gate"
                c_arb_pulse['basis_rotation'].update(
                    {two_qb_gates_info['qbc']: dyn_phase})

                two_qb_block = Block(f"qbc:{qbc} qbt:{qbt}",
                                     [z_qbc, z_qbt, c_arb_pulse])
                ref_point = "start" if len(simultaneous) > 0 else "end"
                simultaneous.extend(two_qb_block.build(ref_point=ref_point))
            U.extend(simultaneous.build())

        return U

    def D(self, name, beta, qubits='all'):
        if qubits == 'all':
            qubits = self.qb_names

        pulses = []
        ops = ["mY90", "Zbeta", "Y90"]

        for qbn in qubits:
            mY90 = deepcopy(self.operation_dict["mY90 " + qbn])
            Zbeta = self.Z_gate(beta*180/np.pi, qbn)
            Y90 = deepcopy(self.operation_dict["Y90 " + qbn])
            block = Block(f"{qbn}", [mY90, Zbeta, Y90])

            for op in ops:
                # get pulse from operation dict or zbeta pulse
                p = deepcopy(
                    self.operation_dict.get(f"{op} {qbn}" , ))

                pulses.append(p)
        print(pulses)
        return Block(name, pulses)