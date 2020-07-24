
import pycqed.measurement.quantum_experiment as qe


class SurfaceCodeExperiment(qe.QuantumExperiment):
    def __init__(self, data_qubits, ancilla_qubits, parity_maps,
                 readout_round_lengths, nr_cycles, initializations,
                 finalizations, ancilla_dd_pulse='Y180', **kw):
        # provide default values
        for k, v in [
            ('qubits', data_qubits + ancilla_qubits),
            ('experiment_name',
             f'S{len(data_qubits + ancilla_qubits)}_experiment'),
        ]:
            kw.update({k: kw.get(k, v)})
        super().__init__(**kw)
        self.data_qubits = data_qubits
        self.ancilla_qubits = ancilla_qubits
        self.parity_maps = parity_maps
        self.readout_round_lengths = readout_round_lengths
        self.nr_cycles = nr_cycles
        self.initializations = initializations
        self.finalizations = finalizations
        self.ancilla_dd_pulse = ancilla_dd_pulse

    def main_block(self):
        """Block tree of the experiment:
        * Full experiment block excluding init. and final (tomo) readout
            o `nr_cycles` copies of the cycle block
                - Gates block for each readout round. The length is explicitly
                  set to the value from readout_round_lengths.
                    + One block for each simultaneous CZ gate step
                    + Interleaved with single-qubit gates for basis rotations
                      and dynamical decouplings
                - The ancilla readout blocks, one for each readout round.
        """

    def _parallel_cz_step_block(self, gates, dd_qubits=None, dd_pulse='Y180'):
        """
        Creates a block containing parallel CZ gates and dynamical decoupling
        pulses. For now all CZ gates are aligned at the start and the dd pulses
        are aligned to the center of the last CZ gate.

        Args:
            gates: a list of qubit pairs for the parallel two-qubit gates
            dd_qubits: a list of qubits to apply the dynamical decoupling pulses
                to.
        Returns:
            A block containing the relevant pulses.
        """
        block_name = 'CZ_' + '_'.join([''.join(gate) for gate in gates])
        ops = ['CZ {} {}'.format(*gate) for gate in gates]
        ops += [dd_pulse + ' ' + dd_qb for dd_qb in dd_qubits]
        pulse_modifs = {}
        pulse_modifs.update({
            idx: {'ref_point': 'start', 'ref_point_new': 'start'}
            for idx in range(1, len(gates))})
        pulse_modifs.update({
            idx: {'ref_point': 'center', 'ref_point_new': 'center'}
            for idx in range(len(gates), len(ops))})
        return self.block_from_ops(block_name, ops, pulse_modifs=pulse_modifs)

    def _readout_round_gates_block(self, readout_round):
        round_parity_maps = [
            pm for pm in self.parity_maps if pm['round'] == readout_round]
        ancilla_steps = {pm['ancilla']: pm['data'] for pm in round_parity_maps}
        total_steps = max([len(v) for v in ancilla_steps.values()])
        ancilla_steps = {a: (total_steps - len(v)) // 2 * [None] +
                            list(v) + (total_steps - len(v) + 1) // 2 * [None]
                         for a, v in ancilla_steps.items()}
        gate_lists = [
            [(a, ds[s]) for a, ds in ancilla_steps.items() if ds[s] is not None]
            for s in range(total_steps)]
        cz_step_blocks = [self._parallel_cz_step_block(gates)
                          for gates in gate_lists]
        if self.ancilla_dd_pulse is not None:
            ancilla_dd_block = self.block_from_ops(
                'ancilla_dd_block',
                [self.ancilla_dd_pulse + ' ' + a for a in ancilla_steps],
                pulse_modifs={'all': {'ref_pulse': 'start'}})
            cz_step_blocks = cz_step_blocks[:len(total_steps)//2] + \
                [ancilla_dd_block] + cz_step_blocks[len(total_steps)//2:]

        # figure out
