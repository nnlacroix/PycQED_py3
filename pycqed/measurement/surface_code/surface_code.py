
import pycqed.measurement.quantum_experiment as qe


class SurfaceCodeExperiment(qe.QuantumExperiment):
    def __init__(self, data_qubits, ancilla_qubits, parity_maps,
                 readout_round_lengths, nr_cycles, initializations,
                 finalizations, **kw):
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

    def _parallel_cz_step_block(self, gates, dd_qubits=None):
        """
        Args:
            gates: a list of qubit pairs for the parallel two-qubit gates
            dd_qubits: a list of qubits to put the 

        Returns:
            A block
        """