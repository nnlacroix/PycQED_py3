from pycqed.measurement.randomized_benchmarking.two_qubit_clifford_group \
    import SingleQubitClifford, TwoQubitClifford
from os.path import join, dirname, abspath
from os import mkdir, listdir
import numpy as np
from zlib import crc32

hash_dir = join(abspath(dirname(__file__)), 'clifford_hash_tables')
try:
    mkdir(hash_dir)
except FileExistsError:
    pass


class CliffordLookuptables():
    def __init__(self):
        self.create_lut_files()

    def __call__(self, pauli_transfer_matrix):
        unique_hash = crc32(pauli_transfer_matrix.astype(int))
        if np.array_equal(np.shape(pauli_transfer_matrix), (4, 4)):
            hash_table = self.get_single_qubit_clifford_hash_table()
        elif np.array_equal(np.shape(pauli_transfer_matrix), (16, 16)):
            hash_table = self.get_two_qubit_clifford_hash_table()
        else:
            raise NotImplementedError()
        idx = hash_table.index(unique_hash)

        return idx

    def create_lut_files(self):
        # create single_qubit_hash_lut.txt if it is not in the hash_dir
        if 'single_qubit_hash_lut.txt' not in listdir(hash_dir):
            self.generate_single_qubit_hash_table()
        # create two_qubit_hash_lut.txt if it is not in the hash_dir
        if 'two_qubit_hash_lut.txt' not in listdir(hash_dir):
            self.generate_two_qubit_hash_table()

    def get_single_qubit_clifford_hash_table(self):
        """
        Gets the sg qubit clifford hash table. Requires this to be generated
        first. To generate, execute "generate_single_qubit_hash_table.py".
        """
        if not hasattr(self, 'single_qubit_hash_table'):
            with open(join(hash_dir, 'single_qubit_hash_lut.txt'),
                      'r') as f:
                self.single_qubit_hash_table = [int(line.rstrip('\n')) for
                                                line in f]
                print('Opened single_qubit_hash_lut.txt.')
        return self.single_qubit_hash_table

    def get_two_qubit_clifford_hash_table(self):
        """
        Gets the two qubit clifford hash table. Requires this to be generated
        first. To generate, execute "generate_two_qubit_hash_table.py".
        """
        if not hasattr(self, 'two_qubit_hash_table'):
            with open(join(hash_dir, 'two_qubit_hash_lut.txt'),
                      'r') as f:
                self.two_qubit_hash_table = [int(line.rstrip('\n')) for
                                             line in f]
            print('Opened two_qubit_hash_lut.txt.')
        return self.two_qubit_hash_table

    def construct_clifford_lookuptable(self, generator, indices):
        lookuptable = []
        for idx in indices:
            clifford = generator(idx=idx)
            # important to use crc32 hashing as this is a non-random hash
            hash_val = crc32(
                clifford.pauli_transfer_matrix.round().astype(int))
            lookuptable.append(hash_val)
        return lookuptable

    def generate_single_qubit_hash_table(self):
        print("Generating Single-Qubit Clifford hash tables.")
        single_qubit_hash_lut = self.construct_clifford_lookuptable(
            SingleQubitClifford, np.arange(24))
        with open(join(hash_dir, 'single_qubit_hash_lut.txt'), 'w') as f:
            for h in single_qubit_hash_lut:
                f.write(str(h)+'\n')
        print("Successfully generated single qubit Clifford hash table.")

    def generate_two_qubit_hash_table(self):
        print("Generating Two-Qubit Clifford hash table.")
        two_qubit_hash_lut = self.construct_clifford_lookuptable(
            TwoQubitClifford, np.arange(11520))
        with open(join(hash_dir, 'two_qubit_hash_lut.txt'), 'w') as f:
            for h in two_qubit_hash_lut:
                f.write(str(h)+'\n')
        print("Successfully generated two-qubit Clifford hash tables.")
