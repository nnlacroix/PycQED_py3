import logging
log = logging.getLogger(__name__)

import itertools
import numpy as np
import qutip as qtp
from collections import OrderedDict
from pycqed.analysis_v3 import fitting as fit_module
from pycqed.analysis_v3 import plotting as plot_module
from pycqed.analysis_v3 import helper_functions as hlp_mod
from pycqed.analysis_v3 import processing_pipeline as pp_mod
from pycqed.measurement.waveform_control import circuit_builder as cb_mod
from copy import deepcopy

import sys
from pycqed.analysis_v3 import pipeline_analysis as pla
pla.search_modules.add(sys.modules[__name__])


def standard_qubit_pulses_to_rotations(pulse_list):
    """
    Converts lists of n-tuples of standard PycQED single-qubit pulse names to
    the corresponding rotation matrices on the n-qubit Hilbert space.
    """
    standard_pulses = {
        'I': qtp.qeye(2),
        'X180': qtp.sigmax(),
        'mX180': qtp.rotation(qtp.sigmax(), -np.pi),
        'Y180': qtp.sigmay(),
        'mY180': qtp.rotation(qtp.sigmay(), -np.pi),
        'X90': qtp.rotation(qtp.sigmax(), np.pi/2),
        'mX90': qtp.rotation(qtp.sigmax(), -np.pi/2),
        'Y90': qtp.rotation(qtp.sigmay(), np.pi/2),
        'mY90': qtp.rotation(qtp.sigmay(), -np.pi/2),
        'Z90': qtp.rotation(qtp.sigmaz(), np.pi/2),
        'mZ90': qtp.rotation(qtp.sigmaz(), -np.pi/2),
        'Z180': qtp.sigmaz(),
        'mZ180': qtp.rotation(qtp.sigmaz(), -np.pi),
        'CZ': qtp.Qobj(np.diag([1, 1, 1, -1]), dims=[[2, 2], [2, 2]])
    }
    rotations = [qtp.tensor(*[standard_pulses[pulse] for pulse in qb_pulses])
                 for qb_pulses in pulse_list]
    for i in range(len(rotations)):
        rotations[i].dims = [[d] for d in rotations[i].shape]
    return rotations


def process_tomography_analysis(data_dict, basis_rots=None, n_qubits=2,
                                gate_of_interest='CZ', verbose=False,
                                show=False, **params):
    if basis_rots is None:
        basis_rots = hlp_mod.get_param(
            'basis_rots', data_dict, default_value=('I', 'X90', 'Y90', 'X180'),
            **params)
    prep_pulses_list = list(itertools.product(basis_rots, repeat=n_qubits))

    # get lambda array
    if verbose:
        print()
        print('Getting lambda array')

    # get measured density matrices
    measured_rhos = len(prep_pulses_list) * ['']
    for i, prep_pulses in enumerate(prep_pulses_list):
        prep_str = ''.join(prep_pulses)
        if verbose:
            print(prep_str)
        measured_rhos[i] = \
            hlp_mod.get_param(f'{prep_str}.rho', data_dict, raise_error=True,
                              error_message=f'Data for preparation pulses '
                                            f'{prep_str} was not found in '
                                            f'data_dict.').full().flatten()
    measured_rhos = np.asarray(measured_rhos)

    # get density matrices for the preparation states
    U1s = {pulse: standard_qubit_pulses_to_rotations([(pulse,)])[0]
           for pulse in ['X90', 'Y90', 'mX90', 'mY90', 'I',
                         'X180', 'Y180', 'mY180', 'mX180']}
    preped_rhos = len(prep_pulses_list) * ['']
    preped_rhos_flatten = len(prep_pulses_list) * ['']
    for i, prep_pulses in enumerate(prep_pulses_list):
        prep_str = ''.join(prep_pulses)
        if verbose:
            print(prep_str, U1s[prep_pulses[0]], U1s[prep_pulses[1]])
        psi_target = (qtp.tensor(U1s[prep_pulses[0]], U1s[prep_pulses[1]]) *
                      qtp.tensor(qtp.basis(2), qtp.basis(2)))
        rho_target = psi_target*psi_target.dag()
        preped_rhos[i] = rho_target
        preped_rhos_flatten[i] = rho_target.full().flatten()

    preped_rhos_flatten = np.asarray(preped_rhos_flatten)
    lambda_array = np.dot(measured_rhos, np.linalg.inv(preped_rhos_flatten))

    # get beta array
    if verbose:
        print()
        print('Geting beta array')
    standard_2qb_pauli_labels = list(itertools.product(
        ['I', 'X180', 'Y180', 'Z180'], repeat=2))
    standard_2qb_pauli_ops = standard_qubit_pulses_to_rotations(
        standard_2qb_pauli_labels)
    if verbose:
        print('len(standard_2qb_pauli_labels) ', len(standard_2qb_pauli_labels))
        print('len(standard_2qb_pauli_ops) ', len(standard_2qb_pauli_ops))

    prepared_rhos_rotated = []
    cnt = 0
    for i, prepared_rho in enumerate(preped_rhos):
        for plhs in standard_2qb_pauli_ops:
            plhs.dims = [[2,2], [2,2]]
            for prhs in standard_2qb_pauli_ops:
                prhs.dims = [[2,2], [2,2]]
                cnt += 1
                prepared_rhos_rotated += [
                    (plhs*prepared_rho*prhs.dag()).full().flatten()]
    prepared_rhos_rotated = np.asarray(prepared_rhos_rotated)
    if verbose:
        print('prepared_rhos_rotated.shape ', prepared_rhos_rotated.shape)
        print('preped_rhos_flatten.shape ', preped_rhos_flatten.shape)
    beta_array = np.dot(prepared_rhos_rotated,
                        np.linalg.inv(preped_rhos_flatten))

    # get chi matrix
    if verbose:
        print()
        print('Getting chi matrix')
    chunck_size = len(standard_2qb_pauli_ops)**2
    beta_array_reshaped = np.zeros(shape=(len(preped_rhos)**2, chunck_size),
                                   dtype='complex128')
    for i in range(len(preped_rhos)):
        beta_array_reshaped[i*len(preped_rhos): (i+1)*len(preped_rhos), :] = \
            beta_array[i*chunck_size:(i+1)*chunck_size, :].T
        if verbose:
            print(beta_array[i*chunck_size:(i+1)*chunck_size, :].T.shape)
    if verbose:
        print('beta_array_reshaped.shape ', beta_array_reshaped.shape)
        print('lambda_array.flatten().size ', lambda_array.flatten().size)

    chi = np.linalg.solve(beta_array_reshaped, lambda_array.flatten())
    chi = chi.reshape((16, 16))
    chi_qtp = qtp.Qobj(chi, dims=[[[2, 2], [2, 2]], [[2, 2], [2, 2]]])
    if show:
        qtp.matrix_histogram_complex(chi_qtp)

    # add found chi matrix to data_dict
    hlp_mod.add_param(f'chi_{gate_of_interest}', chi_qtp.full(), data_dict,
                      **params)

    # add gate error to data_dict
    Ugate = qtp.to_chi(qtp.cphase(np.pi))/16
    hlp_mod.add_param(f'measured_error_{gate_of_interest}',
                      1-np.real(qtp.process_fidelity(chi_qtp, Ugate)),
                      data_dict, **params)