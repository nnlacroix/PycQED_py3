"""
This module defines functions that calculate the energy levels for the full
transmon model, including a coupled resonator according to Koch et al.,
Phys. Rev. A, 76, 042319 (2007) eqs. (2.1) and (3.1).

Also defines the inverse functions, calculating the Hamiltonian parameters
from experimental quantities.
"""

import numpy as np
import scipy as sp
import scipy.optimize
import functools
from typing import Optional, List, Tuple


@functools.lru_cache()
def transmon_charge(ng: float = 0., dim_charge: int = 31):
    """Calculate the transmon charge operator.

    Args:
        ng: Charge offset of the Hamiltonian.
        dim_charge: Number of charge states to use in calculations.

    Returns:
        A (dim_charge x dim_charge) numpy matrix, representing the charge
        operator.
    """
    return np.diag(np.arange(dim_charge) - np.floor(dim_charge / 2) + ng)


@functools.lru_cache()
def transmon_hamiltonian(ec: float, ej: float, ng: float = 0.,
                         dim_charge: int = 31):
    """Calculate the transmon Hamiltonian.

    Args:
        ec: Charging energy of the Hamiltonian.
        ej: Josephson energy of the Hamiltonian.
        ng: Charge offset of the Hamiltonian.
        dim_charge: Number of charge states to use in calculations.

    Returns:
        A (dim_charge x dim_charge) numpy matrix, representing the Hamiltonian.
    """
    ham = 4 * ec * transmon_charge(ng, dim_charge)**2
    ham -= 0.5 * ej * (np.diag(np.ones(dim_charge - 1), 1) +
                       np.diag(np.ones(dim_charge - 1), -1))
    return ham


def transmon_levels(ec: float, ej: float, ng: float = 0., dim_charge: int = 31):
    """Calculate the eigenfrequencies of the transmon Hamiltonian.

    Args:
        ec: Charging energy of the Hamiltonian.
        ej: Josephson energy of the Hamiltonian.
        ng: Charge offset of the Hamiltonian.
        dim_charge: Number of charge states to use in calculations.

    Returns:
        A list of eigenvalues of the transmon Hamiltonian with the ground-state
        energy subtracted and removed.
    """
    ham = transmon_hamiltonian(ec, ej, ng, dim_charge)
    evals = np.sort(np.linalg.eigvalsh(ham))
    return evals[1:] - evals[0]


def transmon_ec_ej(fge: float, anh: float, ng: float = 0.,
                   dim_charge: int = 31):
    """Calculate the Hamiltonian parameters of a transmon.

    Inverts the function `transmon_levels`.

    Args:
        fge: The first transition frequency of the transmon.
        anh: Anharmonicity of the transmon.
        ng: Charge offset of the Hamiltonian.
        dim_charge: Number of charge states to use in calculations.

    Returns:
        The charging energy and the Josephson energy of the transmon.
    """

    def func(ec_ej_, fge_anh_, ng_, dim_charge_):
        fs = transmon_levels(*ec_ej_, ng_, dim_charge_)
        return - fge_anh_ + [fs[0], fs[1] - 2 * fs[0]]

    ec0 = -anh
    ej0 = -(fge - anh)**2 / 8 / anh
    ec_ej = sp.optimize.fsolve(func, np.array([ec0, ej0]),
                               args=(np.array([fge, anh]), ng, dim_charge))
    return ec_ej[0], ec_ej[1]


def transmon_ej_anh(fge: float, ec: float, ng: float = 0.,
                    dim_charge: int = 31):
    """Calculate the Josephson energy and the anharmonicity of a transmon.

    Inverts the function `transmon_levels`. Useful for finding the Josephson
    energy at a new flux bias point.

    Args:
        fge: The first transition frequency of the transmon.
        ec: Charging energy of the Hamiltonian.
        ng: Charge offset of the Hamiltonian.
        dim_charge: Number of charge states to use in calculations.

    Returns:
        The Josephson energy and the anharmonicity of the transmon.
    """

    def func(ej_anh_, fge_ec, ng_, dim_charge_):
        fs = transmon_levels(fge_ec[1], ej_anh_[0], ng_, dim_charge_)
        return [fs[0] - fge_ec[0], fs[1] - 2 * fs[0] - ej_anh_[1]]

    ej0 = (fge + ec)**2 / 8 / ec
    anh0 = -ec
    ej_anh = sp.optimize.fsolve(func, np.array([ej0, anh0]),
                                args=([fge, ec], ng, dim_charge))
    return ej_anh[0], ej_anh[1]


def charge_dispersion_ge_ef(fge: Optional[float] = None,
                            anh: Optional[float] = None,
                            ej: Optional[float] = None,
                            ec: Optional[float] = None,
                            ng: float = 0., dim_charge: int = 31):
    """Calculate charge dispersion for first two transitions of the transmon.

    From the set of parameters fge, anh, ej and ec, one of the following pairs
    should be passed in: (fge, anh), (fge, ec) or (ej, ec).

    Args:
        fge: The first transition frequency of the transmon.
        anh: Anharmonicity of the transmon.
        ej: Josephson energy of the Hamiltonian.
        ec: Charging energy of the Hamiltonian.
        ng: Charge offset of the Hamiltonian.
        dim_charge: Number of charge states to use in calculations.

    Returns:
        Charge dispersion of the first and second transition of the transmon.
    """
    if fge is not None and ec is not None:
        if ej is not None and anh is not None:
            raise ValueError('Too many parameters passed to '
                             '`charge_dispersion_ge_ef`')
        ej, anh = transmon_ej_anh(fge, ec, ng, dim_charge)
    elif fge is not None and anh is not None:
        if ec is not None and ej is not None:
            raise ValueError('Too many parameters passed to '
                             '`charge_dispersion_ge_ef`')
        ec, ej = transmon_ec_ej(fge, anh, ng, dim_charge)

    if ec is None or ej is None:
        raise ValueError('Too few parameters passed to '
                         '`charge_dispersion_ge_ef`')

    dfreqs = transmon_levels(ec, ej, ng, dim_charge)
    dfreqs -= transmon_levels(ec, ej, ng + 0.5, dim_charge)
    return dfreqs[0], dfreqs[0] - dfreqs[1]


@functools.lru_cache()
def resonator_destroy(dim_resonator: int = 10):
    return np.diag(np.sqrt(np.arange(1, dim_resonator)), 1)


@functools.lru_cache()
def resonator_hamiltonian(wrb: float, dim_resonator: int = 10):
    """Calculate the resonator Hamiltonian.

    Args:
        wrb: Bare resonator frequency.
        dim_resonator: Number of photon number states to use in calculations.

    Returns:
        A (dim_resonator x dim_resonator) numpy matrix, representing the
        Hamiltonian.
    """
    return wrb * np.diag(np.arange(dim_resonator))


def transmon_resonator_levels(ec: float, ej: float, wrb: float, gb: float,
                              ng: float = 0., dim_charge: int = 31,
                              dim_resonator: int = 10,
                              states: List[Tuple[int, int]] =
                              ((1, 0), (2, 0), (0, 1), (1, 1))):
    """Calculate eigenfrequencies of the coupled transmon-resonator Hamiltonian.

    Args:
        ec: Charging energy of the Hamiltonian.
        ej: Josephson energy of the Hamiltonian.
        wrb: Bare resonator frequency.
        gb: Bare transmon-resonator coupling strength.
        ng: Charge offset of the Hamiltonian.
        dim_charge: Number of charge states to use in calculations.
        dim_resonator: Number of photon number states to use in calculations.
        states: A list of tuples, corresponding to the transmon and resonator
                level indices for which to calculate the energy levels.

    Returns:
        A list of eigenvalues of the coupled transmon-resonator Hamiltonian
        for the specified states with the ground-state energy subtracted.
    """
    id_mon = np.diag(np.ones(dim_charge))
    id_res = np.diag(np.ones(dim_resonator))
    ham_mon = transmon_hamiltonian(ec, ej, ng, dim_charge)
    ham_res = resonator_hamiltonian(wrb, dim_resonator)
    n_mon = transmon_charge(ng, dim_charge)
    a_res = resonator_destroy(dim_resonator)
    ham_int = gb * np.kron(n_mon, a_res + a_res.T)
    ham = np.kron(ham_mon, id_res) + np.kron(id_mon, ham_res) + ham_int

    levels_full, states_full = np.linalg.eigh(ham)
    levels_transmon, states_transmon = np.linalg.eigh(ham_mon)
    states_transmon = states_transmon[:, np.argsort(levels_transmon)]
    return_idxs = []
    for kt, kr in states:
        bare_state = np.kron(states_transmon[:, kt],
                             np.arange(dim_resonator) == kr)
        return_idxs.append(np.argmax(np.abs(states_full.T @ bare_state)))
    return levels_full[return_idxs] - levels_full.min()


def transmon_resonator_fge_anh_wrg_chi(ec: float, ej: float, wrb: float,
                                       gb: float, ng: float = 0.,
                                       dim_charge: int = 31,
                                       dim_resonator: int = 10):
    """Calculate observable frequencies of a coupled transmon-resonator system.

    Calculates the first transition frequency and anharmonicity of the
    transmon with the resonator in the ground state, the resonator frequency
    for the qubit in the ground state and the dispersive shift of the resonator.

    Args:
        ec: Charging energy of the Hamiltonian.
        ej: Josephson energy of the Hamiltonian.
        wrb: Bare resonator frequency.
        gb: Bare transmon-resonator coupling strength.
        ng: Charge offset of the Hamiltonian.
        dim_charge: Number of charge states to use in calculations.
        dim_resonator: Number of photon number states to use in calculations.

    Returns:
        A tuple of 1) qubit transition frequency, 2) qubit anharmonicity,
        3) the resonator frequency for transmon ground state, and 4) the
        dispersive shift.
    """
    f10, f20, f01, f11 = transmon_resonator_levels(ec, ej, wrb, gb, ng,
                                                   dim_charge,
                                                   dim_resonator)
    return f10, f20 - 2 * f10, f01, (f11 - f10 - f01) / 2


def transmon_resonator_ec_ej_wrb_gb(fge: float, anh: float, wrg: float,
                                    chi: float, ng: float = 0.,
                                    dim_charge: int = 31,
                                    dim_resonator: int = 10):
    """Calculate Hamiltonian parameters of a coupled transmon-resonator system.

    Inverts the function `transmon_resonator_fge_anh_wrg_chi`.

    Args:
        fge: The first transition frequency of the transmon.
        anh: Anharmonicity of the transmon.
        wrg: Resonator frequency for transmon ground state.
        chi: Dispersive shift of the coupled system.
        ng: Charge offset of the Hamiltonian.
        dim_charge: Number of charge states to use in calculations.
        dim_resonator: Number of photon number states to use in calculations.

    Returns:
        A tuple of 1) the charging energy and 2) Josephson energy of the
        transmon, 3) the bare resonator frequency, and 4) the bare coupling
        strength.
    """

    def func(ec_ej_wrb_gb_, fge_anh_wrg_chi, ng_, dim_charge_, dim_resonator_):
        return -fge_anh_wrg_chi + transmon_resonator_fge_anh_wrg_chi(
            *ec_ej_wrb_gb_, ng_, dim_charge_, dim_resonator_)

    ec0 = -anh
    ej0 = -(fge - anh)**2 / 8 / anh
    wrb0 = wrg
    gb0 = np.sqrt(np.abs((fge + anh - wrg) * (fge - wrg) * chi / anh))
    ec_ej_wrb_gb = sp.optimize.fsolve(func, np.array([ec0, ej0, wrb0, gb0]),
                                      args=(np.array([fge, anh, wrg, chi]),
                                            ng, dim_charge, dim_resonator))
    return tuple(ec_ej_wrb_gb)


def transmon_resonator_ej_anh_wrg_chi(fge: float, ec: float, wrb: float,
                                      gb: float, ng: float = 0.,
                                      dim_charge: int = 31,
                                      dim_resonator: int = 10):
    """Calculate Josephson energy and observable frequencies of a coupled
    transmon-resonator system from the qubit transition frequency and
    Hamiltonian parameters

    Calculates the Josephson energy, the transmon anharmonicity  with the
    resonator in the ground state, the resonator frequency for the qubit in
    the ground state and the dispersive shift of the resonator.

    Args:
        ec: Charging energy of the Hamiltonian.
        fge: The first transition frequency of the transmon.
        wrb: Bare resonator frequency.
        gb: Bare transmon-resonator coupling strength.
        ng: Charge offset of the Hamiltonian.
        dim_charge: Number of charge states to use in calculations.
        dim_resonator: Number of photon number states to use in calculations.

    Returns:
        A tuple of 1) transmon Josephson energy, 2) qubit anharmonicity,
        3) the resonator frequency for transmon ground state, and 4) the
        dispersive shift.
    """

    def func(ej_anh_wrg_chi_, fge_ec_wrb_gb, ng_, dim_charge_, dim_resonator_):
        fge_, ec_, wrb_, gb_ = fge_ec_wrb_gb
        ej, anh, wrg, chi = ej_anh_wrg_chi_
        return -np.array([fge_, anh, wrg, chi]) + \
            transmon_resonator_fge_anh_wrg_chi(ec_, ej, wrb_, gb_, ng_,
                                               dim_charge_, dim_resonator_)

    anh0 = -ec
    ej0 = (fge + ec)**2 / 8 / ec
    wrg0 = wrb
    chi0 = gb**2 * ec / (fge - ec - wrb) / (fge - wrb)
    ej_anh_wrg_chi = sp.optimize.fsolve(func, np.array([ej0, anh0, wrg0, chi0]),
                                        args=(np.array([fge, ec, wrb, gb]),
                                              ng, dim_charge, dim_resonator))
    return tuple(ej_anh_wrg_chi)
