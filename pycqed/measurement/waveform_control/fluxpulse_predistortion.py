import numpy as np
import scipy.signal as signal
import logging
import os
from copy import deepcopy

def import_iir(filename):
    """
    imports csv files generated with Mathematica notebooks of the form
    a1_0,b0_0,b1_0
    a1_1,b0_1,b1_1
    a1_2,b0_2,b1_2
    .
    .
    .

    args:
        filename : string containging to full path of the file (or only the filename if in same directory)

    returns:
        [aIIRfilterLis,bIIRfilterList] : list of two numpy arrays compatable for use
        with the scipy.signal.lfilter() function
        used by filterIIR() function

    """
    IIRfilterList = np.loadtxt(filename,
                               delimiter=',')

    if len(IIRfilterList.shape) == 1:
        IIRfilterList = np.reshape(IIRfilterList,(1,len(IIRfilterList)))

    aIIRfilterList = np.transpose(np.vstack((np.ones(len(IIRfilterList)),
                                             -IIRfilterList[:,0])))
    bIIRfilterList = IIRfilterList[:,1:]

    return [aIIRfilterList,bIIRfilterList]


def filter_fir(kernel,x):
    """
    function to apply a FIR filter to a dataset

    args:
        kernel: FIR filter kernel
        x:      data set
    return:
        y :     data convoluted with kernel, aligned such that pulses do not
                shift (expects kernel to have a impulse like peak)
    """
    iMax = kernel.argmax()
    y = np.convolve(x,kernel,mode='full')[iMax:(len(x)+iMax)]
    return y


def filter_iir(aIIRfilterList, bIIRfilterList, x):
    """
    applies IIR filter to the data x (aIIRfilterList and bIIRfilterList are load by the importIIR() function)

    args:
        aIIRfilterList : array containing the a coefficients of the IIR filters
                         (one row per IIR filter with coefficients 1,-a1,-a2,.. in the form required by scipy.signal.lfilter)
        bIIRfilterList : array containing the b coefficients of the IIR filters
                         (one row per IIR filter with coefficients b0, b1, b2,.. in the form required by scipy.signal.lfilter)
        x : data array to be filtered

    returns:
        y : filtered data array
    """
    y = x
    for a,b in zip(aIIRfilterList,bIIRfilterList):
        y = signal.lfilter(b,a,y)
    return y


def gaussian_filter_kernel(sigma,nr_sigma,dt):
    """
    function to generate a Gaussian filter kernel with specified sigma and
    filter kernel width (nr_sigma).

    Args:
        sigma (float): width of the Gaussian
        nr_sigma (int): specifies the length of the filter kernel
        dt (float): AWG sampling period

    Returns:
        kernel (numpy array): Gaussian filter kernel
    """
    nr_samples = int(nr_sigma*sigma/dt)
    if nr_samples == 0:
        logging.warning('sigma too small (much smaller than sampling rate).')
        return np.array([1])
    gauss_kernel = signal.gaussian(nr_samples, sigma/dt, sym=False)
    gauss_kernel = gauss_kernel/np.sum(gauss_kernel)
    return np.array(gauss_kernel)


def scale_and_negate_IIR(filter_coeffs, scale):
    for i in range(len(filter_coeffs[0])):
        filter_coeffs[0][i][1] *= -1
    filter_coeffs[1][0] /= scale


def combine_FIR_filters(kernels):
    if hasattr(kernels[0], '__iter__'):
        kernel_combined = kernels[0]
        for kernel in kernels[1:]:
            kernel_combined = np.convolve(kernel, kernel_combined)
        return kernel_combined
    else:
        return kernels


def convert_expmod_to_IIR(expmod, dt, inverse_IIR=True):
    """
    Convert an exponential model A + B * exp(- t/tau) (or a list of such
    models) to a first-order IIR filter (or a list of such filters).

    :param expmod: list of exponential models  in the form
        [[A_0, B_0, tau_0], ... ], or a single exponential model
        [A_0, B_0, tau_0].
    :param dt: (float) AWG sampling period
    :param inverse_IIR: (bool, default: True) whether the IIR filters inverting
        the exponential model should be returned.

    :return: A list of IIR filter coefficients in the form
        [aIIRfilterList, bIIRfilterList] according to the definition in
        filter_iir(). If expmod is a single exponential model, a single
        filter is returned in the form [a, b].
    """
    if hasattr(expmod[0], '__iter__'):
        iir = [convert_expmod_to_IIR(e, dt, inverse_IIR) for e in expmod]
        a = [i[0] for i in iir]
        b = [i[1] for i in iir]
    else:
        A, B, tau = expmod
        if isinstance(tau, list):  # sum of exp mod
            import sympy
            N = len(tau)
            a = sympy.symbols(','.join([f'a{i}' for i in range(N + 1)]))
            tau_s = sympy.symbols(','.join([f'tau{i + 1}' for i in range(
                N)]))
            T = sympy.symbols('T')
            z = sympy.symbols('z')
            p = 2 / T * (1 - 1 / z) / (1 + 1 / z)
            r = [sympy.prod(
                [tau * p + 1 for i, tau in enumerate(tau_s) if i != j]) for j
                in
                range(N)]
            s = sympy.prod([tau * p + 1 for tau in tau_s])
            f = (a[0] * s + sum(
                [r * tau * a * p for r, tau, a in zip(r, tau_s, a[1:])])) / s

            n, d = sympy.fraction(f.simplify())
            sym_coeffs_n = n.as_poly(z).all_coeffs()
            coeffs_n = sympy.lambdify([a] + [tau_s] + [T], sym_coeffs_n)
            sym_coeffs_d = d.as_poly(z).all_coeffs()
            coeffs_d = sympy.lambdify([a] + [tau_s] + [T], sym_coeffs_d)
            b = np.array(coeffs_d([A] + B, tau, dt))
            a = np.array(coeffs_n([A] + B, tau, dt))
        else:
            if 1 / tau < 1e-14:
                a, b = np.array([1, -1]), np.array([A + B, -(A + B)])
            else:
                a = np.array(
                    [(A + (A + B) * tau * 2 / dt), (A - (A + B) * tau * 2 / dt)])
                b = np.array([1 + tau * 2 / dt, 1 - tau * 2 / dt])
        if not inverse_IIR:
            a, b = b, a
        b = b / a[0]
        a = a / a[0]
    return [a, b]


def convert_IIR_to_expmod(filter_coeffs, dt, inverse_IIR=True):
    """
    Convert a first-order IIR filter (or a list of such filters) to an
    exponential model A + B * exp(- t/tau) (or a list of such models).

    :param filter_coeffs: IIR coefficients in the form
        [aIIRfilterList, bIIRfilterList] according to the definition in
        filter_iir(). Instead of a list, also single filter is accepted, i.e.,
        [a, b] will be interpreted as [[a], [b]].
    :param dt: (float) AWG sampling period
    :param inverse_IIR: (bool, default: True) whether the IIR filters should
        be interpreted as the filters inverting the exponential model

    :return: A list of exponential models is returned in the form
        [[A_0, B_0, tau_0], ... ], or a single exponential model
        [A_0, B_0, tau_0] if filter_coeffs are of the form [a, b].
    """
    if hasattr(filter_coeffs[0][0], '__iter__'):
        expmod = [convert_IIR_to_expmod([a, b], dt, inverse_IIR) for [a, b] in
                  zip(filter_coeffs[0], filter_coeffs[1])]
    else:
        [a, b] = filter_coeffs
        if not inverse_IIR:
            a, b = b, a
            b = b / a[0]
            a = a / a[0]
        gamma = np.mean(b)
        if np.abs(gamma) < 1e-14:
            A, B, tau =  1, 0, np.inf
        else:
            a_, b_ = a / gamma, b / gamma
            A = 1 / 2 * (a_[0] + a_[1])
            tau = 1 / 2 * (b_[0] - b_[1]) * dt / 2
            B = 1 / 2 * (a_[0] - a_[1]) * dt / (2 * tau) - A
        expmod = [A, B, tau]
    return expmod


def process_filter_coeffs_dict(flux_distortion, datadir=None, default_dt=None):
    """
    Prepares a distortion dictionary that can be stored into pulsar
    {AWG}_{channel}_distortion_dict based on information provided in a
    dictionary.

    :param flux_distortion: (dict) A dictionary of the format defined in
        QuDev_transmon.DEFAULT_FLUX_DISTORTION. In particular, the following
        keys FIR_filter_list and IIR_filter are processed by this function.
        They are list of dicts with a key 'type' and further keys.
        type 'csv': load filter from the file specified under the key
            'filename'. In case of an IIR filter, the filter will in
            addition be scaled by the value in the key 'scale_IIR'.
        type 'Gaussian': can be used for FIR filters. Gaussian kernel with
            parameters 'sigma', 'nr_sigma', and 'dt' specified in the
            respective keys. See gaussian_filter_kernel()
        type 'expmod': can be used for IIR filters. A filter that inverts an
            exponential model specified by the keys 'A', 'B', 'tau', and 'dt'.
            See convert_expmod_to_IIR().
        If flux_distortion in addition contains the keys FIR and/or IIR,
        the filters specified as described above will be appended to those
        already existing in FIR/IIR.
        If flux_distortion is already in the format to be stored in pulsar,
        it is returned unchanged.
    :param datadir: (str) base dir for loading csv files. If None,
        it is assumed that the specified filename includes the full path.
    :param default_dt: (float) AWG sampling period to be used in cases where
        'dt' is needed, but not specified in a filter dict.

    """

    filterCoeffs = {}
    for fclass in 'IIR', 'FIR':
        filterCoeffs[fclass] = flux_distortion.get(fclass, [])
        if fclass == 'IIR' and len(filterCoeffs[fclass]) > 1:
            # convert coefficient lists into list of filters so that we can
            # append if needed
            filterCoeffs[fclass] = [[[a], [b]] for a, b in zip(
                filterCoeffs['IIR'][0], filterCoeffs['IIR'][1])]
        for f in flux_distortion.get(f'{fclass}_filter_list', []):
            if f['type'] == 'Gaussian' and fclass == 'FIR':
                coeffs = gaussian_filter_kernel(f.get('sigma', 1e-9),
                                                f.get('nr_sigma', 40),
                                                f.get('dt', default_dt))
            elif f['type'] == 'expmod' and fclass == 'IIR':
                expmod = f.get('expmod', None)
                if expmod is None:
                    expmod = [f.get('A'), f.get('B'), f.get('tau')]
                if not hasattr(expmod[0], '__iter__'):
                    expmod = [expmod]
                coeffs = convert_expmod_to_IIR(expmod,
                                               dt=f.get('dt', default_dt))
            elif f['type'] == 'csv':
                if datadir is not None:
                    filename = os.path.join(datadir,
                                            f['filename'].lstrip('\\'))
                else:
                    filename = f['filename']
                if fclass == 'IIR':
                    coeffs = import_iir(filename)
                    scale_and_negate_IIR(
                        coeffs,
                        f.get('scale_IIR', flux_distortion['scale_IIR']))
                else:
                    coeffs = np.loadtxt(filename)
            else:
                raise NotImplementedError(f"Unknown filter type {f['type']}")
            filterCoeffs[fclass].append(coeffs)

    if len(filterCoeffs['FIR']) > 0:
        filterCoeffs['FIR'] = [
            combine_FIR_filters(filterCoeffs['FIR'])]
    else:
        del filterCoeffs['FIR']
    if len(filterCoeffs['IIR']) > 0:
        filterCoeffs['IIR'] = [
            np.concatenate([i[0] for i in filterCoeffs['IIR']]),
            np.concatenate([i[1] for i in filterCoeffs['IIR']])]
    else:
        del filterCoeffs['IIR']
    return filterCoeffs
