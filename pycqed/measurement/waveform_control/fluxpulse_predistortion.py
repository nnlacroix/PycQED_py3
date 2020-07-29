import numpy as np
import scipy.signal as signal
import logging

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
