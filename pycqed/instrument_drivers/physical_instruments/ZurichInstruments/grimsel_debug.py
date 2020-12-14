import os
import sys
import numpy as np
import time
import struct
from enum import IntEnum
from fnmatch import fnmatch

from scipy.interpolate import interp1d
from scipy.signal import welch

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator, AutoMinorLocator, LinearLocator

# Load the LabOne API
import zhinst.ziPython as zi
import zhinst.utils as utils

# Constants

# Debug register assignments
DEBUG_REG_REG_MIXER_FREQUENCIES_OFFSET32  = 0x00
DEBUG_REG_REG_IQ_VALUES_OFFSET32          = 0x08
# DEBUG_REG_REG_OUTPUT_EN_OFFSET32 is deprecated. Use /RAW/DACSOURCES/<n>/PATHS/<m>/ON instead.
# DEBUG_REG_REG_OUTPUT_EN_OFFSET32          = 0x0E
DEBUG_REG_REG_OUTPUT_CURRENT_OFFSET32     = 0x0F
DEBUG_REG_REG_FREEZE_CAL_OFFSET32         = 0x10
DEBUG_REG_REG_NYQUIST_ZONE                = 0x11
DEBUG_REG_REG_DEC_FACTOR                  = 0x12
DEBUG_REG_REG_ADC_MIXER_TEST_ID           = 0x13
DEBUG_REG_REG_IPL_FACTOR                  = 0x14
DEBUG_REG_REG_DAC_MIXER_TEST_ID           = 0x15
DEBUG_REG_REG_OUTPUT_SINES_GAIN           = 0x16
# DEBUG_REG_REG_OUTPUT_RANDOM_EN is deprecated. Use /RAW/DACSOURCES/<n>/SOURCESELECT instead.
# DEBUG_REG_REG_OUTPUT_RANDOM_EN            = 0x17
DEBUG_REG_REG_OUTPUT_ZERO_Q               = 0x18
DEBUG_REG_REG_RESLOG_STATE                = 0x19
DEBUG_REG_REG_SPECTR_ENA                  = 0x1A
DEBUG_REG_REG_DAC_DECODER_MODE            = 0x1B
DEBUG_REG_REG_DAC_INVERSE_SINC_FIR        = 0x1C
DEBUG_REG_REG_ADC_MIXER_FREQUENCY         = 0x1D
DEBUG_REG_REG_COMMENSURATE_TEST           = 0x1E
DEBUG_REG_REG_QA_SG_CTRL                  = 0x1F
DEBUG_REG_REG_SYNTH_FREQ_0                = 0x20
DEBUG_REG_REG_SYNTH_FREQ_1                = 0x21
DEBUG_REG_REG_SYNTH_FREQ_2                = 0x22
DEBUG_REG_REG_SYNTH_FREQ_3                = 0x23
DEBUG_REG_REG_ZSYNC_LOOPTHROUGH           = 0x24
DEBUG_REG_REG_READOUT_ENA                 = 0x25
DEBUG_REG_REG_TRIG_CTRL_START             = 0x26
DEBUG_REG_REG_TRIG_CTRL_SRC_SEL           = 0x27
DEBUG_REG_REG_SEQENCER_LENGTH             = 0x28

XRFDC_MIXER_MODE_OFF                      = 0x0
XRFDC_MIXER_MODE_C2C                      = 0x1
XRFDC_MIXER_MODE_C2R                      = 0x2
XRFDC_MIXER_MODE_R2C                      = 0x3
XRFDC_MIXER_MODE_R2R                      = 0x4

XRFDC_MIXER_TYPE_COARSE                   = 0x1
XRFDC_MIXER_TYPE_FINE                     = 0x2
XRFDC_MIXER_TYPE_OFF                      = 0x0
XRFDC_MIXER_TYPE_DISABLED                 = 0x3

XRFDC_COARSE_MIX_OFF                      = 0x0
XRFDC_COARSE_MIX_SAMPLE_FREQ_BY_TWO       = 0x2
XRFDC_COARSE_MIX_SAMPLE_FREQ_BY_FOUR      = 0x4
XRFDC_COARSE_MIX_MIN_SAMPLE_FREQ_BY_FOUR  = 0x8
XRFDC_COARSE_MIX_BYPASS                   = 0x10

# DAC decoder modes
GML_HH_RFDC_DAC_DECODER_MODE_MAX_SNR        = 0x0
GML_HH_RFDC_DAC_DECODER_MODE_MAX_LINEARITY  = 0x1

# global variables
_device_name = ''
_scope_length = 0
_segments_count = 1
_scope_averaging_count = 1
_recorded_data = None
_channel_enable = [0, 0, 0, 0]
_input_select = [[],[],[],[]]
_daq = None

# define format for plotting
# TODO(YS): write some setters and getters for these settings
_plot_color = ['C0','C1','C2','C3']

def recorded_data_get():
    return _recorded_data

# Functions defining the scope data input
def scope_data_is_complex( input_select ):
    return (input_select >= 0 and input_select <= 3)

def scope_data_is_real( input_select ):
    return (input_select >= 4 and input_select <= 7)

def scope_data_is_trig( input_select ):
    return (input_select >= 8)

def scope_fs_get():
    Fs = 4e9  # [Hz]
    scope_time_setting = _daq.getInt(f"/{_device_name}/scopes/0/time")
    decimation_rate = 2**scope_time_setting
    return Fs / decimation_rate

def get_input_name( input_select ) :
    if input_select < 4 :
        return "RF in %d" % input_select
    elif input_select < 8 :
        return "AUX in %d" % (input_select - 4)
    else :
        return "TRIG in"

def plot_legend_txt_get():
    global input_select
    plot_legend_txt = []
    for i in range(4):
        plot_legend_txt.append(get_input_name(_input_select[i]))
    return plot_legend_txt


def grimsel_connect( daq, device_name, device_interface = '1GbE' ):
    global _daq, _device_name
    _daq = daq
    _device_name = device_name
    _daq.connectDevice(_device_name, device_interface)

# helper functions
def UINT2FLOAT(integer_value):
    return struct.unpack('!f', struct.pack('!I', integer_value))[0]

def FLOAT2UINT(float_value):
    return struct.unpack('!I', struct.pack('!f', float_value))[0]

# Converts the power relative to full scale to raw I/Q values
# the output of this function is on the diagonal of the I/Q plane (I = Q).
# the raw I/Q values are represented as two concatenated 16-bit integers
def POWER2UINT_IQ(rel_power):
    assert 0 <= rel_power <=1, "rel_power must be between 0 and 1"
    i_value = np.sqrt(rel_power)
    i_uint = int(i_value*0x7FFF)
    return i_uint + (i_uint << 16)


def unsigned(signed_value, bitsize):
    return signed_value if signed_value >= 0 else signed_value + (1 << bitsize)

def signed(unsigned_value, bitsize):
    return unsigned_value if unsigned_value < (1 << bitsize-1) else unsigned_value - (1 << bitsize)

# wrapper functions ala zishell:
def getv(node):
    global _daq, _device_name
    path = '/' + _device_name + '/' + node
    path = path.lower()
    _daq.getAsEvent(path)
    tmp = _daq.poll(0.5, 500, 4, True)
    if path in tmp:
        return tmp[path]
    else:
        return None
    return tmp[path]

def find(*args):
    nodes = _daq.listNodes('/', 7)
    if len(args) and args[0]:
        for m in args:
            nodes = [k.lower() for k in nodes if fnmatch(k.lower(), m.lower())]
    return nodes


def wait_until_node_value_poll(node, expected_value, sleep_time=0.05, max_repetitions=20):
    """Polls a node until it has the expected value."""

    global _daq, _device_name
    _daq.sync()

    for i in range(max_repetitions) :
        readback_value = _daq.getInt(f"/{_device_name}/{node}")
        if readback_value == expected_value:
            return readback_value
        time.sleep(0.05)

    raise ValueError(f"wait_until_node_value_poll() on node {node} did not read the expected value ({expected_value}) after {max_repetitions} iterations. Last read value was: {readback_value}")

    return readback_value


def scope_configure( input_select, channel_enable, scope_length, segments_count=1, scope_time_setting=0, scope_averaging_count=1, scope_trig_delay=0):
    global _daq, _device_name, _input_select, _channel_enable, _scope_length, _segments_count, _scope_averaging_count
    _input_select    = input_select
    _channel_enable  = channel_enable
    _scope_length    = scope_length
    _segments_count  = segments_count
    _scope_averaging_count = scope_averaging_count

    # config the scope
    _daq.setInt(f'/{_device_name}/scopes/0/length', scope_length)
    _daq.setInt(f'/{_device_name}/scopes/0/segments/count', segments_count)
    if segments_count > 1 :
        _daq.setInt(f'/{_device_name}/scopes/0/segments/enable', 1)
    else :
        _daq.setInt(f'/{_device_name}/scopes/0/segments/enable', 0)

    if scope_averaging_count > 1 :
        _daq.setInt(f'/{_device_name}/scopes/0/averaging/enable', 1)
    else :
        _daq.setInt(f'/{_device_name}/scopes/0/averaging/enable', 0)
    _daq.setInt(f'/{_device_name}/scopes/0/averaging/count', scope_averaging_count)

    for channel in range(4):
        _daq.setInt(f'/{_device_name}/scopes/0/channels/{channel}/inputselect/', _input_select[channel])
        _daq.setInt(f'/{_device_name}/scopes/0/channels/{channel}/enable/', _channel_enable[channel])

    # decimation rate (scope_time_setting)
    _daq.setInt(f"/{_device_name}/scopes/0/time", scope_time_setting)

    _daq.setDouble(f'/{_device_name}/scopes/0/trigger/delay', scope_trig_delay)


def scope_enable() :
    _daq.setInt(f'/{_device_name}/scopes/0/enable', 1)


def scope_disable() :
    _daq.setInt(f'/{_device_name}/scopes/0/enable', 0)


def trig_start(trigger_count = 1, wait_time_s = 0.01) :
    """ Generate a trigger signal """

    for i in range(trigger_count) :
        _daq.setInt(f"/{_device_name}/RAW/DEBUG/{DEBUG_REG_REG_TRIG_CTRL_START}/VALUE", 1)
        time.sleep(wait_time_s)


def trig_source_select(value) :
    """ Select trigger source """

    _daq.setInt(f"/{_device_name}/RAW/DEBUG/{DEBUG_REG_REG_TRIG_CTRL_SRC_SEL}/VALUE", value)


def scope_time_get():

    Fs = scope_fs_get();

    # generate a time signal
    return np.array(range(0, _scope_length * _segments_count)) / Fs


def scope_get_trig_data(vector_data, trig_sel) :
    """ Helper function to get trigger data from vector_data for the selected trigger input """

    trig_data = np.zeros(len(vector_data) // 2)
    for i in range(len(vector_data) // 2) :
        #if vector_data[i * 2 + trig_sel] & (1 << 12) != 0 :
        #    trig_data[i] = 1
        trig_data[i] = vector_data[i * 2 + trig_sel] / -4096 / _scope_averaging_count

    return trig_data

def scope_read_data(trig_sel = 0):
    # wait until ready, then read out the recorded data
    #
    # trig_sel: 0 or 1 to select one of the recorded trigger inputs (always two are recorded on the same scope channels)

    global _recorded_data
    global _device_name

    # needed because the dataserver does not remove the extra header information from
    # the vector data... will be removed as soon as this is fixed in the dataserver
    extra_header_length = 17

    # wait until DMA is done
    print("Waiting until scope measurement is done")
    wait_until_node_value_poll('SCOPES/0/ENABLE', 0);
    print("Measurement done")

    _daq.sync()

    # read the recorded data
    vector_data   = [[],[],[],[]]
    _recorded_data = [[],[],[],[]]
    vector = None

    for channel in range(4) :
        if _channel_enable[channel] :
            print("channel ", channel)
            vector = getv(f'scopes/0/channels/{channel}/wave')

            _daq.sync()
            vector_data[channel] = vector[0]['vector'].astype('int32')[extra_header_length:extra_header_length + 2 * _scope_length * _segments_count]

            # convert to complex if the channel has complex data (i.e. real / imaginary samples interleaved)
            if scope_data_is_complex(_input_select[channel]) :
                # change the matrix such that first column is the real part, the second column is the imaginary part
                _recorded_data[channel] = np.reshape(vector_data[channel], (int(len(vector_data[channel])/2), 2 ))
                # multiply by (1, i)
                _recorded_data[channel] = np.matmul(_recorded_data[channel], [[1], [-1j]])
                # transpose
                _recorded_data[channel] = np.matrix.transpose(_recorded_data[channel])[0]
            elif scope_data_is_real(_input_select[channel]) :
                _recorded_data[channel] = vector_data[channel]
            elif scope_data_is_trig(_input_select[channel]) :
                _recorded_data[channel] = scope_get_trig_data(vector_data[channel], trig_sel)
            else :
                print("Error: format of scope data not defined for _input_select = %i" % _input_select[channel])
        else :
            vector_data[channel] = np.zeros(_scope_length * _segments_count)

    for channel in range(4) :
        if _channel_enable[channel] :
            print("length _recorded_data[%d]: " % channel, len(_recorded_data[channel]))

def scope_plot_time_domain():

    global _recorded_data
    global _plot_color

    plot_legend_txt = plot_legend_txt_get()

    t = scope_time_get()

    plt.rcParams['figure.figsize'] = [15, 10] # size of the plot

    # plot the signal in the time domain
    subplot = 0
    for channel in range(4) :
        if _channel_enable[channel] :
            plt.subplot(4, 1, subplot + 1)
            subplot += 1

            if scope_data_is_complex(_input_select[channel]) :
                plt.plot(t[::2] * 1e6, np.real(_recorded_data[channel]), '.-' ,color=_plot_color[channel])
                plt.plot(t[::2] * 1e6, np.imag(_recorded_data[channel]), 'x-' ,color=_plot_color[channel])
            elif scope_data_is_real(_input_select[channel]) :
                plt.plot(t * 1e6, _recorded_data[channel], '.-' ,color=_plot_color[channel])
            elif scope_data_is_trig(_input_select[channel]) :
                plt.plot(t[::2] * 1e6, _recorded_data[channel], '.-' ,color=_plot_color[channel])
                plt.ylim([-0.1,1.1]);
            else :
                print("Error: the plot format of this channel is not defined")

            # plot settings
            plt.grid(True)
            plt.xlabel('time [us]')
            plt.ylabel('digital value')
            #plt.title('input signals in the time domain')
            plt.legend([plot_legend_txt[channel]])

            # zoom
            #plt.ylim([-2**11,2**11]);
            #plt.ylim([-100,100]);
            #plt.xlim([0,t[-1] * 1e6]); # zoom full
            #plt.xlim([0,0.01]); # zoom to 0.01us
            #plt.xlim([0,0.1])

    plt.show()


def scope_calculate_spectrum():
    # calculate the spectrum

    global f
    global input_signal_psd_dB
    global _recorded_data

    Fs = scope_fs_get()

    input_signal_psd = [[],[],[],[]]
    input_signal_psd_dB = [[0],[0],[0],[0]]

    f = [[],[],[],[]]
    for channel in range(4) :
        if _channel_enable[channel] :
            if scope_data_is_complex(_input_select[channel]) :

                # PSD (welch)
                # n_fft = int(_scope_length*_segments_count/128)
                #f[channel], input_signal_psd[channel] = welch(_recorded_data[channel], Fs/2, 'hann', nperseg=n_fft, nfft=n_fft, return_onesided=False)
                #f[channel] = np.fft.fftshift(f[channel])

                # FFT
                n_fft = len(_recorded_data[channel])
                input_signal_psd[channel] = abs(np.fft.fft(_recorded_data[channel], n_fft) )**2 / n_fft
                f[channel] = np.asarray(range(n_fft)) * Fs / n_fft / 2 - Fs/4

                input_signal_psd[channel] = np.fft.fftshift(input_signal_psd[channel])

            else : # scope data is real

                # PSD (welch)
                # n_fft = int(_scope_length*_segments_count/128)
                #f[channel], input_signal_psd[channel] = welch(_recorded_data[channel], Fs, 'hann', nperseg=n_fft, nfft=n_fft, return_onesided=True)

                # FFT
                n_fft = len(_recorded_data[channel])
                input_signal_psd[channel] = abs(np.fft.rfft(_recorded_data[channel], n_fft) )**2 / n_fft
                f[channel] = np.asarray(range(len(input_signal_psd[channel]))) * Fs / n_fft

            # scale it such that the full scale sine wave peaks at 0dB
            # ToDo: how to scale the signals properly?
            dB_scale_factor = 200e-9 / n_fft
            input_signal_psd_dB[channel] = 10 * np.log10(np.asarray(input_signal_psd[channel]) * dB_scale_factor)


    # for comparison, create a sine wave signal
    # freqency_sine = 375e6
    # sine_wave  = np.sin(t * 2 * np.pi * freqency_sine) * 2**12;
    # sine_wave += np.random.normal(scale=1.0, size=len(t))
    # n_fft = len(_recorded_data[channel])
    # sine_f, sine_psd = welch( sine_wave,Fs, 'hann', nperseg=n_fft, nfft=n_fft, return_onesided=False)
    # sine_psd    = np.fft.fftshift(sine_psd)
    # sine_f      = np.fft.fftshift(sine_f)
    # sine_psd_dB = 10 * np.log10(sine_psd * dB_scale_factor)


def scope_plot_frequency_domain(xlim=None):
    # plot the spectrum

    global f
    global _plot_color

    plot_legend_txt = plot_legend_txt_get()

    # plot settings
    span_MHz = scope_fs_get() / 2 / 1e6

    plt.rcParams['figure.figsize'] = [15, 10] # size of the plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    legend_text = []
    for channel in range(4) :
        if _channel_enable[channel] :
            plt.plot(f[channel] / 1e6, input_signal_psd_dB[channel], _plot_color[channel])
            legend_text.append(plot_legend_txt[channel])
    #plt.plot(sine_f / 1e9, sine_psd_dB, '-k')

    # plot formatting
    plt.grid()
    plt.xlabel('f [MHz]')
    plt.ylabel('[dBFS]')
    #plt.title('down-converted input signals in the frequency domain')
    plt.legend(legend_text)

    # Grid
    ax.get_xaxis().set_major_locator(AutoLocator())
    ax.get_xaxis().set_minor_locator(AutoMinorLocator(5))

    ax.get_yaxis().set_major_locator(AutoLocator())
    ax.get_yaxis().set_minor_locator(AutoMinorLocator(2))

    ax.grid(which='minor', alpha=0.4)
    ax.grid(which='major', alpha=0.9)

    # Zoom
    #plt.xlim([-span_MHz/2, span_MHz/2]); # two sided
    #plt.xlim([0, span_MHz]) # one sided
    if xlim is not None:
        plt.xlim(xlim)
    else:
        plt.xlim([-span_MHz/2 * 1.05, span_MHz * 1.05]); # one and two sided
    plt.ylim([-100,0])

    plt.show()


# function to start Result Logger
def reslog_start(reslog_length) :
    _daq.setInt(f"/{_device_name}/qas/0/result/length", reslog_length)
    _daq.setInt(f"/{_device_name}/qas/0/result/enable", 1)


# function to reset the Result Logger
def reslog_reset() :
    #reset reslog
    _daq.setInt(f"/{_device_name}/qas/0/result/reset", 1)
    time.sleep(0.5)
    wait_until_node_value_poll("qas/0/result/reset", 0)


# function to print the state of the Result Logger
def reslog_print_nodes() :
    print( "qas/0/result/... ")
    print( "length        %d" % _daq.getInt(f"/{_device_name}/qas/0/result/length"))
    print( "averages      %d" % _daq.getInt(f"/{_device_name}/qas/0/result/averages"))
    print( "acquired      %d" % _daq.getInt(f"/{_device_name}/qas/0/result/acquired"))
    print( "enable        %d" % _daq.getInt(f"/{_device_name}/qas/0/result/enable"))
    print( "reset         %d" % _daq.getInt(f"/{_device_name}/qas/0/result/reset"))
    print( "errors        %d" % _daq.getInt(f"/{_device_name}/qas/0/result/errors"))


def spectr_run(spectr_int_length, spectr_frequencies, wait_time_s = 0.2) :
    """Runs a spectroscopy measurement.

       In a loop over the spectr_frequencies, this function configures the oscillator
       and then triggers the spectroscopy module.
    """

    # configure spectr
    _daq.setInt(f"/{_device_name}/QAS/0/result/source", 0) # source = 0: spectroscopy
    _daq.setInt(f"/{_device_name}/QAS/0/spectroscopy/length", spectr_int_length)

    # start the result logger
    reslog_start( len(spectr_frequencies) )

    # change frequency and start spectr component
    acquired_results = 0
    for freq in spectr_frequencies :

        print("f = %.2f MHz" % (freq / 1e6), end="\r")

        dac_osc_freq_set(freq)
        time.sleep(wait_time_s)

        # generate trigger to start spectroscopy
        trig_start()

        # wait until the result logger shows that the result is recorded
        acquired_results += 1
        wait_until_node_value_poll("QAS/0/RESULT/ACQUIRED", acquired_results)

    print("")

    # wait until reslog is done
    wait_until_node_value_poll("qas/0/result/enable", 0);
    print("Spectroscopy done")


def readout_test(readout_int_length, num_samples, complex_mode, wait_time_s = 0.2) :
    """Runs a test for qbit readout (weighted integration).

       In a loop over the num_samples, this test configures the oscillator
       and then triggers a readout module.
    """

    global _dev
    global _daq

    # configure
    _daq.setInt(f"/{_device_name}/QAS/0/RESULT/SOURCE", 1) # source = 1: readout
    _daq.setInt(f"/{_device_name}/QAS/0/INTEGRATION/LENGTH", readout_int_length)
    _daq.setInt(f"/{_device_name}/QAS/0/INTEGRATION/COMPLEXMODE", complex_mode) # 0: real multiplication mode, 1: complex multiplication mode

    # start the result logger
    reslog_start( num_samples )

    # start spectr component
    acquired_results = 0
    for i in range(num_samples) :

        print(f"sample {i + 1}", end="\r")

        time.sleep(wait_time_s)

        # generate trigger to start readout
        trig_start()

        # wait until the result logger shows that the result is recorded
        acquired_results += 1
        wait_until_node_value_poll("QAS/0/RESULT/ACQUIRED", acquired_results)

    # wait until reslog is done
    wait_until_node_value_poll("QAS/0/RESULT/ENABLE", 0)

    print("\nreadout done")


# Function to get the Result Logger data
def reslog_data_get(sigins, qbits) :

    reslog_data = dict()

    _daq.subscribe(f'/{_device_name}/qas/0/result/sigins/*')

    for sigin in sigins :
        for qbit in qbits :

            node_name = f'/{_device_name}/qas/0/result/sigins/{sigin}/data/{qbit}/wave'

            print(f"getting data from {node_name}", end="\r");

            _daq.getAsEvent(node_name)
            vector = _daq.poll(0.1, 10, flat=True)
            vector = vector[node_name][0]
            reslog_data.update({f'in{sigin}_q{qbit}': vector['vector']})

    print("\ndone");

    return reslog_data


# function to plot data of the Result Logger
def reslog_data_plot(reslog_data, spectr_frequencies, enable_phase_plot = False, ena_logy = True) :

    plt.rcParams['figure.figsize'] = [15, 10] # size of the plot

    first_channel = list(reslog_data.keys())[0]

    # plot amplitude
    plt.subplot(2, 1, 1)
    for channel in reslog_data :

        if ena_logy :
            # calculate dB values
            ylabel_txt = "amplitude [dB]"
            plot_data = 20 * np.log10 (np.abs(reslog_data[channel]))
            #plot_data = plot_data - 20 * np.log10 (np.abs(reslog_data[first_channel][0])) # shift such that amplitude response starts at 0dB
        else :
            ylabel_txt = "amplitude"
            plot_data = np.abs(reslog_data[channel])

        if spectr_frequencies[0] != spectr_frequencies[-1] :
            plt.plot(spectr_frequencies / 1e6, plot_data, '.-')
        else :
            plt.plot(plot_data, '.-')

    plt.xlabel("frequency [MHz]");
    plt.ylabel(ylabel_txt);
    plt.legend([*reslog_data]);
    plt.grid()

    # plot phase
    if enable_phase_plot :
        for channel in reslog_data:
            plt.subplot(2, 1, 2)

            reslog_data_angle = np.angle(reslog_data[channel])

            if spectr_frequencies[0] != spectr_frequencies[-1] :
                plt.plot( spectr_frequencies / 1e6, reslog_data_angle / np.pi * 180, '.-')
            else :
                plt.plot(reslog_data_angle / np.pi * 180, '.-')

        plt.xlabel("frequency [MHz]");
        plt.ylabel("angle [deg]");
        plt.grid()

    plt.show()



def reslog_data_plot_complex(reslog_data) :
    """Plots data of the result logger in the complex plane"""

    plt.rcParams['figure.figsize'] = [10, 10] # size of the plot

    max_value = 0
    for channel in reslog_data :

        #plt.plot(reslog_data[channel], 'x')
        plt.plot(np.real(reslog_data[channel]), np.imag(reslog_data[channel]), 'x')

        max_value=max(max_value, max(abs(np.real(reslog_data[channel]))))
        max_value=max(max_value, max(abs(np.imag(reslog_data[channel]))))

    plt.legend([*reslog_data]);
    plt.xlabel("real part");
    plt.ylabel("imaginary part");
    plt.grid()

    # zoom such that origin is in the middle
    max_value *= 1.05
    plt.xlim([-max_value, max_value])
    plt.ylim([-max_value, max_value])

    plt.show()


# Freezes / Unfreezes the calibration on the ADC
# adcs is an array defining for which adcs the calibration should be frozen
# TODO(YS): input mapping
def adc_freeze_calibration(adcs):
    freeze_setting = 0
    for i in range(8):
        freeze_setting += (int(adcs[i]) << i)
    _daq.setDouble(f"/{_device_name}/raw/debug/{DEBUG_REG_REG_FREEZE_CAL_OFFSET32}/value", freeze_setting)

# set nyquist zone for ADCs
# Bitwise setting for each ADC (0 = even zones, 1 = odd zones)
def adc_nyquist_set(mask):
    _daq.setInt(f"/{_device_name}/RAW/DEBUG/{DEBUG_REG_REG_NYQUIST_ZONE}/VALUE", mask)

# Disables all DAC outputs
def dac_disable_all_outputs():
    for channel in range(8):
        dac_output_enable_set(channel, 0)

def channel2dacsource_idx(channel_idx):
    """
    Convert from output channel index 0-7 to DACSOURCE index 0-3 and path index 0-1.
    There are 4 so-called DACSOURCES, which can be switched between sine wave, AWG, or random signal
    Each DACSOURCE is connected to a pair of neighbouring DACs.
    Seen from the front panel, DACSOURCE 0 serves the two leftmost DACs and DACSOURCE 3 serves the two rightmost DACs
    Args:
        channel_idx:        output channel index (0-7), 0 corresponds to the leftmost channel
    Returns:
        dacsource_idx:      DACSOURCE index (0-3)
        dacsource_path_idx: DACSOURCE path index (0-1)
    """
    assert channel_idx >= 0 and channel_idx <= 7, 'Output channel index must be between 0 and 7'

    dacsource_idx = int(channel_idx/2)

    # The DACSOURCE path index determines which of the two output channels served by each DACSOURCE should be configured:
    dacsource_path_idx = channel_idx % 2

    return dacsource_idx, dacsource_path_idx

def dac_output_enable_set(channel_idx, enable):
    """
    Enables an output channel
    Args:
        channel_idx:        output channel index (0-7), 0 corresponds to the leftmost channel
        enable:             0: disable, 1: enable
    Returns:
        nothing
    """
    assert enable == 0 or enable == 1, "Argument enable must be either 0 or 1"
    dacsource_idx, dacsource_path_idx = channel2dacsource_idx(channel_idx)
    _daq.setInt(f'/{_device_name}/RAW/DACSOURCES/{dacsource_idx}/PATHS/{dacsource_path_idx}/ON', enable)

class DAC_source(IntEnum):
    sines   = 0 # two-tone sines support
    sg      = 1 # signal generator (AWG)
    random  = 2 # random noise for ADC calibration

def dac_source_select(channel_idx, selection : DAC_source):
    """
    Select between different signal sources for the DAC.
    Args:
        channel_idx:        output channel index (0-7), 0 corresponds to the leftmost channel
        selection:          signal source selection (see DAC_source enumeration)
    Returns:
        nothing
    """
    dacsource_idx, _ = channel2dacsource_idx(channel_idx)
    _daq.setInt(f'/{_device_name}/RAW/DACSOURCES/{dacsource_idx}/SOURCESELECT', int(selection))

# enables the RFdc mixer for all DACs
def dac_mixer_enable():
    # the following enables the fine mixer mode (coarse is not supported at the moment)
    _daq.setInt(f"/{_device_name}/RAW/DEBUG/{DEBUG_REG_REG_DAC_MIXER_TEST_ID}/VALUE", 2)

def dac_mixer_enable_index(index):
    # the following enables the fine mixer mode (coarse is not supported at the moment)
    _daq.setInt(f"/{_device_name}/RAW/RFDAC/{index}/MIXER/TYPE", XRFDC_MIXER_TYPE_FINE)

# disables the RFdc mixer for all DACs
def dac_mixer_disable():
    _daq.setInt(f"/{_device_name}/RAW/DEBUG/{DEBUG_REG_REG_DAC_MIXER_TEST_ID}/VALUE", 0)

# Set DAC mixer frequency in Hz
def dac_mixer_frequency_set(mixer_index, mixer_frequency_hz):
    addr = DEBUG_REG_REG_MIXER_FREQUENCIES_OFFSET32 + mixer_index
    _daq.setInt(f"/{_device_name}/RAW/DEBUG/{addr}/VALUE", FLOAT2UINT(mixer_frequency_hz))

def dac_mixer_frequency_get(mixer_index):
    addr = DEBUG_REG_REG_MIXER_FREQUENCIES_OFFSET32 + mixer_index
    freq = UINT2FLOAT(_daq.getInt(f"/{_device_name}/RAW/DEBUG/{addr}/VALUE"))
    return freq

# Sets the constant I/Q values input to the DAC
# param iq_values: list of 6 32bit words (16 bit I, 16 bit Q)
def dac_iq_values_set(iq_values):
    for i, v in enumerate(iq_values):
        addr = i + DEBUG_REG_REG_IQ_VALUES_OFFSET32
        print("I/Q value path {}: {:05x}".format(i, signed(v, 32)))
        _daq.setInt(f"/{_device_name}/RAW/DEBUG/{addr}/VALUE", signed(v, 32))

# Sets constant output power (in dBFS)
def dac_iq_values_power_set(power):
    rel_power = 10**(power/10)
    rel_power_uint = POWER2UINT_IQ(rel_power)
    dac_iq_values_set(6*[rel_power_uint])

# Sets the gain of the sine wave added to the constant I/Q values
# set this to 0 to disable the second sine wave
def dac_sines_gain_set(gain):
    gain_uint = int(np.abs(gain)*(2**16))
    print ("SINES gain: {:05x}".format(gain_uint))
    addr = DEBUG_REG_REG_OUTPUT_SINES_GAIN
    _daq.setInt("/{:s}/RAW/DEBUG/{:d}/VALUE".format(_device_name, addr), gain_uint)

#%% Oscillator frequency for two-tone support
def dac_osc_freq_set(osc_freq):
    _daq.setDouble("/{:s}/OSCS/0/FREQ".format(_device_name), osc_freq)

# Set ADC mixer frequency in Hz
def adc_all_mixer_frequency_set(mixer_frequency_hz):
    addr = DEBUG_REG_REG_ADC_MIXER_FREQUENCY
    _daq.setInt(f"/{_device_name}/RAW/DEBUG/{addr}/VALUE", FLOAT2UINT(mixer_frequency_hz))

def adc_all_mixer_frequency_get():
    addr = DEBUG_REG_REG_ADC_MIXER_FREQUENCY
    freq = UINT2FLOAT(_daq.getInt(f"/{_device_name}/RAW/DEBUG/{addr}/VALUE"))
    return freq

# sets the RFdc mixer mode for all ADCs
# mode == 0 -> off
# mode == 1 -> coarse fs/4
# mode == 2 -> fine
def adc_mixer_mode_set(mode: int):
    assert 0 <= mode and mode <= 2, 'mode must be 0, 1, or 2'
    # the following enables the fine mixer mode (coarse is not supported at the moment)
    _daq.setInt(f"/{_device_name}/RAW/DEBUG/{DEBUG_REG_REG_ADC_MIXER_TEST_ID}/VALUE", mode)

# Configure settings for commensurate test
# mode == 0 -> off
# mode == 1 -> ~1 GHz
# mode == 2 -> ~2 GHz
def commensurate_test(mode: int):
    assert 0 <= mode and mode <= 2, 'mode must be 0, 1, or 2'
    # the following enables the fine mixer mode (coarse is not supported at the moment)
    _daq.setInt(f"/{_device_name}/RAW/DEBUG/{DEBUG_REG_REG_COMMENSURATE_TEST}/VALUE", mode)

def output_qa_sequencer_en_get():
    addr = DEBUG_REG_REG_QA_SG_CTRL
    value = _daq.getInt(f"/{_device_name}/RAW/DEBUG/{addr}/VALUE")
    return True if value & 0x02 != 0 else False

def output_qa_sequencer_en_set(en: bool):
    addr = DEBUG_REG_REG_QA_SG_CTRL
    current_value = _daq.getInt(f"/{_device_name}/RAW/DEBUG/{addr}/VALUE")
    if en:
        current_value |= 0x02
    else:
        current_value &= ~0x02
    _daq.setInt(f"/{_device_name}/RAW/DEBUG/{addr}/VALUE", current_value)

def output_qa_sequencer_triggered_mode_get():
    addr = DEBUG_REG_REG_QA_SG_CTRL
    value = _daq.getInt(f"/{_device_name}/RAW/DEBUG/{addr}/VALUE")
    return True if value & 0x04 != 0 else False

def output_qa_sequencer_triggered_mode_set(en: bool):
    addr = DEBUG_REG_REG_QA_SG_CTRL
    current_value = _daq.getInt(f"/{_device_name}/RAW/DEBUG/{addr}/VALUE")
    if en:
        current_value |= 0x04
    else:
        current_value &= ~0x04
    _daq.setInt(f"/{_device_name}/RAW/DEBUG/{addr}/VALUE", current_value)

def output_qa_sequencer_parallel_mode_get():
    addr = DEBUG_REG_REG_QA_SG_CTRL
    value = _daq.getInt(f"/{_device_name}/RAW/DEBUG/{addr}/VALUE")
    return True if value & 0x08 != 0 else False

def output_qa_sequencer_parallel_mode_set(en: bool):
    addr = DEBUG_REG_REG_QA_SG_CTRL
    current_value = _daq.getInt(f"/{_device_name}/RAW/DEBUG/{addr}/VALUE")
    if en:
        current_value |= 0x08
    else:
        current_value &= ~0x08
    _daq.setInt(f"/{_device_name}/RAW/DEBUG/{addr}/VALUE", current_value)

def output_qa_sequencer_wave_length_set(length):
    addr = DEBUG_REG_REG_SEQENCER_LENGTH
    _daq.setInt(f"/{_device_name}/RAW/DEBUG/{addr}/VALUE", length)

def output_qa_sequencer_wave_length_get():
    addr = DEBUG_REG_REG_SEQENCER_LENGTH
    value = _daq.getInt(f"/{_device_name}/RAW/DEBUG/{addr}/VALUE")
    return value
