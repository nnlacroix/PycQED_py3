import logging
log = logging.getLogger(__name__)

import numpy as np
import sys
from pycqed.analysis_v3 import helper_functions as hlp_mod
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.measurement.waveform_control import segment as seg_mod
from pycqed.analysis import fitting_models as fit_mods
from pycqed.measurement.waveform_control import fluxpulse_predistortion as fpdist

from pycqed.analysis_v3 import pipeline_analysis as pla
pla.search_modules.add(sys.modules[__name__])


def fd_create_pulse(data_dict, keys_in, keys_out, **params):
    """
    keys_out = ['tvals', 'volts']
    keys_out = ['volts', 'tvals']

    Thresholds the data in data_dict specified by keys_in about the
    threshold values in threshold_list (one for each keyi in keys_in).
    :param data_dict: OrderedDict containing data to be processed and where
                    processed data is to be stored
    :param keys_in: list of key names or dictionary keys paths in
                    data_dict for the data to be processed
    :param keys_out: list of key names or dictionary keys paths in
                    data_dict for the processed data to be saved into
    :param ro_thresholds: dict with keys meas_obj_names and values specifying
        the thresholds around which the data array for each meas_obj
        should be thresholded.
    :param params: keyword arguments.

    Assumptions:
        - this function must be used for one meas_obj only! Meaning:
            - keys_in and keys_out have length 1
            - meas_obj_names exists in either data_dict or params and has one
                entry
    """

    # data_to_proc_dict = hlp_mod.get_data_to_process(data_dict, keys_in)

    mobjn = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['mobjn'], **params)

    pulse_params_dict = {}

    timestamps = hlp_mod.get_param('timestamps', data_dict, **params)
    if timestamps is not None:
        timestamp = timestamps[0]
        # Flux pulse parameters
        # Needs to be changed when support for other pulses is added.
        params_dict = {
            'pulse_type': f'Instrument settings.{mobjn}.flux_pulse_type',
            'channel': f'Instrument settings.{mobjn}.flux_pulse_channel',
            'aux_channels_dict': f'Instrument settings.{mobjn}.'
                                 f'flux_pulse_aux_channels_dict',
            'amplitude': f'Instrument settings.{mobjn}.flux_pulse_amplitude',
            'frequency': f'Instrument settings.{mobjn}.flux_pulse_frequency',
            'phase': f'Instrument settings.{mobjn}.flux_pulse_phase',
            'pulse_length': f'Instrument settings.{mobjn}.'
                            f'flux_pulse_pulse_length',
            'truncation_length': f'Instrument settings.{mobjn}.'
                                 f'flux_pulse_truncation_length',
            'buffer_length_start': f'Instrument settings.{mobjn}.'
                                   f'flux_pulse_buffer_length_start',
            'buffer_length_end': f'Instrument settings.{mobjn}.'
                                 f'flux_pulse_buffer_length_end',
            'extra_buffer_aux_pulse': f'Instrument settings.{mobjn}.'
                                      f'flux_pulse_extra_buffer_aux_pulse',
            'pulse_delay': f'Instrument settings.{mobjn}.'
                           f'flux_pulse_pulse_delay',
            'basis_rotation': f'Instrument settings.{mobjn}.'
                              f'flux_pulse_basis_rotation',
            'gaussian_filter_sigma': f'Instrument settings.{mobjn}.'
                                     f'flux_pulse_gaussian_filter_sigma',
            'volt_freq_conv': f'Instrument settings.{mobjn}.'
                              f'fit_ge_freq_from_flux_pulse_amp',
            'flux_channel': f'Instrument settings.{mobjn}.'
                            f'flux_pulse_channel',
        }
        hlp_mod.get_params_from_hdf_file(pulse_params_dict, params_dict,
                                         folder=a_tools.get_folder(timestamp),
                                         **params)

    pulse_dict = hlp_mod.get_param('pulse_dict', data_dict, default_value={},
                                   **params)
    pulse_params_dict.update(pulse_dict)
    pulse_params_dict['element_name'] = 'element'

    pulse = seg_mod.UnresolvedPulse(pulse_params_dict).pulse_obj
    pulse.algorithm_time(0)

    tvals = hlp_mod.get_param(keys_in[0], data_dict, **params)
    if tvals is None:
        tvals = np.arange(0, pulse.length, 1 / 2.4e9)

    if 'volt_freq_conv' not in data_dict:
        hlp_mod.add_param('volt_freq_conv',
                          pulse_params_dict['volt_freq_conv'], data_dict,
                          **params)

    hlp_mod.add_param(keys_out[0],
                      [tvals, pulse.chan_wf(
                          pulse_params_dict['flux_channel'], tvals),],
                      data_dict, **params)

def fd_load_qb_params(data_dict, params_dict, timestamp=None, **params):
    mobjn = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['mobjn'], **params)
    if timestamp is None:
        timestamp = hlp_mod.get_param('timestamps', data_dict, **params)[0]
    pulse_params_dict = {}
    params_dict = {k: f'Instrument settings.{mobjn}.{v}'
                   for k, v in params_dict.items()}
    hlp_mod.get_params_from_hdf_file(pulse_params_dict, params_dict,
                                     folder=a_tools.get_folder(timestamp),
                                     **params)
    for k in params_dict.keys():
        hlp_mod.add_param(k, pulse_params_dict[k], data_dict,
                          replace_value=True, **params)

def fd_load_distortion_dict(data_dict, timestamp=None, **params):
    mobjn = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['mobjn'], **params)
    if timestamp is None:
        timestamp = hlp_mod.get_param('timestamps', data_dict, **params)[0]
    folder = a_tools.get_folder(timestamp)
    params_dict = {}
    hlp_mod.get_params_from_hdf_file(
        params_dict,
        {'ch': f'Instrument settings.{mobjn}.flux_pulse_channel'},
        folder=a_tools.get_folder(timestamp), **params)
    hlp_mod.get_params_from_hdf_file(
        params_dict,
        {'distortion': f"Instrument settings.Pulsar."
                       f"{params_dict['ch']}_distortion",
         'distortion_dict': f"Instrument settings.Pulsar."
                            f"{params_dict['ch']}_distortion_dict",
        },
        folder=a_tools.get_folder(timestamp), **params)
    print(params_dict)
    if params_dict['distortion'] == 'off':
        params_dict['distortion_dict'] = {}
    elif isinstance(params_dict['distortion_dict'], str):
        params_dict['distortion_dict'] = eval(params_dict['distortion_dict'])
    hlp_mod.add_param('distortion_dict', params_dict['distortion_dict'],
                      data_dict, replace_value=True, **params)

def fd_volt_to_freq(data_dict, keys_in, keys_out, **params):
    volt_freq_conv = hlp_mod.get_param('volt_freq_conv', data_dict, **params)
    s = hlp_mod.get_param(keys_in[0], data_dict, **params)
    hlp_mod.add_param(keys_out[0],
                      [s[0], fit_mods.Qubit_dac_to_freq(
                          s[1], **volt_freq_conv)],
                      data_dict, **params)

def fd_freq_to_volt(data_dict, keys_in, keys_out, **params):
    volt_freq_conv = hlp_mod.get_param('volt_freq_conv', data_dict, **params)
    s = hlp_mod.get_param(keys_in[0], data_dict, **params)
    hlp_mod.add_param(
        keys_out[0],
        [s[0], np.mod(fit_mods.Qubit_freq_to_dac(s[1], **volt_freq_conv,
                                                 branch='negative'),
                      volt_freq_conv['V_per_phi0'])],
        data_dict, **params)

def fd_apply_predistortion(data_dict, keys_in, keys_out, **params):
    def my_resample(tvals_gen_new, tvals_gen, freqs_meas):
        return np.interp(tvals_gen_new, tvals_gen, freqs_meas)

    if 'distortion_dict' not in data_dict:
        fd_load_distortion_dict(data_dict)
    distortion_dict = hlp_mod.get_param('distortion_dict', data_dict, **params)
    dt = hlp_mod.get_param('dt', data_dict, **params)
    for ki, ko in zip(keys_in, keys_out):
        tvals, wf = hlp_mod.get_param(ki, data_dict, **params)

        tvals_rs = np.arange(tvals[0], tvals[-1], dt)
        wf_rs = my_resample(tvals_rs, tvals, wf)

        fir_kernels = distortion_dict.get('FIR', None)
        if fir_kernels is not None:
            if hasattr(fir_kernels, '__iter__') and not \
                    hasattr(fir_kernels[0], '__iter__'):  # 1 kernel
                wf_rs = fpdist.filter_fir(fir_kernels, wf_rs)
            else:
                for kernel in fir_kernels:
                    wf_rs = fpdist.filter_fir(kernel, wf_rs)

        iir_filters = distortion_dict.get('IIR', None)
        if iir_filters is not None:
            wf_rs = fpdist.filter_iir(iir_filters[0], iir_filters[1], wf_rs)

        wf = my_resample(tvals, tvals_rs, wf_rs)
        hlp_mod.add_param(ko, [tvals, wf], data_dict, **params)
