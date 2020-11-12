import logging
log = logging.getLogger(__name__)

import numpy as np
import sys
from pycqed.analysis_v3 import helper_functions as hlp_mod
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.measurement.waveform_control import segment as seg_mod

from pycqed.analysis_v3 import pipeline_analysis as pla
pla.search_modules.add(sys.modules[__name__])


def create_pulse(data_dict, keys_in, keys_out, **params):
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

    data_to_proc_dict = hlp_mod.get_data_to_process(data_dict, keys_in)

    mobjn = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['mobjn'], **params)

    pulse_params_dict = {}

    timestamp = hlp_mod.get_param('timestamp', data_dict, **params)
    if timestamp is not None:
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
        hlp_mod.add_param('timestamps', [timestamp], data_dict, **params)

    pulse_dict = hlp_mod.get_param('pulse_dict', data_dict, default_value={},
                                   **params)
    pulse_params_dict.update(pulse_dict)
    pulse_params_dict['element_name'] = 'element'

    pulse = seg_mod.UnresolvedPulse(pulse_params_dict).pulse_obj
    pulse.algorithm_time(0)

    tvals = hlp_mod.get_param('tvals', data_dict, **params)
    if tvals is None:
        tvals = np.arange(0, pulse.length, 1 / 2.4e9)

    hlp_mod.add_param('volt_freq_conv', pulse_params_dict['volt_freq_conv'],
                      data_dict, **params)
    hlp_mod.add_param('volt_freq_conv', pulse_params_dict['volt_freq_conv'],
                      data_dict, **params)

    volts_gen = pulse.chan_wf(pulse_params_dict['flux_channel'], tvals)
    volt_freq_conv = pulse_params_dict['volt_freq_conv']

    return tvals, volts_gen,
