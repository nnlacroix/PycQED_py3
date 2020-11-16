import logging
log = logging.getLogger(__name__)

import numpy as np
import scipy.optimize as optimize
import sys
from copy import deepcopy
from pycqed.analysis_v3 import helper_functions as hlp_mod
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.measurement.waveform_control import segment as seg_mod
from pycqed.analysis import fitting_models as fit_mods
from pycqed.analysis_v2 import timedomain_analysis as tda
from pycqed.measurement.waveform_control import fluxpulse_predistortion as fpdist
from pycqed.measurement import sweep_points as sp_mod

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

    delay = hlp_mod.get_param('delay', data_dict, default_value=0, **params)
    tvals = hlp_mod.get_param(keys_in[0], data_dict, **params)
    if tvals is None:
        dt = hlp_mod.get_param('dt', data_dict, raise_error=True, **params)
        tvals = np.arange(0, pulse.length, dt) + delay

    if 'volt_freq_conv' not in data_dict:
        hlp_mod.add_param('volt_freq_conv',
                          pulse_params_dict['volt_freq_conv'], data_dict,
                          **params)

    hlp_mod.add_param(keys_out[0],
                      [tvals, pulse.chan_wf(
                          pulse_params_dict['flux_channel'], tvals - delay),],
                      data_dict, **params)

def fd_load_qb_params(data_dict, params_dict, timestamp=None, **params):
    mobjn = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['mobjn'], **params)
    if timestamp is None:
        timestamp = hlp_mod.get_param('timestamps', data_dict, **params)[0]
    params_dict = {k: f'Instrument settings.{mobjn}.{v}'
                   for k, v in params_dict.items()}
    hlp_mod.get_params_from_hdf_file(data_dict, params_dict,
                                     folder=a_tools.get_folder(timestamp),
                                     replace_value=True, **params)

def fd_load_distortion_dict(data_dict, timestamp=None,
                            key_distortion='distortion_dict', **params):
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
    if params_dict['distortion'] == 'off':
        params_dict['distortion_dict'] = {}
    elif isinstance(params_dict['distortion_dict'], str):
        params_dict['distortion_dict'] = eval(params_dict['distortion_dict'])
    hlp_mod.add_param(key_distortion, params_dict['distortion_dict'],
                      data_dict, replace_value=True, **params)

def fd_IIR_from_distortion_dict(data_dict, keys_in, keys_out, **params):
    distortion_dict = hlp_mod.get_param(keys_in[0], data_dict, **params)
    if len(keys_out) == 1 and len(distortion_dict['IIR'][0]) > 1:
        keys_out = [f"{keys_out[0]}_{i}"
                    for i in range(len(distortion_dict['IIR'][0]))]
    for a, b, ko in zip(distortion_dict['IIR'][0], distortion_dict['IIR'][1],
                        keys_out):
        hlp_mod.add_param(ko, [a, b], data_dict, **params)

def fd_FIR_from_distortion_dict(data_dict, keys_in, keys_out, **params):
    for ki, ko in zip(keys_in, keys_out):
        distortion_dict = hlp_mod.get_param(keys_in[0], data_dict, **params)
        hlp_mod.add_param(ko, distortion_dict['FIR'], data_dict, **params)

def fd_volt_to_freq(data_dict, keys_in, keys_out, **params):
    volt_freq_conv = hlp_mod.get_param('volt_freq_conv', data_dict, **params)
    s = hlp_mod.get_param(keys_in[0], data_dict, **params)
    hlp_mod.add_param(keys_out[0],
                      [s[0], fit_mods.Qubit_dac_to_freq(
                          s[1], **volt_freq_conv)],
                      data_dict, **params)

def fd_freq_to_volt(data_dict, keys_in, keys_out, method='fit_mods', **params):
    s = hlp_mod.get_param(keys_in[0], data_dict, **params)
    if method == 'fit_mods':
        volt_freq_conv = hlp_mod.get_param('volt_freq_conv', data_dict, **params)
        hlp_mod.add_param(
            keys_out[0],
            [s[0], np.mod(fit_mods.Qubit_freq_to_dac(s[1], **volt_freq_conv,
                                                     branch='negative'),
                          volt_freq_conv['V_per_phi0'])],
            data_dict, **params)
    elif method == 'optimize':
        d = hlp_mod.get_param('asymmetry', data_dict, raise_error=True,
                              **params)
        from_lower = hlp_mod.get_param('from_lower_sweet_spot', data_dict,
                                       default_value=False, **params)
        if from_lower:
            d = 1 / d
        f_sweet_spot = hlp_mod.get_param('ge_freq', data_dict,
                                         raise_error=True, **params)
        pulse_amp = hlp_mod.get_param('flux_pulse_amplitude', data_dict,
                                      raise_error=True, **params)
        phi_0 = hlp_mod.get_param('phi_0', data_dict, raise_error=True,
                                  **params)
        dphidV = hlp_mod.get_param('dphidV', data_dict, raise_error=True,
                                   **params)

        func_for_V = lambda voltages: s[1] - f_sweet_spot * (
                    (np.cos(-dphidV * voltages + phi_0)) ** 2 +
                    (d * np.sin(-dphidV * voltages + phi_0)) ** 2) ** (0.25)
        voltages = optimize.fsolve(func_for_V, pulse_amp * np.ones(len(s[1])))
        hlp_mod.add_param(keys_out[0], [s[0], voltages], data_dict, **params)
    else:
        raise NotImplementedError(f"fd_freq_to_volt: method {method} not "
                                  f"implemented.")

def fd_extract_volt_freq_conv(data_dict, keys_in=None, **params):
    if keys_in is None:
        keys_in = ['volt_freq_conv']
    vfc = hlp_mod.get_param(keys_in[0], data_dict, **params)
    f_sweet_spot = hlp_mod.get_param('ge_freq', data_dict, raise_error=True,
                                     **params)
    from_lower = (np.abs(f_sweet_spot - vfc['f_max']) / f_sweet_spot > 0.01)
    hlp_mod.add_param('from_lower_sweet_spot', from_lower, data_dict,
                      replace_value=True, **params)
    hlp_mod.add_param('asymmetry', vfc['asymmetry'], data_dict,
                      replace_value=True, **params)
    dphidV = -np.pi / vfc['V_per_phi0']
    hlp_mod.add_param('dphidV', dphidV, data_dict,
                      replace_value=True, **params)
    # TODO: check whether the following line is correct
    phi_0 = -dphidV * (vfc['dac_sweet_spot'] + (vfc['V_per_phi0'] / 2 if
                                                from_lower else 0))
    hlp_mod.add_param('phi_0', phi_0, data_dict,
                      replace_value=True, **params)

def fd_create_volt_freq_conv(data_dict, keys_out=None, **params):
    if keys_out is None:
        keys_out = ['volt_freq_conv']
    d = hlp_mod.get_param('asymmetry', data_dict, raise_error=True, **params)
    f_sweet_spot = hlp_mod.get_param('ge_freq', data_dict, raise_error=True,
                                     **params)
    phi_0 = hlp_mod.get_param('phi_0', data_dict, raise_error=True, **params)
    dphidV = hlp_mod.get_param('dphidV', data_dict, raise_error=True, **params)
    from_lower = hlp_mod.get_param('from_lower_sweet_spot', data_dict,
                                   raise_error=True, **params)
    vfc = {'asymmetry': d,
           'V_per_phi0': -np.pi / dphidV,
           # TODO: check whether the following line is correct
           'dac_sweet_spot': -phi_0 / dphidV
          }
    if from_lower:
        vfc['dac_sweet_spot'] -= vfc['V_per_phi0'] / 2
        vfc['f_max'] = f_sweet_spot * (
                (np.cos(np.pi / 2)) ** 2 + ((1 / d) * np.sin(np.pi / 2)) ** 2
            ) ** (0.25)  # rough approximation
    else:
        vfc['f_max'] = f_sweet_spot
    hlp_mod.add_param(keys_out[0], vfc, data_dict, replace_value=True,
                      **params)

def fd_resample(tvals_new, tvals, wf, method='interp', **params):
    if method == 'interp':
        return np.interp(tvals_new, tvals, wf)
    else:
        raise NotImplementedError(f"Resampling method {method} not "
                                  f"implemented.")

def fd_apply_distortion_dict(data_dict, keys_in, keys_out, **params):
    resampling_method = hlp_mod.get_param('resampling_method', data_dict,
                                          default_value='interp', **params)
    dt = hlp_mod.get_param('dt', data_dict, **params)
    key_distortion = 'distortion_dict'
    distortion_dict = hlp_mod.get_param(key_distortion, data_dict, **params)
    if isinstance(distortion_dict, str):  # a key was passed instead of a dict
        distortion_dict = hlp_mod.get_param(distortion_dict, data_dict,
                                            raise_error=True, **params)
    elif distortion_dict is None:
        fd_load_distortion_dict(data_dict, key_distortion=key_distortion)
        distortion_dict = hlp_mod.get_param(key_distortion, data_dict,
                                            **params)

    for ki, ko in zip(keys_in, keys_out):
        tvals, wf = hlp_mod.get_param(ki, data_dict, **params)
        resampling = not np.all(np.abs(np.diff(tvals) - dt) < 1e-14)
        if resampling:
            tvals_rs = np.arange(tvals[0], tvals[-1], dt)
            wf = fd_resample(tvals_rs, tvals, wf, resampling_method)

        fir_kernels = distortion_dict.get('FIR', None)
        if fir_kernels is not None:
            if hasattr(fir_kernels, '__iter__') and not \
                    hasattr(fir_kernels[0], '__iter__'):  # 1 kernel
                wf = fpdist.filter_fir(fir_kernels, wf)
            else:
                for kernel in fir_kernels:
                    wf = fpdist.filter_fir(kernel, wf)

        iir_filters = distortion_dict.get('IIR', None)
        if iir_filters is not None:
            wf = fpdist.filter_iir(iir_filters[0], iir_filters[1], wf)

        if resampling:
            wf = fd_resample(tvals, tvals_rs, wf, resampling_method)
        hlp_mod.add_param(ko, [tvals, wf], data_dict, **params)

def fd_apply_distortion(data_dict, keys_in, keys_out, keys_filter, filter_type,
                        **params):
    if len(keys_filter) == 1 and len(keys_in) > 1:
        keys_filter = keys_filter * len(keys_in)
    if len(keys_in) == 1 and len(keys_filter) > 1:
        keys_in = keys_in * len(keys_filter)
    for ki, ko, kk in zip(keys_in, keys_out, keys_filter):
        dist = hlp_mod.get_param(kk, data_dict, **params)
        if filter_type == 'IIR' and not hasattr(dist[0][0], '__iter__'):
            dist = [[d] for d in dist]
        fd_apply_distortion_dict(data_dict, [ki], [ko],
                                 distortion_dict={filter_type: dist}, **params)

def fd_expmod_to_IIR(data_dict, keys_in, keys_out, inverse_IIR=True, **params):
    dt = hlp_mod.get_param('dt', data_dict, **params)
    for ki, ko in zip(keys_in, keys_out):
        A, B, tau = hlp_mod.get_param(ki, data_dict, **params)
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
        hlp_mod.add_param(ko, [a, b], data_dict, **params)

def fd_IIR_to_expmod(data_dict, keys_in, keys_out, inverse_IIR=True, **params):
    dt = hlp_mod.get_param('dt', data_dict, **params)
    for ki, ko in zip(keys_in, keys_out):
        a, b = hlp_mod.get_param(ki, data_dict, **params)
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
        hlp_mod.add_param(ko, [A, B, tau], data_dict, **params)

def fd_invert_IIR(data_dict, keys_in, keys_out, **params):
    for ki, ko in zip(keys_in, keys_out):
        dist = hlp_mod.get_param(ki, data_dict, **params)
        if hasattr(dist[0][0], '__iter__'):
            # multiple IIR filters
            dist = [[b / b[0] for a, b in zip(dist[0], dist[1])],
                    [a / b[0] for a, b in zip(dist[0], dist[1])]]
        else:
            a, b = dist
            dist = [b / b[0], a / b[0]]
        hlp_mod.add_param(ko, dist, data_dict, **params)

def fd_scale_and_negate_IIR(data_dict, keys_in, keys_out, scale=1, **params):
    for ki, ko in zip(keys_in, keys_out):
        dist = deepcopy(hlp_mod.get_param(ki, data_dict, **params))
        if isinstance(dist, dict):
            fpdist.scale_and_negate_IIR(dist['IIR'], scale)
        elif hasattr(dist[0][0], '__iter__'):
            # multiple IIR filters as expected by the function in fpdist
            fpdist.scale_and_negate_IIR(dist, scale)
        else:
            dist = [np.array([dist[0]]), np.array([dist[1]])]
            fpdist.scale_and_negate_IIR(dist, scale)
            dist = [dist[0][0], dist[1][0]]
        hlp_mod.add_param(ko, dist, data_dict, **params)

def fd_identity(data_dict, keys_in, keys_out, **params):
    for ki, ko in zip(keys_in, keys_out):
        hlp_mod.add_param(ko, hlp_mod.get_param(ki, data_dict, **params),
                          data_dict, **params)

def fd_transform(data_dict, keys_in, keys_out, **params):
    key_transform = 'transform'
    transform = hlp_mod.get_param(key_transform, data_dict,
                                  raise_error=True, **params)
    if isinstance(transform, str):  # a key was passed instead of a dict
        transform = hlp_mod.get_param(transform, data_dict,
                                      raise_error=True, **params)
    for ki, ko in zip(keys_in, keys_out):
        data = np.asarray(hlp_mod.get_param(ki, data_dict, **params))
        if data.ndim == 2:
            # interpret as time series
            data[1:] = transform(data[1:])
        else:
            data = transform(data)
        hlp_mod.add_param(ko, data, data_dict, **params)

def fd_comparison(data_dict, keys_in, keys_out, method='difference', **params):
    def cmp(a, b, method):
        if method == 'difference':
            return a - b
        elif method == 'ratio':
            return a / b
        else:
            raise NotImplementedError(f'Comparison method {method} not '
                                      f'implemented.')

    data = hlp_mod.get_param(keys_in[0], data_dict, **params)
    data2 = hlp_mod.get_param(keys_in[1], data_dict, **params)
    if np.asarray(data).ndim == 2:
        # interpret as time series
        [tvals, wf], [tvals2, wf2] = data, data2
        try:
            np.testing.assert_equal(tvals, tvals2)
        except AssertionError:
            wf2 = fd_resample(tvals, tvals2, wf2, **params)
        hlp_mod.add_param(keys_out[0], [tvals, cmp(wf, wf2, method)],
                          data_dict,  **params)
    else:
        hlp_mod.add_param(keys_out[0],
                          cmp(np.asarray(data), np.asarray(data2), method),
                          data_dict, **params)

def fd_difference(data_dict, keys_in, keys_out, **params):
    fd_comparison(data_dict, keys_in, keys_out, method='difference', **params)

def fd_ratio(data_dict, keys_in, keys_out, **params):
    fd_comparison(data_dict, keys_in, keys_out, method='ratio', **params)

def fd_estimate_f_park(data_dict, keys_in, **params):
    _, fitted_freqs = hlp_mod.get_param(keys_in[0], data_dict, **params)
    hlp_mod.add_param('f_park', fitted_freqs[-1], data_dict, **params)

def fd_estimate_f_pulsed(data_dict, keys_in, **params):
    """
    Heuristic to determine f_pulsed used in Pierre-Antoine's class.
    """
    _, fitted_freqs = hlp_mod.get_param(keys_in[0], data_dict, **params)
    f_park = hlp_mod.get_param('f_park', data_dict, raise_error=True, **params)
    # cut away calpoints and falling edge ("7" is hard-coded in
    # Pierre-Antoine's class)
    freqs = fitted_freqs[:-7]
    f_pulsed = freqs[np.argmin(freqs)]
    from_lower = False
    if (abs(f_park - f_pulsed) / f_pulsed) < 0.01:
        f_pulsed = freqs[np.argmax(freqs)]
        from_lower = True
    hlp_mod.add_param('f_pulsed', f_pulsed, data_dict, **params)
    hlp_mod.add_param('from_lower_sweet_spot', from_lower, data_dict,
                      **params)

def fd_estimate_phi0(data_dict, **params):
    """
    Estimation of phi0 used in Pierre-Antoine's class.
    """
    d = hlp_mod.get_param('asymmetry', data_dict, raise_error=True, **params)
    f_park = hlp_mod.get_param('f_park', data_dict, raise_error=True, **params)
    f_sweet_spot = hlp_mod.get_param('ge_freq', data_dict, raise_error=True,
                                     **params)
    guess_phi_0 = hlp_mod.get_param('guess_phi_0', data_dict,
                                    default_value=0.001, **params)
    from_lower = hlp_mod.get_param('from_lower_sweet_spot', data_dict,
                                   default_value=False, **params)
    if from_lower:
        d = 1 / d
    func_for_phi_0 = lambda phi_0: f_park - f_sweet_spot * (
        (np.cos(phi_0)) ** 2 + (d * np.sin(phi_0)) ** 2) ** (0.25)
    phi_0 = optimize.fsolve(func_for_phi_0, guess_phi_0)[0]
    hlp_mod.add_param('phi_0', phi_0, data_dict, **params)

def fd_estimate_dphidV(data_dict, **params):
    """
    Estimation of dphidV used in Pierre-Antoine's class.
    """
    d = hlp_mod.get_param('asymmetry', data_dict, raise_error=True, **params)
    f_pulsed = hlp_mod.get_param('f_pulsed', data_dict, raise_error=True,
                                 **params)
    f_sweet_spot = hlp_mod.get_param('ge_freq', data_dict, raise_error=True,
                                     **params)
    pulse_amp = hlp_mod.get_param('flux_pulse_amplitude', data_dict,
                                  raise_error=True, **params)
    phi_0 = hlp_mod.get_param('phi_0', data_dict, raise_error=True, **params)
    guess_dphidV = hlp_mod.get_param('guess_dphidV', data_dict,
                                     raise_error=True, **params)
    from_lower = hlp_mod.get_param('from_lower_sweet_spot', data_dict,
                                   default_value=False, **params)
    if from_lower:
        d = 1 / d
    func_for_dphidV = lambda dphidV: f_pulsed - f_sweet_spot * (
                (np.cos(-dphidV * pulse_amp + phi_0)) ** 2 +
                (d * np.sin(-dphidV * pulse_amp + phi_0)) ** 2) ** (0.25)
    dphidV = optimize.fsolve(func_for_dphidV, guess_dphidV)[0]
    if np.abs(dphidV - guess_dphidV) > 0.1:
        log.warning(f"dphidV is significantly different from typical value: "
                    f"dphidV={dphidV}, where typical value is {guess_dphidV}")
    hlp_mod.add_param('dphidV', dphidV, data_dict, **params)

def fd_import_fp_scope(data_dict, keys_out, keys_err=None, **params):
    """
    Runs a flux pulse scope analysis and provides the fitted frequencies as
    a time series in keys_out.
    """
    mobjn = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['mobjn'], **params)
    extract_only = not params.get('plot', False)
    ts = hlp_mod.get_param('timestamps', data_dict, **params)
    assert len(ts) == len(keys_out), 'The number of keys_out and the number ' \
                                     'of timestamps do not match.'
    if keys_err is None:
        keys_err = [None] * len(ts)
    assert len(ts) == len(keys_err), 'The number of keys_err and the number ' \
                                     'of timestamps do not match.'
    fp_ana = []
    for timestamp, ko, ke in zip(ts, keys_out, keys_err):
        options_dict = {'TwoD': True, 'rotation_type': 'global_PCA',
                        'save_figs': False}
        qb_names = [a_tools.get_folder(timestamp)[
                    a_tools.get_folder(timestamp).index('q'):]]
        metadata = hlp_mod.get_param_from_metadata_group(timestamp)
        sp = sp_mod.SweepPoints('delay',
                                metadata['sweep_points_dict'][qb_names[0]],
                                's', 'delay, $t$', dimension=0)
        sp.add_sweep_parameter('freq',
                               metadata['sweep_points_dict_2D'][qb_names[0]],
                               'Hz', 'drive frequency, $f_d$', dimension=1)
        mospm = sp.get_meas_obj_sweep_points_map(qb_names)
        options_dict['meas_obj_sweep_points_map'] = mospm
        options_dict['sweep_points'] = sp
        options_dict.update(params)
        fp_ana.append(tda.FluxPulseScopeAnalysis(
            t_start=timestamp, qb_names=[mobjn], extract_only=extract_only,
            options_dict=options_dict, raise_exceptions=True))
        delays = fp_ana[-1].proc_data_dict['sweep_points_dict'][mobjn][
            'sweep_points']
        res = fp_ana[-1].proc_data_dict['analysis_params_dict'][
            f'fitted_freqs_{mobjn}']
        hlp_mod.add_param(ko, [delays, res['val']], data_dict, **params)
        if ke is not None:
            hlp_mod.add_param(ke, res['stderr'], data_dict, **params)

