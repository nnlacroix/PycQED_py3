import logging
log = logging.getLogger(__name__)

import os
import numpy as np
import scipy.optimize as optimize
from copy import deepcopy
from pycqed.analysis_v3 import helper_functions as hlp_mod
from pycqed.analysis_v3 import processing_pipeline as pp_mod
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.measurement.waveform_control import segment as seg_mod
from pycqed.analysis import fitting_models as fit_mods
from pycqed.analysis_v2 import timedomain_analysis as tda
from pycqed.measurement.waveform_control import fluxpulse_predistortion as fpdist
from pycqed.measurement import sweep_points as sp_mod
import pycqedscripts.scripts.predistortion_filters.IIR_fitting as IIR_fitting

import sys
pp_mod.search_modules.add(sys.modules[__name__])


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
    if tvals.ndim == 2:  # interpret as time series
        tvals = tvals[0]
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

def fd_IIR_from_distortion_dict(data_dict, keys_in, keys_out, separate=False,
                                **params):
    def get_dist_dict(ki):
        distortion_dict = deepcopy(hlp_mod.get_param(ki, data_dict, **params))
        if 'IIR' not in distortion_dict:
            distortion_dict['IIR'] = [[], []]
        return distortion_dict

    if separate:
        distortion_dict = get_dist_dict(keys_in[0])
        if len(keys_out) == 1 and len(distortion_dict['IIR'][0]) > 1:
            keys_out = [f"{keys_out[0]}_{i}"
                        for i in range(len(distortion_dict['IIR'][0]))]
        for a, b, ko in zip(distortion_dict['IIR'][0], distortion_dict['IIR'][1],
                            keys_out):
            hlp_mod.add_param(ko, [a, b], data_dict, **params)
    else:
        for ki, ko in zip(keys_in, keys_out):
            distortion_dict = get_dist_dict(ki)
            hlp_mod.add_param(ko, distortion_dict['IIR'], data_dict, **params)

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
    else:
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
        E_c = hlp_mod.get_param('E_c', data_dict, default_value=0, **params)
        if method == 'optimize':
            phi_0 = hlp_mod.get_param('phi_0', data_dict, raise_error=True,
                                      **params)
            dphidV = hlp_mod.get_param('dphidV', data_dict, raise_error=True,
                                       **params)
            # print(s[1].shape, f_sweet_spot, dphidV, phi_0, np.ones(len(s[1])).shape)
            func_for_V = lambda voltages: s[1] - (
                    -E_c + (f_sweet_spot + E_c) *
                    ((np.cos(-dphidV * voltages + phi_0)) ** 2 +
                     (d * np.sin(-dphidV * voltages + phi_0)) ** 2) ** (0.25))
            voltages = optimize.fsolve(
                func_for_V, pulse_amp * np.ones(len(s[1])))
            hlp_mod.add_param(
                keys_out[0], [s[0], voltages], data_dict, **params)
        elif method == 'analytic':
            f_other_ss = -E_c + (f_sweet_spot + E_c) * d ** 0.5
            if from_lower:
                f = np.minimum(np.maximum(s[1], f_sweet_spot), f_other_ss)
            else:
                f = np.maximum(np.minimum(s[1], f_sweet_spot), f_other_ss)
            y = np.arccos(np.sqrt(
                (((f + E_c) / (f_sweet_spot + E_c)) ** 4 - d ** 2)
                / (1 - d ** 2)))
            if from_lower:
                y += np.pi / 2
            hlp_mod.add_param(
                keys_out[0], [s[0], y], data_dict, **params)
        else:
            raise NotImplementedError(f"fd_freq_to_volt: method {method} not "
                                      f"implemented.")

def fd_extract_volt_freq_conv(data_dict, keys_in=None, **params):
    if keys_in is None:
        keys_in = ['volt_freq_conv']
    vfc = hlp_mod.get_param(keys_in[0], data_dict, **params)
    f_sweet_spot = hlp_mod.get_param('ge_freq', data_dict, raise_error=True,
                                     **params)
    from_lower = ((vfc['f_max'] - f_sweet_spot) / f_sweet_spot > 0.1)
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
            ) ** (0.25)  # TODO: rough approximation
    else:
        vfc['f_max'] = f_sweet_spot
    hlp_mod.add_param(keys_out[0], vfc, data_dict, replace_value=True,
                      **params)

def fd_resample(data_dict, keys_in, keys_tvals, keys_out, **params):
    method = hlp_mod.get_param('method', data_dict, default_value='interp',
                               **params)
    dt = hlp_mod.get_param('dt', data_dict, **params)
    for ki, kt, ko in zip(keys_in, keys_tvals, keys_out):
        tvals, wf = hlp_mod.get_param(ki, data_dict, **params)
        tvals_new = np.array(hlp_mod.get_param(kt, data_dict, **params))
        if tvals_new.ndim == 2:  # interpret as time series
            tvals_new = tvals_new[0]
        wf = fd_do_resample(tvals_new, tvals, wf, method=method)
        hlp_mod.add_param(ko, [tvals_new, wf], data_dict, **params)

def fd_do_resample(tvals_new, tvals, wf, method='interp', **params):
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
            wf = fd_do_resample(tvals_rs, tvals, wf, resampling_method)

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
            wf = fd_do_resample(tvals, tvals_rs, wf, resampling_method)
        hlp_mod.add_param(ko, [tvals, wf], data_dict, **params)

def fd_apply_distortion(data_dict, keys_in, keys_out, keys_filter, filter_type,
                        **params):
    if len(keys_filter) == 1 and len(keys_in) > 1:
        keys_filter = keys_filter * len(keys_in)
    if len(keys_in) == 1 and len(keys_filter) > 1:
        keys_in = keys_in * len(keys_filter)
    for ki, ko, kf in zip(keys_in, keys_out, keys_filter):
        dist = hlp_mod.get_param(kf, data_dict, **params)
        if filter_type == 'expmod':
            tvals, wf = hlp_mod.get_param(ki, data_dict, **params)
            tmpdict = {'expmod': dist, 'dt': np.diff(tvals)[0]}
            fd_expmod_to_IIR(tmpdict, ['expmod'], ['iir'], **params)
            wf = fpdist.filter_iir([tmpdict['iir'][0]], [tmpdict['iir'][1]],
                                   wf)
            hlp_mod.add_param(ko, [tvals, wf], data_dict, **params)
        if filter_type == 'IIR' and not hasattr(dist[0][0], '__iter__'):
            dist = [[d] for d in dist]
            fd_apply_distortion_dict(data_dict, [ki], [ko],
                                     distortion_dict={filter_type: dist},
                                     **params)

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
        a_, b_ = hlp_mod.get_param(ki, data_dict, **params)
        single_iir = not hasattr(a_[0], '__iter__')
        if single_iir:
            a_, b_ = [a_], [b_]
        expmods = []
        for a, b in zip(a_, b_):
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
            expmods.append([A, B, tau])
        hlp_mod.add_param(ko, expmods[0] if single_iir else expmods,
                          data_dict, **params)

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

def fd_transform(data_dict, keys_in, keys_out, apply_to='values', **params):
    key_transform = 'transform'
    transform = hlp_mod.get_param(key_transform, data_dict,
                                  raise_error=True, **params)
    if isinstance(transform, str):  # a key was passed instead of a function
        transform = hlp_mod.get_param(transform, data_dict,
                                      raise_error=True, **params)
    for ki, ko in zip(keys_in, keys_out):
        data = np.asarray(hlp_mod.get_param(ki, data_dict, **params))
        if data.ndim == 2:
            # interpret as time series
            d0, d1 = data
            if apply_to in ['values', 'both']:
                d1 = transform(d1)
            if apply_to in ['tvals', 'both']:
                d0 = transform(d0)
            data = np.array([d0, d1])
        else:
            data = transform(data)
        hlp_mod.add_param(ko, data, data_dict, **params)

def fd_compare(data_dict, keys_in, keys_out, method='difference', **params):
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
            wf2 = fd_do_resample(tvals, tvals2, wf2, **params)
        hlp_mod.add_param(keys_out[0], [tvals, cmp(wf, wf2, method)],
                          data_dict,  **params)
    else:
        hlp_mod.add_param(keys_out[0],
                          cmp(np.asarray(data), np.asarray(data2), method),
                          data_dict, **params)

def fd_difference(data_dict, keys_in, keys_out, **params):
    fd_compare(data_dict, keys_in, keys_out, method='difference', **params)

def fd_ratio(data_dict, keys_in, keys_out, **params):
    fd_compare(data_dict, keys_in, keys_out, method='ratio', **params)

def fd_estimate_f_park(data_dict, keys_in, **params):
    _, fitted_freqs = hlp_mod.get_param(keys_in[0], data_dict, **params)
    hlp_mod.add_param('f_park', fitted_freqs[0], data_dict, **params)
    # TODO rely on calib point? or on first point?

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
                      replace_value=True, **params)

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
    E_c = hlp_mod.get_param('E_c', data_dict, default_value=0, **params)
    from_lower = hlp_mod.get_param('from_lower_sweet_spot', data_dict,
                                   default_value=False, **params)
    if from_lower:
        d = 1 / d
    func_for_phi_0 = lambda phi_0: f_park - (-E_c + (f_sweet_spot + E_c) * (
        (np.cos(phi_0)) ** 2 + (d * np.sin(phi_0)) ** 2) ** (0.25))
    phi_0 = optimize.fsolve(func_for_phi_0, guess_phi_0)[0]
    hlp_mod.add_param('phi_0', phi_0, data_dict, replace_value=True, **params)

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
    E_c = hlp_mod.get_param('E_c', data_dict, default_value=0, **params)
    from_lower = hlp_mod.get_param('from_lower_sweet_spot', data_dict,
                                   default_value=False, **params)
    if from_lower:
        d = 1 / d
    func_for_dphidV = lambda dphidV: f_pulsed - (-E_c + (f_sweet_spot + E_c)
                                                 * (
                (np.cos(-dphidV * pulse_amp + phi_0)) ** 2 +
                (d * np.sin(-dphidV * pulse_amp + phi_0)) ** 2) ** (0.25))
    dphidV = optimize.fsolve(func_for_dphidV, guess_dphidV)[0]
    if np.abs(dphidV - guess_dphidV) > 0.1:
        log.warning(f"dphidV is significantly different from typical value: "
                    f"dphidV={dphidV}, where typical value is {guess_dphidV}")
    hlp_mod.add_param('dphidV', dphidV, data_dict, replace_value=True,
                      **params)

def fd_import_fp_scope(data_dict, keys_out, keys_err=None,
                       keys_projected=None, **params):
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
    if keys_projected is None:
        keys_projected = [None] * len(ts)
    # if keys_fit is None:
    #     keys_fit = [None] * len(ts)

    for timestamp, ko, ke, kp in zip(ts, keys_out, keys_err, keys_projected):
        options_dict = {'TwoD': True, 'rotation_type': 'global_PCA',
                        'save_figs': False}
        qb_names = [a_tools.get_folder(timestamp)[
                    a_tools.get_folder(timestamp).index('q'):]]
        metadata = hlp_mod.get_param_from_metadata_group(timestamp)
        if 'sweep_points' not in metadata:
            sp = sp_mod.SweepPoints(
                'delay', metadata['sweep_points_dict'][qb_names[0]], 's',
                'delay, $t$', dimension=0)
            sp.add_sweep_parameter(
                'freq', metadata['sweep_points_dict_2D'][qb_names[0]], 'Hz',
                'drive frequency, $f_d$', dimension=1)
            mospm = sp.get_meas_obj_sweep_points_map(qb_names)
            options_dict['meas_obj_sweep_points_map'] = mospm
            options_dict['sweep_points'] = sp
        od = params.pop('options_dict')
        options_dict.update(params)
        options_dict.update(od)
        for k in ['rotation_type', 'freq_ranges_remove',
                  'delay_ranges_remove', 'rectangles_exclude',
                  'sign_of_peaks']:
            try:
                v = hlp_mod.get_param(k, data_dict, raise_error=True, **params)
                options_dict.update({k: v})
            except (KeyError, ValueError):
                pass
        fp_ana = tda.FluxPulseScopeAnalysis(
            t_start=timestamp, qb_names=[mobjn], extract_only=extract_only,
            options_dict=options_dict, raise_exceptions=True)
        pdd = fp_ana.proc_data_dict
        delays = pdd['proc_sweep_points_dict'][mobjn]['sweep_points']
        res = fp_ana.proc_data_dict['analysis_params_dict'][
            f'fitted_freqs_{mobjn}']
        hlp_mod.add_param(ko, [delays, res['val']], data_dict, **params)
        hlp_mod.add_param(f'fpscope_analysis_{timestamp}', fp_ana, data_dict,
                          **params)
        if ke is not None:
            hlp_mod.add_param(ke, [delays, res['stderr']], data_dict, **params)
        if kp is not None:
            freqs = pdd['sweep_points_2D_dict'][mobjn]['freq']
            hlp_mod.add_param(
                kp, [[freqs, pdd['projected_data_dict'][mobjn]['pe'][:, i]]
                     for i in range(len(delays))],
                data_dict, **params)
        hlp_mod.add_param('fit_dicts', fp_ana.fit_dicts, data_dict,
                          update_value=True, **params)
        # if kf is not None:
        #     hlp_mod.add_param(
        #         # kf, [fp_ana.fit_res[f"gauss_fit_{mobjn}_slice{i}"] for i in
        #         #      range(len(delays))],
        #         kf, [fp_ana.fit_res[k] for k in fp_ana.fit_res.keys()
        #              if k.startswith(f"gauss_fit_{mobjn}_slice")],
        #         data_dict, **params)

def fd_select_from_list(data_dict, keys_in, keys_out, index, **params):
    for ki, ko in zip(keys_in, keys_out):
        hlp_mod.add_param(
            ko, hlp_mod.get_param(ki, data_dict, **params)[index],
            data_dict, **params)

def fd_fit_iir(data_dict, keys_in, keys_out, keys_corrected=None, method='JB',
               set_param_hints=True, fixed_A=None, return_expmod=False,
               **params):
    if method not in ['JB', 'JB_iter', 'integral']:
        raise NotImplementedError(f"Method {method} not implemented.")
    if isinstance(fixed_A, str):  # a key was given
        fixed_A = hlp_mod.get_param(fixed_A, data_dict, **params)
    folder = hlp_mod.get_param('folder', data_dict, **params)
    if folder is None:
        timestamp = hlp_mod.get_param('timestamps', data_dict, **params)[0]
        folder = a_tools.get_folder(timestamp)
    if keys_corrected is None:
        keys_corrected = [None]
    fit_range = hlp_mod.get_param('fitting_range', data_dict, **params)
    pulse_amp = hlp_mod.get_param('flux_pulse_amplitude', data_dict,
                                  default_value=1, **params)
    for ki, ko, kcorr in zip(keys_in, keys_out, keys_corrected):
        data = hlp_mod.get_param(ki, data_dict, **params)
        dt = hlp_mod.get_param('dt', data_dict, raise_error=True, **params)

        t_factor = 1e-6  # seems reasonable for numerical stability
        if method in ['JB', 'JB_iter']:
            prep_data, delays_filter, dt_osc, i_start, i_end, t_end = \
                IIR_fitting.prepare_data(
                    np.array([data[0], data[1] / pulse_amp]).T,
                    plot=False)
            lmfit_model = deepcopy(IIR_fitting.f_exp_model)
            initial_values_dict = hlp_mod.get_param(
                'initial_values_dict', data_dict, default_value={}, **params)
            if fixed_A is not None:
                lmfit_model.set_param_hint('A', vary=False)
                initial_values_dict.update({'A': fixed_A})
            if method == 'JB_iter':
                if fit_range is not None:
                    fit_range = [np.array(fit_range)]
                IIR_coeffs, data_corr, fit_res = \
                    IIR_fitting.iterate_IIR_fitting_exp_model(
                        prep_data, i_start, i_end, keep_i_end=False,
                        fitting_ranges=fit_range,
                        dt_awg=dt, lmfit_model=lmfit_model,
                        n_fits=1, plot=False, plot_lmfit=False,
                        IIR_filter_coeffs={},
                        initial_values_dict=initial_values_dict,
                        set_param_hints=set_param_hints, folder=folder)
                if return_expmod:
                    tmp_dict = {'iir': list(IIR_coeffs.values())[0], 'dt': dt}
                    fd_IIR_to_expmod(tmp_dict, ['iir'], ['expmod'], **params)
                    hlp_mod.add_param(ko, tmp_dict['expmod'], data_dict,
                                      **params)
                else:
                    hlp_mod.add_param(ko, list(IIR_coeffs.values())[0],
                                      data_dict, **params)
                if kcorr is not None:
                    data_corr[:,1] *= pulse_amp
                    hlp_mod.add_param(kcorr, data_corr.T, data_dict, **params)
            else:
                mask = np.logical_and(data[0] > fit_range[0],
                                      data[0] < fit_range[1])
                fit_res, dt_osc = IIR_fitting.fit_IIR(
                    np.array([data[0][mask], data[1][mask] / pulse_amp]).T,
                    initial_values_dict, lmfit_model=lmfit_model,
                    plot=False, folder=folder)
                A, B, tau = [fit_res.result.params[p].value
                             for p in ['A', 'B', 'tau']]
            hlp_mod.add_param('fit_dicts',
                              {f'IIR_{method}_{ki}': dict(fit_res=fit_res)},
                              data_dict, update_value=True, **params)
        if method=='integral':
            mask = np.logical_and(data[0] > fit_range[0],
                                  data[0] < fit_range[1])
            A, B, C = fit_exp_integral_method(
                data[0][mask] / t_factor, data[1][mask] / pulse_amp,
                fixed_A=fixed_A)
            tau = -1 / C * t_factor
        if method in ['integral', 'JB']:
            if return_expmod:
                hlp_mod.add_param(ko, [A, B, tau], data_dict, **params)
            else:
                tmp_dict = {'expmod': [A, B, tau], 'dt': dt}
                fd_expmod_to_IIR(tmp_dict, ['expmod'], ['coeffs'], **params)
                hlp_mod.add_param(ko, tmp_dict['coeffs'], data_dict,
                                  **params)
            if kcorr is not None:
                fd_apply_distortion(data_dict, [ki], [kcorr], [ko],
                                    'expmod' if return_expmod else 'IIR',
                                    **params)

def fit_exp_integral_method(x, y, fixed_A=None):
    """
    Fits y(x) = A + B * exp(C * x) to given data.

    Based on [1].

    [1] https://de.scribd.com/doc/14674814/Regressions-et-equations-integrales, page 16-18
    """
    s = np.zeros(len(x))
    for i in range(1, len(x)):
        s[i] = s[i - 1] + .5 * (y[i] + y[i - 1]) * (
                    x[i] - x[i - 1])

    x_x = np.sum((x - x[0]) ** 2)
    x_s = np.sum((x - x[0]) * s)
    s_s = np.sum(s ** 2)
    y_x = np.sum((x - x[0]) * (y - y[0]))
    y_s = np.sum((y - y[0]) * s)

    if fixed_A is not None:
        a1 = fixed_A
        alpha = (-a1 * (x - x[0]) + s)
        alpha_beta = np.sum(alpha * (y - y[0]))
        alpha_alpha = np.sum(alpha ** 2)
        c1 = alpha_beta / alpha_alpha

    else:
        res = np.linalg.solve(np.matrix([[x_x, x_s], [x_s, s_s]]),
                              np.array([y_x, y_s]))
        a1 = - res[0] / res[1]
        c1 = res[1]
    e = np.exp(c1 * x)
    sum_e = sum(e)
    e_e = sum(e ** 2)
    sum_y = sum(y)
    y_e = sum(y * e)

    if fixed_A is not None:
        a2 = fixed_A
        gamma = y - a2
        e_gamma = np.sum(e * gamma)
        b2 = e_gamma / e_e
    else:
        res = np.linalg.solve(
            np.matrix([[len(x), sum_e], [sum_e, e_e]]),
            np.array([sum_y, y_e]))
        a2 = res[0]
        b2 = res[1]

    # we could also return a1 as A, but [1] suggests that this new value is better (to be checked)
    A = a2
    B = b2
    C = c1
    return A, B, C

def fd_cut_time_range(data_dict, keys_in, keys_out, keys_range, **params):
    if len(keys_range) == 1 and len(keys_in) > 1:
        keys_range = keys_range * len(keys_in)
    elif len(keys_in) == 1 and len(keys_range) > 1:
        keys_in = keys_in * len(keys_range)
    for ki, ko, kr in zip(keys_in, keys_out, keys_range):
        data = hlp_mod.get_param(ki, data_dict, **params)
        range = hlp_mod.get_param(kr, data_dict, **params)
        mask = np.logical_and(data[0] > range[0],
                              data[0] < range[1])
        hlp_mod.add_param(ko, np.array([data[0][mask], data[1][mask]]),
                          data_dict, **params)

def fd_derivative(data_dict, keys_in, keys_out, **params):
    for ki, ko in zip(keys_in, keys_out):
        data = np.array(hlp_mod.get_param(ki, data_dict, **params))
        if data.ndim == 2:
            # interpret as time series
            data = [(data[0,0:-1] + data[0,1:]) / 2,
                    np.diff(data[1,:]) / np.diff(data[0,:])]
        else:
            data = np.diff(data)
        hlp_mod.add_param(ko, data, data_dict, **params)

def fd_normalize(data_dict, keys_in, keys_out, target_interval=None, **params):
    def normalize(x):
        if target_interval is None:
            return x / np.max(np.abs(x))
        else:
            m = np.min(x)
            return (x - m) / (np.max(x) - m)

    for ki, ko in zip(keys_in, keys_out):
        data = np.array(hlp_mod.get_param(ki, data_dict, **params))
        if data.ndim == 2:  # interpret as time series
            data[1] = normalize(data[1])
        else:
            data = normalize(data)
        hlp_mod.add_param(ko, data, data_dict, **params)

def fd_combine_IIR(data_dict, keys_in, keys_out, **params):
    iir = [[], []]
    for ki in keys_in:
        a, b = hlp_mod.get_param(ki, data_dict, **params)
        if len(a) == 0:
            continue
        if hasattr(a[0], '__iter__'):
            [iir[0].append(v) for v in a]
            [iir[1].append(v) for v in b]
        else:
            iir[0].append(a)
            iir[1].append(b)
    hlp_mod.add_param(keys_out[0], iir, data_dict, **params)

def fd_export_IIR(data_dict, keys_in, keys_out=None, folder=None, **params):
    mobjn = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['mobjn'], **params)
    if len(keys_in) > 1:
        fd_combine_IIR(data_dict, keys_in, [f'{keys_in[0]}_combined'],
                       **params)
        iir = hlp_mod.get_param(f'{keys_in[0]}_combined', data_dict, **params)
    else:
        iir = hlp_mod.get_param(keys_in[0], data_dict, **params)
    timestamps = hlp_mod.get_param('timestamps', data_dict, **params)
    if folder is None:
        folder = a_tools.get_folder(timestamps[0])
    k = 'flux_pulse_channel'
    ch = hlp_mod.get_param(k, data_dict, **params)
    if ch is None:
        fd_load_qb_params(data_dict, {k: k}, timestamp=timestamps[0], **params)
        ch = hlp_mod.get_param(k, data_dict, **params)
    iir_dict = {f'IIR_{i}': [a, b] for i, [a, b]
                in enumerate(zip(iir[0], iir[1]))}
    file_name = IIR_fitting.export_IIR_coeffs(
        iir_dict, fluxline=mobjn, awg_channel=ch, folder=folder)
    if keys_out is not None:
        hlp_mod.add_param(
            keys_out[0],
            os.path.join(folder[len(a_tools.datadir):], file_name),
            data_dict, **params)


