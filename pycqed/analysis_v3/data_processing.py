import logging
log = logging.getLogger(__name__)

import itertools
import numpy as np
from collections import OrderedDict
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.analysis_v3 import helper_functions as hlp_mod
from pycqed.analysis_v3 import processing_pipeline as pp_mod
from pycqed.measurement.calibration.calibration_points import CalibrationPoints

import sys
pp_mod.search_modules.add(sys.modules[__name__])


def filter_data(data_dict, keys_in, keys_out=None, **params):
    """
    Filters data in data_dict for each keys_in according to data_filter
    in params. Creates new keys_out in the data dict for the filtered data.

    To be used for example for filtering:
        - reset readouts
        - data with and without flux pulse/ pi pulse etc.

    :param data_dict: OrderedDict containing data to be processed and where
                    processed data is to be stored
    :param keys_in: list of key names or dictionary keys paths in
                    data_dict for the data to be processed
    :param keys_out: list of key names or dictionary keys paths in
                    data_dict for the processed data to be saved into
    :param params: keyword arguments:
        data_filter (str, default: 'lambda x: x'): filtering condition passed
            as a string that will be evaluated with eval.

    Assumptions:
        - if any keyo in keys_out contains a '.' string, keyo is assumed to
        indicate a path in the data_dict.
        - len(keys_out) == len(keys_in)
    """
    data_to_proc_dict = hlp_mod.get_data_to_process(data_dict, keys_in)
    if keys_out is None:
        keys_out = [k+' filtered' for k in keys_in]
    if len(keys_out) != len(data_to_proc_dict):
        raise ValueError('keys_out and keys_in do not have '
                         'the same length.')
    data_filter_func = hlp_mod.get_param('data_filter', data_dict,
                                  default_value=lambda data: data, **params)
    if hasattr(data_filter_func, '__iter__'):
        data_filter_func = eval(data_filter_func)
    for keyo, keyi in zip(keys_out, list(data_to_proc_dict)):
        hlp_mod.add_param(
            keyo, data_filter_func(data_to_proc_dict[keyi]), data_dict,
            **params)
    return data_dict


def get_std_deviation(data_dict, keys_in, keys_out=None, **params):
    """
    Finds the standard deviation of the num_bins in data_dict for each
    keys_in.

    To be used for example for:
        - shots
        - RB seeds

    :param data_dict: OrderedDict containing data to be processed and where
                    processed data is to be stored
    :param keys_in: list of key names or dictionary keys paths in
                    data_dict for the data to be processed
    :param keys_out: list of key names or dictionary keys paths in
                    data_dict for the processed data to be saved into
    :param params: keyword arguments:
        num_bins (int): number of averaging bins for each entry in keys_in

    Assumptions:
        - if any keyo in keys_out contains a '.' string, keyo is assumed to
        indicate a path in the data_dict.
        - len(keys_out) == len(keys_in)
        - num_bins exists in params
        - num_bins exactly divides data_dict[keyi] for all keyi in keys_in.
    """
    data_to_proc_dict = hlp_mod.get_data_to_process(data_dict, keys_in)
    if keys_out is None:
        keys_out = [k + ' std' for k in keys_in]
    shape = hlp_mod.get_param('shape', data_dict, raise_error=True,
                                    **params)
    averaging_axis = hlp_mod.get_param('averaging_axis', data_dict,
                                             default_value=-1, **params)
    if len(keys_out) != len(data_to_proc_dict):
        raise ValueError('keys_out and keys_in do not have '
                         'the same length.')
    for k, keyi in enumerate(data_to_proc_dict):
        if len(data_to_proc_dict[keyi]) % shape[0] != 0:
            raise ValueError(f'{shape[0]} does not exactly divide '
                             f'data from ch {keyi} with length '
                             f'{len(data_to_proc_dict[keyi])}.')
        data_for_std = data_to_proc_dict[keyi] if shape is None else \
            np.reshape(data_to_proc_dict[keyi], shape)
        hlp_mod.add_param(
            keys_out[k], np.std(data_for_std, axis=averaging_axis), data_dict,
            update_value=params.get('update_value', False), **params)
    return data_dict


def classify_gm(data_dict, keys_out, keys_in, **params):
    """
    BROKEN
    TODO: need to correctly handle channel tuples

    Predict gaussian mixture posterior probabilities for single shots
    of different levels of a qudit. Data to be classified expected in the
    shape (n_datapoints, n_channels).
    :param data_dict: OrderedDict containing data to be processed and where
                    processed data is to be stored
    :param keys_in:
                    qubit: list of key names or dictionary keys paths
                    qutrit: list of tuples of key names or dictionary keys paths
                        in data_dict for the data to be processed
    :param keys_out: list of tuples of key names or dictionary keys paths
                        in data_dict for the processed data to be saved into
    :param params: keyword arguments:
        clf_params: list of dictionaries with parameters for
            Gaussian Mixture classifier:
                means_: array of means of each component of the GM
                covariances_: covariance matrix
                covariance_type: type of covariance matrix
                weights_: array of priors of being in each level. (n_levels,)
                precisions_cholesky_: array of precision_cholesky
            For more info see about parameters see :
            https://scikit-learn.org/stable/modules/generated/sklearn.mixture.
            GaussianMixture.html
    For each item in keys_out, stores int data_dict an
    (n_datapoints, n_levels) array of posterior probability of being
    in each level.

    Assumptions:
        - if any keyo in keys_out contains a '.' string, keyo is assumed to
        indicate a path in the data_dict.
        - keys_in is a list of tuples for qutrit and
            list of strings for qubit
        - keys_out is a list of tuples
        - len(keys_out) == len(keys_in) + 1
        - clf_params exist in **params
    """
    pass
    # clf_params = hlp_mod.get_param('clf_params', data_dict, **params)
    # if clf_params is None:
    #     raise ValueError('clf_params is not specified.')
    # reqs_params = ['means_', 'covariances_', 'covariance_type',
    #                'weights_', 'precisions_cholesky_']
    #
    # data_to_proc_dict = hlp_mod.get_data_to_process(
    #     data_dict, keys_in)
    #
    # data = data_dict
    # all_keys = keys_out[k].split('.')
    # for i in range(len(all_keys)-1):
    #     if all_keys[i] not in data:
    #         data[all_keys[i]] = OrderedDict()
    #     else:
    #         data = data[all_keys[i]]
    #
    # clf_params_temp = deepcopy(clf_params)
    # for r in reqs_params:
    #     assert r in clf_params_temp, "Required Classifier parameter {} " \
    #                                  "not given.".format(r)
    # gm = GM(covariance_type=clf_params_temp.pop('covariance_type'))
    # for param_name, param_value in clf_params_temp.items():
    #     setattr(gm, param_name, param_value)
    # data[all_keys[-1]] = gm.predict_proba(data_to_proc_dict[keyi])
    # return data_dict


def do_standard_preselection(data_dict, keys_in, keys_out=None,
                             joint_processing=False, **params):
    """
    Does standard preselection on the data shot arrays in data_dict specified
    by keys_in. Only the data shots for which the preselection readout
    preceding it found the qubit in the 0 state.
    :param data_dict: OrderedDict containing data to be processed and where
                    processed data is to be stored
    :param keys_in: list of key names or dictionary keys paths in
                    data_dict for the data to be processed
    :param keys_out: list of key names or dictionary keys paths in
                    data_dict for the processed data to be saved into
    :param joint_processing: bool specifying whether to preselect on all the
        measurement objects (arrays specified by keys_in) being in ground state
    :param params: keyword arguments

    Assumptions:
        - the data pointed to by keys_in is assumed to be 1D arrays of
            thresholded/classified shots
        - every other shot starting at 0 is assumed to be a preselection readout

    WARNING! The processed data array will not necessarily have the same length
    as the input array.
    """
    data_to_proc_dict = hlp_mod.get_data_to_process(data_dict, keys_in)
    if not joint_processing:
        for k, keyi in enumerate(data_to_proc_dict):
            th_shots = data_to_proc_dict[keyi]
            if not all(e in [0, 1, 2] for e in np.unique(th_shots)):
                raise TypeError(f'The data corresponding to {keyi} does not '
                                f'contain thresholded shots.')
            presel_shots = th_shots[::2]
            data_shots = th_shots[1::2]
            hlp_mod.add_param(
                keys_out[k], data_shots[
                    np.logical_not(np.ma.make_mask(presel_shots))] ,
                data_dict, update_value=params.get('update_value', False),
                **params)
    else:
        all_shots = np.array(list(data_to_proc_dict.values()))
        presel_states = all_shots[:, ::2]
        data_states = all_shots[:, 1::2]
        presel_data = data_states[:, np.where(
            np.all(presel_states.transpose() == (0, 0, 0, 0), axis=1))[0]]
        for k, keyo in enumerate(keys_out):
            hlp_mod.add_param(
                keyo, presel_data[k],
                data_dict, update_value=params.get('update_value', False),
                **params)


def do_preselection(data_dict, classified_data, keys_out, **params):
    """
    Keeps only the data for which the preselection readout data in
    classified_data satisfies the preselection condition.
    :param data_dict: OrderedDict containing data to be processed and where
                    processed data is to be stored
    :param classified_data: list of arrays of 0,1 for qubit, and
                    0,1,2 for qutrit, or list of keys pointing to the binary
                    (or trinary) arrays in the data_dict
    :param keys_out: list of key names or dictionary keys paths in
                    data_dict for the processed data to be saved into
    :param params: keyword arguments.:
        presel_ro_idxs (function, default: lambda idx: idx % 2 == 0):
            specifies which (classified) data entry is a preselection ro
        keys_in (list): list of key names or dictionary keys paths in
            data_dict for the data to be processed
        presel_condition (int, default: 0): 0, 1 (, or 2 for qutrit). Keeps
            data for which the data in classified data corresponding to
            preselection readouts satisfies presel_condition.

    Assumptions:
        - if any keyo in keys_out contains a '.' string, keyo is assumed to
        indicate a path in the data_dict.
        - len(keys_out) == len(classified_data)
        - if keys_in are given, len(keys_in) == len(classified_data)
        - classified_data either list of arrays or list of strings
        - if classified_data contains strings, assumes they are keys in
            data_dict
    """
    if len(keys_out) != len(classified_data):
        raise ValueError('classified_data and keys_out do not have '
                         'the same length.')

    keys_in = params.get('keys_in', None)
    presel_ro_idxs = hlp_mod.get_param('presel_ro_idxs', data_dict,
                               default_value=lambda idx: idx % 2 == 0, **params)
    presel_condition = hlp_mod.get_param('presel_condition', data_dict,
                                 default_value=0, **params)
    if keys_in is not None:
        if len(keys_in) != len(classified_data):
            raise ValueError('classified_data and keys_in do not have '
                             'the same length.')
        data_to_proc_dict = hlp_mod.get_data_to_process(data_dict, keys_in)
        for i, keyi in enumerate(data_to_proc_dict):
            # Check if the entry in classified_data is an array or a string
            # denoting a key in the data_dict
            if isinstance(classified_data[i], str):
                if classified_data[i] in data_dict:
                    classif_data = data_dict[classified_data[i]]
                else:
                    raise KeyError(
                        f'{classified_data[i]} not found in data_dict.')
            else:
                classif_data = classified_data[i]

            mask = np.zeros(len(data_to_proc_dict[keyi]))
            val = True
            for idx in range(len(data_to_proc_dict[keyi])):
                if presel_ro_idxs(idx):
                    val = (classif_data[idx] == presel_condition)
                    mask[idx] = False
                else:
                    mask[idx] = val
            preselected_data = data_to_proc_dict[keyi][mask]
            hlp_mod.add_param(keys_out[i], preselected_data, data_dict,
                              update_value=params.get('update_value', False))
    else:
        for i, keyo in enumerate(keys_out):
            # Check if the entry in classified_data is an array or a string
            # denoting a key in the data_dict
            if isinstance(classified_data[i], str):
                if classified_data[i] in data_dict:
                    classif_data = data_dict[classified_data[i]]
                else:
                    raise KeyError(
                        f'{classified_data[i]} not found in data_dict.')
            else:
                classif_data = classified_data[i]

            mask = np.zeros(len(classif_data))
            val = True
            for idx in range(len(classif_data)):
                if presel_ro_idxs(idx):
                    val = (classif_data[idx] == 0)
                    mask[idx] = False
                else:
                    mask[idx] = val
            hlp_mod.add_param(keyo, classif_data[mask], data_dict,
                              update_value=params.get('update_value', False))
    return data_dict


def average_data(data_dict, keys_in, keys_out=None, **params):
    """
    Averages data in data_dict specified by keys_in into num_bins.
    :param data_dict: OrderedDict containing data to be processed and where
                    processed data is to be stored
    :param keys_in: list of key names or dictionary keys paths in
                    data_dict for the data to be processed
    :param keys_out: list of key names or dictionary keys paths in
                    data_dict for the processed data to be saved into
    :param params: keyword arguments.:
        num_bins (int): number of averaging bins for each entry in keys_in

    Assumptions:
        - if any keyo in keys_out contains a '.' string, keyo is assumed to
        indicate a path in the data_dict.
        - len(keys_out) == len(keys_in)
        - num_bins exists in params
        - num_bins exactly divides data_dict[keyi] for all keyi in keys_in.
    """
    data_to_proc_dict = hlp_mod.get_data_to_process(data_dict, keys_in)
    if keys_out is None:
        keys_out = [k + ' averaged' for k in keys_in]
    shape = hlp_mod.get_param('shape', data_dict, **params)
    averaging_axis = hlp_mod.get_param('averaging_axis', data_dict,
                                             default_value=-1, **params)
    if len(keys_out) != len(data_to_proc_dict):
        raise ValueError('keys_out and keys_in do not have '
                         'the same length.')
    for k, keyi in enumerate(data_to_proc_dict):
        if len(data_to_proc_dict[keyi]) % shape[0] != 0:
            raise ValueError(f'{shape[0]} does not exactly divide '
                             f'data from ch {keyi} with length '
                             f'{len(data_to_proc_dict[keyi])}.')
        data_to_avg = data_to_proc_dict[keyi] if shape is None else \
            np.reshape(data_to_proc_dict[keyi], shape)
        avg_data = np.mean(data_to_avg, axis=averaging_axis)
        if avg_data.ndim > 1:
            avg_data = avg_data.flatten()
        hlp_mod.add_param(
            keys_out[k], avg_data ,
            data_dict, update_value=params.get('update_value', False), **params)
    return data_dict


def transform_data(data_dict, keys_in, keys_out, **params):
    """
    Maps data in data_dict specified by data_keys_in using transform_func
     callable (can be any function).
    :param data_dict: OrderedDict containing data to be processed and where
                    processed data is to be stored
    :param keys_in: list of key names or dictionary keys paths in
                    data_dict for the data to be processed
    :param keys_out: list of key names or dictionary keys paths in
                    data_dict for the processed data to be saved into
    :param params: keyword arguments.:
        transform_func (callable): string form of a callable or callable
            function
        transform_func_kwargs (dict): additional arguments to forward to
            transform_func

    """
    transform_func = hlp_mod.get_param('transform_func',
                                             data_dict, **params)
    tf_kwargs = hlp_mod.get_param('transform_func_kwargs', data_dict,
                                        default_value=dict(), **params)
    if transform_func is None:
        raise ValueError('mapping is not specified.')
    elif isinstance(transform_func, str):
        transform_func = eval(transform_func)
    data_to_proc_dict = hlp_mod.get_data_to_process(data_dict, keys_in)

    if len(keys_out) != len(data_to_proc_dict):
        raise ValueError('keys_out and keys_in do not have the same length.')

    for keyi, keyo in zip(data_to_proc_dict, keys_out):
        hlp_mod.add_param(
            keyo, transform_func(data_to_proc_dict[keyi], **tf_kwargs),
            data_dict, update_value=params.get('update_value', False))
    return data_dict


def correct_readout(data_dict, keys_in, keys_out, state_prob_mtx, **params):
    """
    Maps data in data_dict specified by data_keys_in using transform_func
     callable (can be any function).
    :param data_dict: OrderedDict containing data to be processed and where
                    processed data is to be stored
    :param keys_in: list of key names or dictionary keys paths in
                    data_dict for the data to be processed
    :param keys_out: list of key names or dictionary keys paths in
                    data_dict for the processed data to be saved into
    :param state_prob_mtx: correct state assignment probability matrix
    :param params: keyword arguments.

    Assumptions:
        - keys_in correspond to the data in the correct order with respect to
        the order of the rows and columns in the state_prob_mtx (usually
        'g', 'e', 'f').
    """
    data_to_proc_dict = hlp_mod.get_data_to_process(data_dict, keys_in)

    if len(keys_out) != len(data_to_proc_dict):
        raise ValueError('keys_out and keys_in do not have the same length.')

    uncorrected_data = np.stack(list(data_to_proc_dict.values()))
    corrected_data = (np.linalg.inv(state_prob_mtx).T @ uncorrected_data).T
    for i, keyo in enumerate(keys_out):
        hlp_mod.add_param(
            keyo, corrected_data[:, i],
            data_dict, update_value=params.get('update_value', False))
    return data_dict


def rotate_iq(data_dict, keys_in, keys_out=None, **params):
    """
    Rotates IQ data based on information in the CalibrationPoints objects.
    :param data_dict: OrderedDict containing data to be processed and where
                processed data is to be stored
    :param keys_in: list of length-2 tuples of key names or dictionary
                keys paths in data_dict for the data to be processed
    :param keys_out: list of key names or dictionary keys paths in
                data_dict for the processed data to be saved into
    :param params: keyword arguments.:
        cal_points (CalibrationPoints object or its repr):
            CalibratinonPoints object for the meaj_obj_name, or its repr
        last_ge_pulse (bool): only for a measurement on ef. Whether a ge
            pi-pulse was applied before measurement.
        meas_obj_value_names_map (dict): {meaj_obj_name: [value_names]}.

    Assumptions:
        - if any keyo in keys_out contains a '.' character, keyo is assumed to
        indicate a path in the data_dict.
        - len(keys_in) == 2 must be True; the 2 entries are I and Q data
        - len(keys_out) == 1 must be True.
        - cal_points exists in **params, data_dict, or metadata
        - assumes meas_obj_names is one of the keys of the dicts returned by
        CalibrationPoints.get_indices(), CalibrationPoints.get_rotations()
        - keys_in exist in meas_obj_value_names_map
    """
    data_to_proc_dict = hlp_mod.get_data_to_process(
        data_dict, keys_in)
    keys_in = list(data_to_proc_dict)
    if keys_out is None:
        keys_out = ['rotated datadata [' + ','.join(keys_in)+']']
    if len(keys_in) != 2:
        raise ValueError(f'keys_in must have length two. {len(keys_in)} '
                         f'entries were given.')

    cp = hlp_mod.get_param('cal_points', data_dict, raise_error=True, **params)
    if isinstance(cp, str):
        cp = CalibrationPoints.from_string(cp)
    last_ge_pulse = hlp_mod.get_param('last_ge_pulse', data_dict,
                                      default_value=[], **params)
    mobjn = hlp_mod.get_param('meas_obj_names', data_dict,
                              raise_error=True, **params)
    if isinstance(mobjn, list):
        mobjn = mobjn[0]
    if mobjn not in cp.qb_names:
        raise KeyError(f'{mobjn} not found in cal_points.')

    rotations = cp.get_rotations(last_ge_pulses=last_ge_pulse)
    ordered_cal_states = []
    for ii in range(len(rotations[mobjn])):
        ordered_cal_states += \
            [k for k, idx in rotations[mobjn].items() if idx == ii]
    rotated_data, _, _ = \
        a_tools.rotate_and_normalize_data_IQ(
            data=list(data_to_proc_dict.values()),
            cal_zero_points=None if len(ordered_cal_states) == 0 else
                cp.get_indices()[mobjn][ordered_cal_states[0]],
            cal_one_points=None if len(ordered_cal_states) == 0 else
                cp.get_indices()[mobjn][ordered_cal_states[1]])
    hlp_mod.add_param(keys_out[0], rotated_data, data_dict,
                      update_value=params.get('update_value', False))
    return data_dict


def rotate_1d_array(data_dict, keys_in, keys_out=None, **params):
    """
    Rotates 1d array based on information in the CalibrationPoints objects.
    The number of CalibrationPoints objects should equal the number of
    key names in keys_in.
    :param data_dict: OrderedDict containing data to be processed and where
                processed data is to be stored
    :param keys_in: list of key names or dictionary keys paths in
                data_dict for the data to be processed
    :param keys_out: list of key names or dictionary keys paths in
                data_dict for the processed data to be saved into
    :param params: keyword arguments.:
        cal_points_list (list): list of CalibrationPoints objects.
        last_ge_pulses (list): list of booleans

    Assumptions:
        - if any keyo in keys_out contains a '.' string, keyo is assumed to
        indicate a path in the data_dict.
        - len(keys_in) == len(keys_out) == 1 must all be True
        - cal_points exists in **params, data_dict, or metadata
        - assumes meas_obj_namess is one of the keys of the dicts returned by
        CalibrationPoints.get_indices(), CalibrationPoints.get_rotations()
        - keys_in exists in meas_obj_value_names_map
    """
    data_to_proc_dict = hlp_mod.get_data_to_process(data_dict, keys_in)
    keys_in = list(data_to_proc_dict)
    if keys_out is None:
        keys_out = [f'rotated data [{keys_in[0]}]']
    if len(keys_in) != 1:
        raise ValueError(f'keys_in must have length one. {len(keys_in)} '
                         f'entries were given.')
    if len(keys_out) != len(data_to_proc_dict):
        raise ValueError('keys_out and keys_in do not have '
                         'the same length.')
    cp = hlp_mod.get_param('cal_points', data_dict, raise_error=True,
                                 **params)
    if isinstance(cp, str):
        cp = CalibrationPoints.from_string(cp)
    last_ge_pulses = hlp_mod.get_param('last_ge_pulses', data_dict,
                                             default_value=[], **params)
    mobjn = hlp_mod.get_param('meas_obj_names', data_dict,
                                    raise_error=True, **params)
    if isinstance(mobjn, list):
        mobjn = mobjn[0]
    if mobjn not in cp.qb_names:
        raise KeyError(f'{mobjn} not found in cal_points.')

    rotations = cp.get_rotations(last_ge_pulses=last_ge_pulses)
    ordered_cal_states = []
    for ii in range(len(rotations[mobjn])):
        ordered_cal_states += \
            [k for k, idx in rotations[mobjn].items() if idx == ii]
    rotated_data = \
        a_tools.rotate_and_normalize_data_1ch(
            data=data_to_proc_dict[keys_in[0]],
            cal_zero_points=None if len(ordered_cal_states) == 0 else
                cp.get_indices()[mobjn][ordered_cal_states[0]],
            cal_one_points=None if len(ordered_cal_states) == 0 else
                cp.get_indices()[mobjn][ordered_cal_states[1]])
    hlp_mod.add_param(keys_out[0], rotated_data, data_dict,
                      update_value=params.get('update_value', False))
    return data_dict


def classify_data(data_dict, keys_in, threshold_list, keys_out=None, **params):
    """
    Thresholds the data in data_dict specified by keys_in according to the
    threshold_mapping and the threshold values in threshold_list.
    This node will create nr_states entries in the data_dict, where
    nr_states = len(set(threshold_mapping.values())).
    :param data_dict: OrderedDict containing data to be processed and where
                    processed data is to be stored
    :param keys_in: list of key names or dictionary keys paths in
                    data_dict for the data to be processed
    :param keys_out: list of key names or dictionary keys paths in
                    data_dict for the processed data to be saved into
    :param threshold_list: list of values around which to threshold each
        data array corresponding to keys_in.
    :param params: keyword arguments.:
        threshold_map (dict): dict of the form {idx: state_label}.
            Ex: {0: 'e', 1: 'g', 2: 'f', 3: 'g'}. Default value if
            len(threshold_list) == 1 is {0: 'g', 1: 'e'}. Else, None and a
            ValueError will be raise.
        !!! IMPORTANT: quadrants 1 and 2 are reversed compared to the
        threshold_map we save in qubit preparation parameters !!!

    Assumptions:
        - len(threshold_list) == len(keys_in)
        - data arrays corresponding to keys_in must all have the same length
        - the order of the values in threshold_list is important!
        The thresholds are in the same order as the data corresponding to
        the keys_in. See the 3 lines after extraction of threshold_map below.
    """
    if not hasattr(threshold_list, '__iter__'):
        threshold_list = [threshold_list]

    data_to_proc_dict = hlp_mod.get_data_to_process(data_dict, keys_in)
    if len(threshold_list) != len(data_to_proc_dict):
        raise ValueError('threshold_list and keys_in do not have '
                         'the same length.')
    if len(set([arr.size for arr in data_to_proc_dict.values()])) > 1:
        raise ValueError('The data arrays corresponding to keys_in must all '
                         'have the same length.')
    keys_in = list(data_to_proc_dict)
    threshold_map = hlp_mod.get_param('threshold_map', data_dict,
                                      raise_error=False, **params)
    if threshold_map is None:
        if len(threshold_list) == 1:
            threshold_map = {0: 'g', 1: 'e'}
        else:
            raise ValueError(f'threshold_map was not found in either '
                             f'exp_metadata or input params.')
    if keys_out is None:
        keyo = keys_in[0] if len(keys_in) == 1 else ','.join(keys_in)
        keys_out = [f'{keyo} {s}' for s in set(threshold_map.values())]

    # generate boolean array of size (nr_data_pts_per_ch, len(keys_in).
    thresh_data_binary = np.stack(
        [data_to_proc_dict[keyi] >= th for keyi, th in
         zip(keys_in, threshold_list)], axis=1)

    # convert each row of thresh_data_binary into the decimal value whose
    # binary representation is given by the booleans in each row.
    # thresh_data_decimal is a 1d array of size nr_data_pts_per_ch
    thresh_data_decimal = thresh_data_binary.dot(1 << np.arange(
        thresh_data_binary.shape[-1] - 1, -1, -1))

    for k, state in enumerate(set(threshold_map.values())):
        dd = data_dict
        all_keys = keys_out[k].split('.')
        for i in range(len(all_keys)-1):
            if all_keys[i] not in dd:
                dd[all_keys[i]] = OrderedDict()
            dd = dd[all_keys[i]]
        hlp_mod.add_param(all_keys[-1], np.zeros(
            len(list(data_to_proc_dict.values())[0])), dd,
            update_value=params.get('update_value', False))

        # get the decimal values corresponding to state from threshold_map.
        state_idxs = [k for k, v in threshold_map.items() if v == state]

        for idx in state_idxs:
            dd[all_keys[-1]] = np.logical_or(
                dd[all_keys[-1]], thresh_data_decimal == idx).astype('int')

    return data_dict


def threshold_data(data_dict, keys_in, keys_out, ro_thresholds=None, **params):
    """
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
    if len(keys_in) != 1:
        raise ValueError('keys_in must have length 1.')

    mobjn = hlp_mod.get_measurement_properties(data_dict,
                                               props_to_extract=['mobjn'],
                                               enforce_one_meas_obj=True,
                                               **params)
    if ro_thresholds is None:
        acq_classifier_params = hlp_mod.get_param(
            f'{mobjn}.acq_classifier_params', data_dict, raise_error=True,
            **params)
        if 'thresholds' not in acq_classifier_params:
            raise KeyError(f'thresholds does not exist in the '
                           f'acq_classifier_params for {mobjn}.')
        ro_thresholds = {mobjn: acq_classifier_params['thresholds'][0]}

    data_to_proc_dict = hlp_mod.get_data_to_process(data_dict, keys_in)
    keys_in = list(data_to_proc_dict)
    if len(keys_out) != len(keys_in):
        raise ValueError('keys_out and keys_in do not have '
                         'the same length.')

    # generate boolean array of size (nr_data_pts_per_ch, len(keys_in).
    if mobjn not in ro_thresholds:
        raise KeyError(f'{mobjn} not found in ro_thresholds={ro_thresholds}.')
    threshold_list = [ro_thresholds[mobjn]]
    thresh_dat = np.stack(
        [data_to_proc_dict[keyi] > th for keyi, th in
         zip(keys_in, threshold_list)], axis=1)

    for i, keyo, in enumerate(keys_out):
        hlp_mod.add_param(
            keyo, thresh_dat[:, i].astype('int'), data_dict,
            update_value=params.get('update_value', False))


def correlate_qubits(data_dict, keys_in, keys_out, **params):
    """
    Calculated the ZZ correlator of the arrays of shots indicated by keys_in
    as follows:
        - correlator = 0 if even number of 0's
        - correlator = 1 if odd number of 0's
    :param data_dict: OrderedDict containing data to be processed and where
                    processed data is to be stored
    :param keys_in: list of key names or dictionary keys paths in
                    data_dict for the data to be processed
    :param keys_out: list with one entry specifying the key name or dictionary
        key path in data_dict for the processed data to be saved into
    :param params: keyword arguments.
    :return:
        Saves in data_dict, under keys_out[0], the np.array of correlated shots
            with the same dimension as one of the arrays indicated by keys_in

    Assumptions:
        - data must be THRESHOLDED single shots (0's and 1's)
        - all data arrays indicated by keys_in will be correlated
    """
    if len(keys_out) != 1:
        raise ValueError('keys_out must have length 1.')
    data_to_proc_dict = hlp_mod.get_data_to_process(data_dict, keys_in)
    all_data_arr = np.array(list(data_to_proc_dict.values()))
    if not np.all(np.logical_or(all_data_arr == 0, all_data_arr == 1)):
        raise ValueError('Not all shots have been thresholded.')
    hlp_mod.add_param(
        keys_out[0], np.sum(all_data_arr, axis=0) % 2, data_dict,
        update_value=params.get('update_value', False))


def calculate_probability_table(data_dict, keys_in, keys_out=None, **params):
    """
    Creates a general table of normalized counts averaging out all but
        specified set of correlations.
    :param data_dict: OrderedDict containing data to be processed and where
        processed data is to be stored
    :param keys_in: list of key names or dictionary keys paths in
        data_dict for the data to be processed (thresholded shots)
    :param observables: List of observables. Observable is a dictionary with
        name of the qubit as key and boolean value indicating if it is
        selecting exited states. If the qubit is missing from the list
        of states it is averaged out. Instead of just the qubit name, a
        tuple of qubit name and a shift value can be passed, where the
        shift value specifies the relative readout index for which the
        state is checked.
        Example qb2-qb4 state tomo with preselection:
                {'pre': {('qb2', -1): False,
                        ('qb4', -1): False}, # preselection conditions
                 '$\\| gg\\rangle$': {'qb2': False,
                                      'qb4': False,
                                      ('qb2', -1): False,
                                      ('qb4', -1): False},
                 '$\\| ge\\rangle$': {'qb2': False,
                                      'qb4': True,
                                      ('qb2', -1): False,
                                      ('qb4', -1): False},
                 '$\\| eg\\rangle$': {'qb2': True,
                                      'qb4': False,
                                      ('qb2', -1): False,
                                      ('qb4', -1): False},
                 '$\\| ee\\rangle$': {'qb2': True,
                                      'qb4': True,
                                      ('qb2', -1): False,
                                      ('qb4', -1): False}}
    :param n_readouts: Assumed to be the period in the list of shots between
        experiments with the same prepared state. If shots_of_qubits
        includes preselection readout results or if there was several
        readouts for a single readout then n_readouts has to include them.
    :param params: keyword arguments: used if get_observables is called
        - preselection_shift (int, default: -1)
        - do_preselection (bool, default: False)
        - return_counts (bool, default: False): whether to return raw counts
            for each state (True), or normalize the counts by
            n_readouts/n_shots (False)
    :return adds to data_dict, under keys_out, a dict with observables as keys
        and np.array of normalized counts with size n_readouts as values

    Assumptions:
        - len(keys_out) == 1 -> one probability table is calculated
        - !!! This function returns a dict not an array like the static method
        probability_table in readout_analysis.py/MultiQubit_SingleShot_Analysis
    """

    # Get shots_of_qubits: Dictionary of np.arrays of thresholded shots for
    # each qubit.
    data_to_proc_dict = hlp_mod.get_data_to_process(data_dict, keys_in)
    n_readouts = hlp_mod.get_param('n_readouts', data_dict, raise_error=True,
                                   **params)
    observables = hlp_mod.get_param('observables', data_dict,
                                    raise_error=True, **params)

    return_counts = hlp_mod.get_param('return_counts', data_dict,
                                      default_value=False, **params)

    n_shots = next(iter(data_to_proc_dict.values())).shape[0]
    table = OrderedDict({obs: np.zeros(n_readouts) for obs in observables})
    res_e = {}
    res_g = {}
    for keyi, results in data_to_proc_dict.items():
        mobjn = keyi.split('.')[0]
        res_e[mobjn] = np.array(results).reshape((n_readouts, -1),
                                                 order='F')
        # This makes copy, but allows faster AND later
        res_g[mobjn] = np.logical_not(
            np.array(results)).reshape((n_readouts, -1), order='F')

    for readout_n in range(n_readouts):
        # first result all ground
        for obs, states_of_mobjs in observables.items():
            mask = np.ones((n_shots//n_readouts), dtype=np.bool)
            # slow qubit is the first in channel_map list
            for mobjn, state in states_of_mobjs.items():
                if isinstance(mobjn, tuple):
                    seg = (readout_n+mobjn[1]) % n_readouts
                    mobjn = mobjn[0]
                else:
                    seg = readout_n
                if state:
                    mask = np.logical_and(mask, res_e[mobjn][seg])
                else:
                    mask = np.logical_and(mask, res_g[mobjn][seg])
            table[obs][readout_n] = np.count_nonzero(mask) * (
                    n_readouts/n_shots if not return_counts else 1)

    if keys_out is not None:
        if len(keys_out) != 1:
            raise ValueError(f'keys_out must have length one. {len(keys_out)} '
                             f'entries were given.')
        hlp_mod.add_param(keys_out[0], table, data_dict, **params)
    else:
        return table


def calculate_meas_ops_and_covariations(
        data_dict, observables, keys_out=None, meas_obj_names=None,
        **params):
    """
    Calculates the measurement operators corresponding to each observable and
        the expected covariance matrix between the operators from the
        observables and meas_obj_names.
    :param data_dict: OrderedDict containing data to be processed and where
        processed data is to be stored
    :param observables: measurement observables, see docstring of hlp_mod.
        get_observables. If not provided, it will default to hlp_mod.
        get_observables. See required input params there.
    :param keys_out: list of key names or dictionary keys paths in
        data_dict for the processed data to be saved into
    :param meas_obj_names: list of measurement object names
    :param params: keyword arguments: meas_obj_names can be passed here
    :return: adds to data_dict:
        - the measurement operators corresponding to each observable and the
            expected covariance matrix between the operators under keys_out
        - if keys_out is None, it will saves the aforementioned quantities
            under  'measurement_ops' and 'cov_matrix_meas_obs'
    Assumptions:
     - len(keys_out) == 2
     - order in keys_out corresponds to [measurement_operators, covar_matrix]
    """
    if keys_out is None:
        keys_out = ['measurement_ops', 'cov_matrix_meas_obs']
    if len(keys_out) != 2:
        raise ValueError(f'keys_out must have length 2. {len(keys_out)} '
                         f'entries were given.')

    if meas_obj_names is None:
        meas_obj_names = hlp_mod.get_measurement_properties(
            data_dict, props_to_extract=['mobjn'], enforce_one_meas_obj=False,
            **params)

    Fs = []
    Fsingle = {None: np.array([[1, 0], [0, 1]]),
               True: np.array([[0, 0], [0, 1]]),
               False: np.array([[1, 0], [0, 0]])}
    Omega = []
    for obs in observables.values():
        F = np.array([[1]])
        nr_meas = 0
        for qb in meas_obj_names:
            # TODO: does not handle conditions on previous readouts
            Fqb = Fsingle[obs.get(qb, None)]
            # Kronecker product convention - assumed the same as QuTiP
            F = np.kron(F, Fqb)
            if qb in obs:
                nr_meas += 1
        Fs.append(F)
        # The variation is proportional to the number of qubits we have
        # a condition on, assuming that all readout errors are small
        # and equal.
        Omega.append(nr_meas)
    Omega = np.array(Omega)

    hlp_mod.add_param(keys_out[0], Fs, data_dict, **params)
    hlp_mod.add_param(keys_out[1], Omega, data_dict, **params)


def calculate_meas_ops_and_covariations_cal_points(
        data_dict, keys_in, observables, keys_out=None, **params):
    """
    Calculates the measurement operators corresponding to each observable and
        the expected covariance matrix between the operators from the
        observables and calibration points.
    :param data_dict: OrderedDict containing data to be processed and where
        processed data is to be stored
    :param keys_in: list of key names or dictionary keys paths in data_dict
        for the data to be processed (thresholded shots)
    :param observables: measurement observables, see docstring of hlp_mod.
        get_observables. If not provided, it will default to hlp_mod.
        get_observables. See required input params there.
    :param keys_out: list of key names or dictionary keys paths in
        data_dict for the processed data to be saved into
    :param params: keyword arguments:
        Expects to find either in data_dict or in params:
            - cal_points:  repr of instance of CalibrationPoints
            - preparation_params: preparation parameters as stored in
                QuDev_transmon or Device. Used in CalibrationPoints.get_indices()
            - meas_obj_names: list of measurement object names
            - probability_table if it is not in data_dict; see
                calculate_probability_table for details.
            - n_readouts: the total number of readouts including
                preselection.
    :return: adds to data_dict:
        - the measurement operators corresponding to each observable and the
            expected covariance matrix between the operators under keys_out
        - if keys_out is None, it will saves the aforementioned quantities
            under  'measurement_ops' and 'cov_matrix_meas_obs'

    Assumptions:
        - all qubits in CalibrationPoints have the same cal point indices
    """
    if keys_out is None:
        keys_out = ['measurement_ops', 'cov_matrix_meas_obs']
    if len(keys_out) != 2:
        raise ValueError(f'keys_out must have length 2. {len(keys_out)} '
                         f'entries were given.')

    prob_table = hlp_mod.get_param('probability_table', data_dict,
                                   raise_error=True, **params)
    prob_table = np.array(list(prob_table.values())).T
    cp = hlp_mod.get_measurement_properties(data_dict, props_to_extract=['cp'],
                                            **params)
    meas_obj_names = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['mobjn'], enforce_one_meas_obj=False,
        **params)
    prep_params = hlp_mod.get_param('preparation_params', data_dict, **params)

    try:
        preselection_obs_idx = list(observables.keys()).index('pre')
    except ValueError:
        preselection_obs_idx = None
    observable_idxs = [i for i in range(len(observables))
                       if i != preselection_obs_idx]

    # calculate the mean for each reference state and each observable
    cp_indices = cp.get_indices(meas_obj_names, prep_params)
    cal_readouts = hlp_mod.flatten_list(
        [cp_indices[list(cp_indices)[0]].get(state, [])
         for state in ['g', 'e', 'f', 'h']])
    means = np.array([np.mean([prob_table[cal_idx][observable_idxs]], axis=0)
                      for cal_idx in cal_readouts])
    # normalize the assignment matrix
    for r in range(means.shape[0]):
        means[r] /= means[r].sum()
    Fs = [np.diag(ms) for ms in means.T]

    # find the means for all the products of the operators and the average
    # covariation of the operators
    observables_data ={k: v for k, v in observables.items() if k != 'pre'}
    n_readouts = hlp_mod.get_param('n_readouts', data_dict, raise_error=True,
                                   **params)
    prod_obss = OrderedDict()
    prod_obs_idxs = {}
    obs_products = np.zeros([n_readouts] + [len(observables_data)]*2)
    for i, obs in enumerate(observables_data):
        obsi = observables_data[obs]
        for j, obsj in enumerate(observables_data.values()):
            if i > j:
                continue
            obsp = hlp_mod.observable_product(obsi, obsj)
            if obsp is None:
                obs_products[:, i, j] = 0
                obs_products[:, j, i] = 0
            else:
                prod_obs_idxs[(i, j)] = len(prod_obss)
                prod_obs_idxs[(j, i)] = len(prod_obss)
                prod_obss[obs] = obsp

    prod_prob_table = calculate_probability_table(data_dict, keys_in=keys_in,
                                                  observables=prod_obss,
                                                  n_readouts=n_readouts)
    prod_prob_table = np.array(list(prod_prob_table.values())).T
    for (i, j), k in prod_obs_idxs.items():
        obs_products[:, i, j] = prod_prob_table[:, k]
    Omega = -np.array([np.outer(ro, ro) for ro in
                        prob_table[:, observable_idxs]]) + obs_products
    Omega = np.mean(Omega[list(cal_readouts)], 0)

    hlp_mod.add_param(keys_out[0], Fs, data_dict, **params)
    hlp_mod.add_param(keys_out[1], Omega, data_dict, **params)


def count_states(data_dict, keys_in, keys_out, states=None, n_meas_objs=None,
                 **params):
    """
    Averages data in data_dict specified by keys_in into num_bins.
    :param data_dict: OrderedDict containing data to be processed and where
                    processed data is to be stored
    :param keys_in: list of key names or dictionary keys paths in
                    data_dict for the data to be processed
    :param keys_out: list of key names or dictionary keys paths in
                    data_dict for the processed data to be saved into
    :param states: list of tuples of 0, 1, (2 if qutrits) denoting qubit/qutrit
        basis states to match
    :param n_meas_objs: number of measurement objects. If this is given and
        states is None, all possible basis states for n_meas_objs qubits will
        be generated
    :param params: keyword arguments

    Assumptions:
        - each array pointed to by keys_in is assumed to correspond to one of
            the qubits/qutrits in the basis states. IMPORTANT TO SPECIFY
            KEYS_IN IN THE CORRECT ORDER, SUCH THAT ARRAYS CORRESPOND TO YOUR
            STATE
        - the data pointed to by keys_in is assumed to be 1D arrays of
            thresholded/classified shots
        - len(keys_out) == 1
    """

    if len(keys_out) != 1:
        raise ValueError(f'keys_out must have length one. {len(keys_out)} '
                         f'entries were given.')
    if states is None:
        if n_meas_objs is None:
            raise ValueError('Please specify either states or n_meas_objs.')
        states = itertools.product((0, 1), repeat=n_meas_objs)

    data_to_proc_dict = hlp_mod.get_data_to_process(data_dict, keys_in)
    shots = np.array(list(data_to_proc_dict.values()))
    state_counts = np.array([
        np.count_nonzero(np.all(shots.transpose() == state, axis=1))
        for state in states])
    hlp_mod.add_param(
        keys_out[0], state_counts,
        data_dict, update_value=params.get('update_value', False), **params)