import logging
log = logging.getLogger(__name__)
import re
import os
import h5py
import numpy as np
from copy import deepcopy
from collections import OrderedDict
from more_itertools import unique_everseen
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.measurement.hdf5_data import read_dict_from_hdf5
from pycqed.measurement.calibration.calibration_points import CalibrationPoints


def get_hdf_param_value(group, param_name):
    '''
    Returns an attribute "key" of the group "Experimental Data"
    in the hdf5 datafile.
    '''
    s = group.attrs[param_name]
    # converts byte type to string because of h5py datasaving
    if type(s) == bytes:
        s = s.decode('utf-8')
    # If it is an array of value decodes individual entries
    if type(s) == np.ndarray:
        s = [s.decode('utf-8') for s in s]
    try:
        return eval(s)
    except Exception:
        return s


def get_channel_names_from_timestamp(timestamp):
    folder = a_tools.get_folder(timestamp)
    h5filepath = a_tools.measurement_filename(folder)
    data_file = h5py.File(h5filepath, 'r+')
    try:
        channel_names = get_hdf_param_value(data_file['Experimental Data'],
                                            'value_names')
        data_file.close()
        return channel_names
    except Exception as e:
        data_file.close()
        raise e


def get_sweep_points_from_timestamp(timestamp):
    folder = a_tools.get_folder(timestamp)
    h5filepath = a_tools.measurement_filename(folder)
    data_file = h5py.File(h5filepath, 'r+')
    try:
        group = data_file['Experimental Data'][
            'Experimental Metadata']['sweep_points']
        sweep_points = OrderedDict()
        sweep_points = read_dict_from_hdf5(sweep_points, group)
        data_file.close()
        return sweep_points
    except Exception as e:
        data_file.close()
        raise e


def get_params_from_hdf_file(data_dict, **params):
    params_dict = get_param('params_dict', data_dict, **params)
    numeric_params = get_param('numeric_params', data_dict,
                               default_value=[], **params)
    append_key = get_param('append_key', data_dict, default_value=True,
                           **params)
    update_key = get_param('update_key', data_dict, default_value=False,
                           **params)

    if params_dict is None:
        raise ValueError('params_dict was not specified.')

    # if folder is not specified, will take the last folder in the list
    folder = params.get('folders', data_dict.get('folders', None))
    if folder is None:
        raise ValueError('No folder was found.')
    else:
        folder = folder[-1]
    h5mode = get_param('h5mode', data_dict, default_value='r+', **params)
    h5filepath = a_tools.measurement_filename(folder)
    data_file = h5py.File(h5filepath, h5mode)

    try:
        if 'measurementstrings' in params_dict:
            # assumed data_dict['measurementstrings'] is a list
            if 'measurementstrings' in data_dict:
                data_dict['measurementstrings'] += [os.path.split(folder)[1][7:]]
            else:
                data_dict['measurementstrings'] = [os.path.split(folder)[1][7:]]
        if 'measured_data' in params_dict:
            if 'measured_data' in data_dict:
                data_dict['measured_data'] = np.concatenate(
                    (data_dict['measured_data'],
                     np.array(data_file['Experimental Data']['Data']).T), axis=1)
            else:
                data_dict['measured_data'] = np.array(
                    data_file['Experimental Data']['Data']).T

        for save_par, file_par in params_dict.items():
            epd = data_dict
            all_keys = save_par.split('.')
            for i in range(len(all_keys)-1):
                if all_keys[i] not in epd:
                    epd[all_keys[i]] = OrderedDict()
                else:
                    epd = epd[all_keys[i]]

            if len(file_par.split('.')) == 1:
                par_name = file_par.split('.')[0]
                for group_name in data_file.keys():
                    if par_name in list(data_file[group_name].attrs):
                        add_param(all_keys[-1],
                                  get_hdf_param_value(data_file[group_name],
                                                      par_name),
                                  epd, append_key=append_key, update_key=update_key)
            else:
                group_name = '/'.join(file_par.split('.')[:-1])
                par_name = file_par.split('.')[-1]
                if group_name in data_file:
                    if par_name in list(data_file[group_name].attrs):
                        add_param(all_keys[-1],
                                  get_hdf_param_value(data_file[group_name],
                                                      par_name),
                                  epd, append_key=append_key, update_key=update_key)
                    elif par_name in list(data_file[group_name].keys()):
                        add_param(all_keys[-1],
                                  read_dict_from_hdf5(
                                      {}, data_file[group_name][par_name]),
                                  epd, append_key=append_key, update_key=update_key)

            if all_keys[-1] not in epd:
                log.warning(f'Parameter {file_par} was not found.')
                epd[all_keys[-1]] = 0
        data_file.close()
    except Exception as e:
        data_file.close()
        raise e

    for par_name in data_dict:
        if par_name in numeric_params:
            if hasattr(data_dict[par_name], '__iter__'):
                data_dict[par_name] = [np.double(p) for p
                                       in data_dict[par_name]]
                data_dict[par_name] = np.asarray(data_dict[par_name])
            else:
                data_dict[par_name] = np.double(data_dict[par_name])

    return data_dict


def get_data_to_process(data_dict, keys_in):
    """
    Finds data to be processed in unproc_data_dict based on keys_in.

    :param data_dict: OrderedDict containing data to be processed
    :param keys_in: list of channel names or dictionary paths leading to
            data to be processed. For example: raw w1, filtered_data.raw w0
    :return:
        data_to_proc_dict: dictionary {ch_in: data_ch_in}
    """
    data_to_proc_dict = OrderedDict()
    key_found = True
    for keyi in keys_in:
        all_keys = keyi.split('.')
        if len(all_keys) == 1:
            try:
                if isinstance(data_dict[all_keys[0]], dict):
                    data_to_proc_dict = {f'{keyi}.{k}': deepcopy(v) for k, v
                                         in data_dict[all_keys[0]].items()}
                else:
                    data_to_proc_dict[keyi] = data_dict[all_keys[0]]
            except KeyError:
                key_found = False
        else:
            try:
                data = data_dict
                for k in all_keys:
                    data = data[k]
                if isinstance(data, dict):
                    data_to_proc_dict = {f'{keyi}.{k}': deepcopy(data[k])
                                         for k in data}
                else:
                    data_to_proc_dict[keyi] = deepcopy(data)
            except KeyError:
                key_found = False
        if not key_found:
            raise ValueError(f'Channel {keyi} was not found.')
    return data_to_proc_dict


def get_param(name, data_dict, default_value=None, raise_error=False, **params):
    p = params
    md = data_dict.get('exp_metadata', dict())
    dd = data_dict
    all_keys = name.split('.')
    if len(all_keys) > 1:
        for i in range(len(all_keys)-1):
            if all_keys[i] not in p:
                p[all_keys[i]] = OrderedDict()
            if all_keys[i] not in md:
                md[all_keys[i]] = OrderedDict()
            if all_keys[i] not in dd:
                dd[all_keys[i]] = OrderedDict()
            p = p[all_keys[i]]
            md = md[all_keys[i]]
            dd = dd[all_keys[i]]
    value = p.get(all_keys[-1],
                  dd.get(all_keys[-1],
                         md.get(all_keys[-1], default_value)
                         )
                  )
    if raise_error and value is None:
        raise ValueError(f'{name} was not found in either data_dict, or '
                         f'exp_metadata or input params.')
    return value


def add_param(name, value, data_dict, update_key=False, append_key=False,
              **params):
    """
    Adds a new key-value pair to the data_dict, with key = name.
    If update, it will try data_dict[name].update(value), else raises KeyError.
    :param name: key of the new parameter in the data_dict
    :param value: value of the new parameter
    :param data_dict: OrderedDict containing data to be processed
    :param update: whether to try data_dict[name].update(value)
    :param params: keyword arguments
    :return:
    """

    dd = data_dict
    all_keys = name.split('.')
    if len(all_keys) > 1:
        for i in range(len(all_keys)-1):
            if all_keys[i] not in dd:
                dd[all_keys[i]] = OrderedDict()
            dd = dd[all_keys[i]]

    if all_keys[-1] in dd:
        if update_key:
            if isinstance(value, dict):
                dd[all_keys[-1]].update(value)
            else:
                dd[all_keys[-1]] = value
        elif append_key:
            v = dd[all_keys[-1]]
            dd[all_keys[-1]] = [v]
            dd[all_keys[-1]].extend([value])

        else:
            raise KeyError(f'{all_keys[-1]} already exists in data_dict.')
    else:
        dd[all_keys[-1]] = value


def get_measobj_properties(data_dict, props_to_extract='all', **params):
    """
    Extracts cal_points, sweep_points, meas_obj_sweep_points_map and
    meas_obj_names from experiment metadata or from params.
    :param data_dict: OrderedDict containing experiment metadata (exp_metadata)
    :param params: keyword arguments
        enforce_one_meas_obj (default True): checks if meas_obj_names contains
            more than one element. If True, raises an error, else returns
            meas_obj_names[0].
    :return: cal_points, sweep_points, meas_obj_sweep_points_map and
    meas_obj_names

    Assumptions:
        - if cp or sp are strings, then it assumes they can be evaluated
    """
    if props_to_extract == 'all':
        props_to_extract = ['cp', 'sp', 'mospm', 'movnm', 'mobjn']

    props_to_return = []
    if 'cp' in props_to_extract:
        cp = get_param('cal_points', data_dict, raise_error=True, **params)
        if isinstance(cp, str):
            cp = CalibrationPoints.from_string(cp)
        props_to_return += [cp]
    if 'sp' in props_to_extract:
        sp = get_param('sweep_points', data_dict, raise_error=True, **params)
        if isinstance(sp, str):
            sp = eval(sp)
        props_to_return += [sp]
    if 'mospm' in props_to_extract:
        meas_obj_sweep_points_map = get_param(
            'meas_obj_sweep_points_map', data_dict, raise_error=True, **params)
        props_to_return += [meas_obj_sweep_points_map]
    if 'movnm' in props_to_extract:
        meas_obj_value_names_map = get_param(
            'meas_obj_value_names_map', data_dict, raise_error=True, **params)
        props_to_return += [meas_obj_value_names_map]
    if 'mobjn' in props_to_extract:
        mobjn = get_param('meas_obj_names', data_dict,
                          raise_error=True, **params)
        if params.get('enforce_one_meas_obj', True):
            if isinstance(mobjn, list):
                if len(mobjn) > 1:
                    raise ValueError(f'This node expects one measurement '
                                     f'object, {len(mobjn)} were given.')
                else:
                    mobjn = mobjn[0]
        props_to_return += [mobjn]
    if len(props_to_return) == 1:
        props_to_return = props_to_return[0]
    return props_to_return


def get_qb_channel_map_from_file(data_dict, data_keys, **params):
    file_type = params.get('file_type', 'hdf')
    qb_names = get_param('qb_names', data_dict, **params)
    if qb_names is None:
        raise ValueError('Either channel_map or qb_names must be specified.')

    folder = get_param('folders', data_dict, **params)[-1]
    if folder is None:
        raise ValueError('Path to file must be saved in '
                         'data_dict[folders] in order to extract '
                         'channel_map.')

    if file_type == 'hdf':
        qb_channel_map = a_tools.get_qb_channel_map_from_hdf(
            qb_names, value_names=data_keys, file_path=folder)
    else:
        raise ValueError('Only "hdf" files supported at the moment.')
    return qb_channel_map


## Helper functions ##
def get_msmt_data(all_data, cal_points, qb_name):
    """
    Extracts data points from all_data that correspond to the measurement
    points (without calibration points data).
    :param all_data: array containing both measurement and calibration
                     points data
    :param cal_points: CalibrationPoints instance or its repr
    :param qb_name: qubit name
    :return: measured data without calibration points data
    """
    if cal_points is None:
        return all_data

    if isinstance(cal_points, str):
        cal_points = repr(cal_points)
    if qb_name in cal_points.qb_names:
        n_cal_pts = len(cal_points.get_states(qb_name)[qb_name])
        if n_cal_pts == 0:
            return all_data
        else:
            return deepcopy(all_data[:-n_cal_pts])
    else:
        return all_data


def get_cal_data(all_data, cal_points, qb_name):
    """
    Extracts data points from all_data that correspond to the calibration points
    data.
    :param all_data: array containing both measurement and calibration
                     points data
    :param cal_points: CalibrationPoints instance or its repr
    :param qb_name: qubit name
    :return: Calibration points data
    """
    if cal_points is None:
        return np.array([])

    if isinstance(cal_points, str):
        cal_points = repr(cal_points)
    if qb_name in cal_points.qb_names:
        n_cal_pts = len(cal_points.get_states(qb_name)[qb_name])
        if n_cal_pts == 0:
            return np.array([])
        else:
            return deepcopy(all_data[-n_cal_pts:])
    else:
        return np.array([])


def get_cal_sweep_points(sweep_points_array, cal_points, qb_name):
    """
    Creates the sweep points corresponding to the calibration points data as
    equally spaced number_of_cal_states points, with the spacing given by the
    spacing in sweep_points_array.
    :param sweep_points_array: array of physical sweep points
    :param cal_points: CalibrationPoints instance or its repr
    :param qb_name: qubit name
    """
    if cal_points is None:
        return np.array([])

    if isinstance(cal_points, str):
        cal_points = repr(cal_points)
    if qb_name in cal_points.qb_names:
        n_cal_pts = len(cal_points.get_states(qb_name)[qb_name])
        if n_cal_pts == 0:
            return np.array([])
        else:
            try:
                step = np.abs(sweep_points_array[-1] - sweep_points_array[-2])
            except IndexError:
                # This fallback is used to have a step value in the same order
                # of magnitude as the value of the single sweep point
                step = np.abs(sweep_points_array[0])
            return np.array([sweep_points_array[-1] + i * step for
                             i in range(1, n_cal_pts + 1)])
    else:
        return np.array([])


## Plotting nodes ##
def get_cal_state_color(cal_state_label):
    if cal_state_label == 'g' or cal_state_label == r'$|g\rangle$':
        return 'k'
    elif cal_state_label == 'e' or cal_state_label == r'$|e\rangle$':
        return 'gray'
    elif cal_state_label == 'f' or cal_state_label == r'$|f\rangle$':
        return 'C8'
    else:
        return 'C4'


def get_latex_prob_label(prob_label):
    if 'pg ' in prob_label.lower():
        return r'$|g\rangle$ state population'
    elif 'pe ' in prob_label.lower():
        return r'$|e\rangle$ state population'
    elif 'pf ' in prob_label.lower():
        return r'$|f\rangle$ state population'
    else:
        return prob_label


def flatten_list(lst_of_lsts):
    """
    Flattens the list of lists lst_of_lsts.
    :param lst_of_lsts: a list of lists
    :return: flattened list
    """
    if all([isinstance(e, list) for e in lst_of_lsts]):
        return [e for l1 in lst_of_lsts for e in l1]
    elif any([isinstance(e, list) for e in lst_of_lsts]):
        l = []
        for e in lst_of_lsts:
            if isinstance(e, list):
                l.extend(e)
            else:
                l.append(e)
        return l
    else:
        return lst_of_lsts


def get_sublst_with_all_strings_of_list(lst_to_search, lst_to_match):
    """
    Finds all string elements in lst_to_search that contain the
    string elements of lst_to_match.
    :param lst_to_search: list of strings to search
    :param lst_to_match: list of strings to match
    :return: list of strings from lst_to_search that contain all string
    elements in lst_to_match
    """
    lst_w_matches = []
    for etm in lst_to_match:
        for ets in lst_to_search:
            r = re.search(etm, ets)
            if r is not None:
                lst_w_matches += [ets]
    return list(unique_everseen(lst_w_matches))