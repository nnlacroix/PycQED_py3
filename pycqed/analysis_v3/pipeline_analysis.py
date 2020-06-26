"""
File containing the BaseDataAnalyis class.
"""
import h5py
import numpy as np
from collections import OrderedDict

import logging
log = logging.getLogger(__name__)

# analysis_v3 node modules
from pycqed.analysis_v3 import saving as save_module
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.analysis_v3 import helper_functions as hlp_mod
from pycqed.analysis_v3.processing_pipeline import ProcessingPipeline

search_modules = set()
search_modules.add(hlp_mod)


def get_timestamps(data_dict=None, t_start=None, t_stop=None,
                   label='', data_file_path=None, **params):
    # if i put data_dict = OrderedDict() in the input params, somehow this
    # function sees the data_dict i have in my notebook. How???
    if data_dict is None:
        data_dict = OrderedDict()

    timestamps = None
    if data_file_path is None:
        if t_start is None:
            if isinstance(label, list):
                timestamps = [a_tools.latest_data(
                    contains=l, return_timestamp=True, **params)[0]
                              for l in label]
            else:
                timestamps = [a_tools.latest_data(
                    contains=label, return_timestamp=True, **params)[0]]
        elif t_stop is None:
            if isinstance(t_start, list):
                timestamps = t_start
            else:
                timestamps = [t_start]
        else:
            timestamps = a_tools.get_timestamps_in_range(
                t_start, timestamp_end=t_stop,
                label=label if label != '' else None, **params)

    if timestamps is None or len(timestamps) == 0:
        raise ValueError('No data file found.')

    data_dict['timestamps'] = timestamps
    return data_dict


def extract_data_hdf(timestamps=None, data_dict=None,
                     params_dict=OrderedDict(), numeric_params=OrderedDict(),
                     append_data=False, replace_data=False, **params):
    """
    Extracts the data specified in
        params_dict
        pumeric_params
    from each timestamp in timestamps and stores it into
    data_dict
    """
    if data_dict is None:
        data_dict = OrderedDict()

    if timestamps is None:
        timestamps = hlp_mod.get_param('timestamps', data_dict)
    if timestamps is None:
        get_timestamps(data_dict, **params)
        timestamps = hlp_mod.get_param('timestamps', data_dict)
    if isinstance(timestamps, str):
        timestamps = [timestamps]
        hlp_mod.add_param('timestamps', timestamps, data_dict,
                          replace_value=True)

    data_dict['folders'] = []
    for i, timestamp in enumerate(timestamps):
        folder = a_tools.get_folder(timestamp)
        data_dict['folders'] += [folder]
        add_measured_data_hdf(data_dict, folder, append_data, replace_data)

    params_dict_temp = params_dict
    params_dict = OrderedDict(
        {'exp_metadata': 'Experimental Data.Experimental Metadata',
         'exp_metadata.sweep_parameter_names': 'sweep_parameter_names',
         'exp_metadata.sweep_parameter_units': 'sweep_parameter_units',
         'exp_metadata.value_names': 'value_names',
         'exp_metadata.value_units': 'value_units',
         'measurementstrings': 'measurementstring'})
    params_dict.update(params_dict_temp)
    # call get_params_from_hdf_file which gets values for params
    # in params_dict and adds them to the dictionary data_dict
    hlp_mod.get_params_from_hdf_file(
        data_dict, params_dict=params_dict, numeric_params=numeric_params,
        folder=data_dict['folders'][-1],
        append_value=False, update_value=True, replace_value=True)
    add_measured_data_dict(data_dict)

    metadata = hlp_mod.get_param('exp_metadata', data_dict,
                                  default_value=OrderedDict())
    data_dict['exp_metadata'].update(metadata)
    return data_dict


def add_measured_data_hdf(data_dict, folder=None, append_data=False,
                          replace_data=False, **params):

    if folder is None:
        folder = hlp_mod.get_param('folders', data_dict, raise_error=True,
                                   **params)
        if len(folder) > 0:
            folder = folder[-1]

    h5mode = hlp_mod.get_param('h5mode', data_dict, default_value='r+',
                               **params)
    h5filepath = a_tools.measurement_filename(folder)
    data_file = h5py.File(h5filepath, h5mode)
    meas_data_array = np.array(data_file['Experimental Data']['Data']).T
    if 'measured_data' in data_dict:
        if replace_data:
            data_dict['measured_data'] = meas_data_array
        elif append_data:
            data_dict['measured_data'] = np.concatenate(
                (data_dict['measured_data'], meas_data_array), axis=1)
        else:
            raise ValueError('Both "append_data" and "replace_data" are False. '
                             'Unclear how to add "measured_data" to data_dict.')
    else:
        data_dict['measured_data'] = meas_data_array
    return data_dict


def add_measured_data_dict(data_dict):
    metadata = hlp_mod.get_param('exp_metadata', data_dict, raise_error=True)
    if 'measured_data' in data_dict and 'value_names' in metadata:
        value_names = metadata['value_names']
        measured_data = data_dict.pop('measured_data')
        data = measured_data[-len(value_names):]
        if data.shape[0] != len(value_names):
            raise ValueError('Shape mismatch between data and ro channels.')

        TwoD = hlp_mod.get_param('TwoD', data_dict, default_value=False)
        sweep_points = measured_data[:-len(value_names)]
        for i, ro_ch in enumerate(value_names):
            if sweep_points.shape[0] > 1 and TwoD:
                hsl = len(np.unique(sweep_points[0]))
                ssl = len(np.unique(sweep_points[1:], axis=1)[0])
                measured_data = np.reshape(data[i], (ssl, hsl)).T
            else:
                measured_data = data[i]
            data_dict[ro_ch] = measured_data
    else:
        raise ValueError('"measured_data" was not added.')
    return data_dict


def process_pipeline(data_dict, processing_pipeline=None,
                     save_processed_data=True, save_figures=True, **params):
    """
    Calls all the classes/functions found in processing_pipeline,
    which is a list of dictionaries of the form:

    [
        {'node_name': function_name0, **node_params0},
        {'node_name': function_name1, **node_params1},
    ]

    All node functions must exist in the modules specified in the global vaiable
    "search_modules" define at the top of this module, and will process the
    data corresponding to the keys specified as "keys_in" in the **node_params
    of each node.

    Each node in the pipeline will put the processed data in the data_dict,
    under the key(s)/dictionary key path(s) specified in 'keys_out' in the
    the **node_params of each node.
    """
    if processing_pipeline is None:
        processing_pipeline = hlp_mod.get_param('processing_pipeline',
                                                data_dict, raise_error=True)
    else:
        proc_pipe_in_dd = hlp_mod.get_param('processing_pipeline', data_dict,
                                            default_value=[])
        for node_params in processing_pipeline:
            if node_params not in proc_pipe_in_dd:
                hlp_mod.add_param('processing_pipeline', [node_params],
                                  data_dict, append_value=True)

    # instantiate a ProcessingPipeline instance in case it is an ordinary list
    processing_pipeline = ProcessingPipeline(from_dict_list=processing_pipeline)
    # resolve pipeline in case it wasn't resolved yet
    movnm = hlp_mod.get_param('meas_obj_value_names_map', data_dict, **params)
    if movnm is not None:
        processing_pipeline(movnm)
    else:
        log.warning('Processing pipeline may not have been resolved.')

    for node_params in processing_pipeline:
        node = None
        for module in search_modules:
            try:
                node = getattr(module, node_params["node_name"])
                break
            except AttributeError:
                continue
        if node is None:
            raise KeyError(f'Node function "{node_params["node_name"]}" '
                           f'not recognized')
        node(data_dict, **node_params)

    if save_processed_data or save_figures:
        save_module.Save(data_dict, save_processed_data=save_processed_data,
                         save_figures=save_figures)