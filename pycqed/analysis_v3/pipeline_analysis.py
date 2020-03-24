"""
File containing the BaseDataAnalyis class.
"""
import h5py
import numpy as np
from collections import OrderedDict
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.analysis_v3 import helper_functions as hlp_mod
from pycqed.analysis_v3 import data_processing as dat_proc
from pycqed.analysis_v3 import fitting as fit_module
from pycqed.analysis_v3 import plotting as plot_module
import copy
import logging
log = logging.getLogger(__name__)


class PipelineDataAnalysis(object):
    def __init__(self, data_dict: dict = None,
                 timestamp: str = None, label: str = '',
                 data_file_path: str = None, options_dict: dict = None,
                 auto=True, params_dict=dict(), numeric_params=dict(),
                 **params):
        '''
        There are several ways to specify where the data should be loaded
        from.
        :param t_start, t_stop: give a range of timestamps in where data is
                                loaded from. Filtering options can be given
                                through the options dictionary. If t_stop is
                                omitted, the extraction routine looks for
                                the data with time stamp t_start.
        :param label: Only process datasets with this label.
        :param data_file_path: directly give the file path of a data file that
                                should be loaded. Note: data_file_path has
                                priority, i.e. if this argument is given time
                                stamps are ignored.
        :param options_dict: available options are:
                                -'processing_pipe'
                                -'exp_metadata'
        '''
        self.data_dict = data_dict
        self.params_dict = params_dict
        self.numeric_params = numeric_params

        if options_dict is None:
            self.options_dict = OrderedDict()
        else:
            self.options_dict = options_dict

        ################################################
        # These options determine what data to extract #
        ################################################
        self.timestamps = None
        if data_file_path is None:
            if timestamp is None:
                self.timestamps = [a_tools.latest_data(
                    contains=label, return_timestamp=True)[0]]
            else:
                self.timestamps = [timestamp]

        if self.timestamps is None or len(self.timestamps) == 0:
            raise ValueError('No data file found.')

        if auto:
            self.run_analysis()

    def run_analysis(self):
        """
        This function is at the core of all analysis and defines the flow.
        This function is typically called after the __init__.
        """
        if self.data_dict is None:
            self.extract_data()  # extract data specified in params dict
        if len(self.processing_pipe) > 0:
            process_pipe(self.data_dict, self.processing_pipe)
        else:
            print('There is no data processing pipe.')

    def extract_data(self):
        """
        Extracts the data specified in
            self.params_dict
            self.numeric_params
        from each timestamp in self.timestamps and stores it into
        self.data_dict
        """
        self.data_dict = OrderedDict()
        self.params_dict.update(
            {'exp_metadata': 'Experimental Data.Experimental Metadata',
             'exp_metadata.sweep_parameter_names': 'sweep_parameter_names',
             'exp_metadata.sweep_parameter_units': 'sweep_parameter_units',
             'exp_metadata.value_names': 'value_names',
             'exp_metadata.value_units': 'value_units',
             'measurementstrings': 'measurementstring',
             'measured_data': 'measured_data'})

        self.data_dict.update(self.get_data_from_timestamp_list())
        self.metadata = self.data_dict.get('exp_metadata', {})
        self.metadata.update(self.get_param_value('exp_metadata', {}))
        self.data_dict['exp_metadata'] = self.metadata
        self.data_dict = add_measured_data_dict(self.data_dict)

        self.processing_pipe = self.get_param_value('processing_pipe')
        self.data_dict.update(self.options_dict)
        if self.processing_pipe is None:
            self.processing_pipe = []

    def get_param_value(self, param_name, default_value=None):
        return self.options_dict.get(param_name, self.metadata.get(
            param_name, default_value))

    def get_data_from_timestamp_list(self):
        raw_data_dict = OrderedDict()
        raw_data_dict['timestamps'] = self.timestamps
        folder = a_tools.get_folder(self.timestamps[0])
        raw_data_dict['folders'] = [folder]
        add_measured_data_hdf(raw_data_dict, folder)
        # call get_params_from_hdf_file which gets values for params
        # in self.params_dict and adds them to the dictionary
        # raw_data_dict_ts
        hlp_mod.get_params_from_hdf_file(
            raw_data_dict, params_dict=self.params_dict,
            numeric_params=self.numeric_params,
            folder=folder)
        return raw_data_dict


class PipelineDataAnalysis_multi_timestamp(object):
    def __init__(self, data_dict: dict = None, timestamps: str = None,
                 t_start: str = None, t_stop: str = None,
                 label: str = '', data_file_path: str = None,
                 options_dict: dict = None, auto=True, params_dict=dict(),
                 numeric_params=dict(), **params):

        self.data_dict = data_dict
        self.params_dict = params_dict
        self.numeric_params = numeric_params

        if options_dict is None:
            self.options_dict = OrderedDict()
        else:
            self.options_dict = options_dict

        ################################################
        # These options determine what data to extract #
        ################################################
        self.timestamps = timestamps
        if self.timestamps is None:
            if data_file_path is None:
                if t_start is None:
                    if isinstance(label, list):
                        self.timestamps = [a_tools.latest_data(
                            contains=l, return_timestamp=True)[0]
                                           for l in label]
                    else:
                        self.timestamps = [a_tools.latest_data(
                            contains=label, return_timestamp=True)[0]]
                elif t_stop is None:
                    if isinstance(t_start, list):
                        self.timestamps = t_start
                    else:
                        self.timestamps = [t_start]
                else:
                    self.timestamps = a_tools.get_timestamps_in_range(
                        t_start, timestamp_end=t_stop,
                        label=label if label != '' else None)

            if self.timestamps is None or len(self.timestamps) == 0:
                raise ValueError('No data file found.')

        if auto:
            self.run_analysis()

    def run_analysis(self):
        """
        This function is at the core of all analysis and defines the flow.
        This function is typically called after the __init__.
        """
        if self.data_dict is None:
            self.extract_data()  # extract data specified in params dict
        if len(self.processing_pipe) > 0:
            process_pipe(self.data_dict, self.processing_pipe)
        else:
            print('There is no data processing pipe.')

    def extract_data(self):
        """
        Extracts the data specified in
            self.params_dict
            self.numeric_params
        from each timestamp in self.timestamps and stores it into
        self.data_dict
        """
        self.data_dict = OrderedDict()
        self.params_dict.update(
            {'exp_metadata': 'Experimental Data.Experimental Metadata',
             'exp_metadata.sweep_parameter_names': 'sweep_parameter_names',
             'exp_metadata.sweep_parameter_units': 'sweep_parameter_units',
             'exp_metadata.value_names': 'value_names',
             'exp_metadata.value_units': 'value_units',
             'measurementstrings': 'measurementstring',
             'measured_data': 'measured_data'})

        self.data_dict = self.get_data_from_timestamp_list()
        self.metadata = self.data_dict.get('exp_metadata', {})
        self.metadata.update(self.get_param_value('exp_metadata', {}))
        self.data_dict['exp_metadata'] = self.metadata
        self.data_dict = add_measured_data_dict(self.data_dict)

        self.processing_pipe = self.get_param_value('processing_pipe')
        self.data_dict.update(self.options_dict)
        if self.processing_pipe is None:
            self.processing_pipe = []

    def get_param_value(self, param_name, default_value=None):
        return self.options_dict.get(param_name, self.metadata.get(
            param_name, default_value))

    def get_data_from_timestamp_list(self):
        raw_data_dict = OrderedDict()
        raw_data_dict['timestamps'] = []
        raw_data_dict['folders'] = []

        for i, timestamp in enumerate(self.timestamps):
            folder = a_tools.get_folder(timestamp)
            raw_data_dict['timestamps'] += [timestamp]
            raw_data_dict['folders'] += [folder]

            add_measured_data_hdf(raw_data_dict, folder, append_data=True)
            # call get_params_from_hdf_file which gets values for params
            # in self.params_dict and adds them to the dictionary
            # raw_data_dict_ts
            hlp_mod.get_params_from_hdf_file(
                raw_data_dict, params_dict=self.params_dict,
                numeric_params=self.numeric_params, folder=folder,
                append_value=False, update_value=True, replace_value=True)
        return raw_data_dict


def get_timestamps(data_dict=OrderedDict(), t_start=None, t_stop=None,
                   label='', data_file_path=None, **params):
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
        timestamps = hlp_mod.get_param('timestamps', data_dict,
                                       raise_error=True)
    if isinstance(timestamps, str):
        timestamps = [timestamps]
        hlp_mod.add_param('timestamps', timestamps, data_dict, replace_value=True)

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


def process_pipe(data_dict, processing_pipe=None):
    """
    Calls all the classes/functions found in metadata[
    'processing_pipe'], which is a list of dictionaries of the form:

    [
        {'node_type': obj0_name, **kw},
        {'node_type': obj1_name, **kw},
    ]

    These classes all live in the data_processing.py module, and will
    process the data corresponding to the channels passed in as kwargs.

    Each node in the pipeline will put the processed data in the data_dict,
    under the key/dictionary keys path specified in 'chs_out' in the
    **kw of each node.
    """
    if processing_pipe is None:
        processing_pipe = hlp_mod.get_param('processing_pipe', data_dict,
                                            raise_error=True)
    else:
        hlp_mod.add_param('processing_pipe', processing_pipe, data_dict,
                          replace_value=True)

    for node_dict in processing_pipe:
        node = None
        for module in [dat_proc, plot_module, fit_module]:
            try:
                node = getattr(module, node_dict["node_type"])
                break
            except AttributeError:
                continue
        if node is None:
            raise KeyError(f'Processing node "{node_dict["node_type"]}" '
                           f'not recognized')
        node(data_dict, **node_dict)