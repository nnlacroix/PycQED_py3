"""
File containing the BaseDataAnalyis class.
"""
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
            process_data(self.data_dict, self.processing_pipe)
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
            {'exp_metadata':'Experimental Data.Experimental Metadata',
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
        self.data_dict = add_measured_data(self.data_dict)

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
        # call get_params_from_hdf_file which gets values for params
        # in self.params_dict and adds them to the dictionary
        # raw_data_dict_ts
        hlp_mod.get_params_from_hdf_file(
            raw_data_dict, params_dict=self.params_dict,
            numeric_params=self.numeric_params,
            folder=folder)
        return raw_data_dict


class PipelineDataAnalysis_multi_timestamp(object):
    def __init__(self, data_dict: dict = None,
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
        self.timestamps = None
        if data_file_path is None:
            if t_start is None:
                if isinstance(label, list):
                    self.timestamps = [a_tools.latest_data(
                        contains=lab, return_timestamp=True)[0] for lab in label]
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
            process_data(self.data_dict, self.processing_pipe)
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
            {'exp_metadata':'Experimental Data.Experimental Metadata',
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
        self.data_dict = add_measured_data(self.data_dict)

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

            # call get_params_from_hdf_file which gets values for params
            # in self.params_dict and adds them to the dictionary
            # raw_data_dict_ts
            hlp_mod.get_params_from_hdf_file(
                raw_data_dict, params_dict=self.params_dict,
                numeric_params=self.numeric_params,
                folder=folder, append_key=False, update_key=True)
        return raw_data_dict


def process_data(data_dict, processing_pipe):
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
    if len(processing_pipe) == 0:
        raise ValueError('Experimental metadata or options_dict must '
                         'contain "processing_pipe."')

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


def add_measured_data(raw_data_dict):
    if 'measured_data' in raw_data_dict and \
            'value_names' in raw_data_dict['exp_metadata']:
        value_names = raw_data_dict['exp_metadata']['value_names']
        measured_data = raw_data_dict.pop('measured_data')
        data = measured_data[-len(value_names):]
        if data.shape[0] != len(value_names):
            raise ValueError('Shape mismatch between data and ro channels.')

        TwoD = hlp_mod.get_param('TwoD', raw_data_dict,
                                       default_value=False)
        sweep_points = measured_data[:-len(value_names)]
        for i, ro_ch in enumerate(value_names):
            if sweep_points.shape[0] > 1 and TwoD:
                hsl = len(np.unique(sweep_points[0]))
                ssl = len(np.unique(sweep_points[1:], axis=1)[0])
                measured_data = np.reshape(data[i], (ssl, hsl)).T
            else:
                measured_data = data[i]
            raw_data_dict[ro_ch] = measured_data
    else:
        raise ValueError('"measured_data" was not added.')
    return raw_data_dict