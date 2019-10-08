"""
File containing the BaseDataAnalyis class.
"""
import os
import numpy as np
from collections import OrderedDict
from matplotlib import pyplot as plt
import matplotlib as mpl
from pycqed.analysis import analysis_toolbox as a_tools
import json
import lmfit
import h5py
from pycqed.measurement.hdf5_data import read_dict_from_hdf5
from pycqed.analysis_v3 import helper_functions as help_func_mod
from pycqed.analysis_v3 import data_processing as dat_proc
from pycqed.analysis_v3 import fitting as fit_module
from pycqed.analysis_v3 import plotting as plot_module
import copy
import logging
log = logging.getLogger(__name__)


class PipelineDataAnalysis(object):
    """
    Abstract Base Class (not intended to be instantiated directly) for
    analysis.

    Children inheriting from this method should specify the following methods
        - __init__      -> specify params to be extracted, set options
                           specific to analysis and call run_analysis method.
        - process_data  -> mundane tasks such as binning and filtering
        - prepare_plots -> specify default plots and set up plotting dicts
        - run_fitting   -> perform fits to data

    The core of this class is the flow defined in run_analysis and should
    be called at the end of the __init__. This executes
    the following code:

        self.extract_data()    # extract data specified in params dict
        self.process_data()    # binning, filtering etc
        if self.do_fitting:
            self.run_fitting() # fitting to models
        self.prepare_plots()   # specify default plots
        if not self.extract_only:
            self.plot(key_list='auto')  # make the plots

    """

    fit_res = None
    '''
    Dictionary containing fitting objects
    '''
    fit_dict = None
    '''
    Dictionary containing fitting results
    '''

    def __init__(self, data_dict: dict = None,
                 t_start: str = None, t_stop: str = None,
                 label: str = '', data_file_path: str = None,
                 options_dict: dict = None, auto=True, params_dict=dict(),
                 numeric_params=dict(), **kwargs):
        '''
        This is the __init__ of the abstract base class.
        It is intended to be called at the start of the init of the child
        classes followed by "run_analysis".

        __init__ of the child classes:
            The __init__ of child classes  should implement the following
            functionality:
                - call the ASB __init__ (this method)
                - define self.params_dict and self.numeric_params
                - specify options specific to that analysis
                - call self.run_analysis


        This method sets several attributes of the analysis class.
        These include assigning the arguments of this function to attributes.
        Other arguments that get created are
            axs (dict)
            figs (dict)
            plot_dicts (dict)

        and a bunch of stuff specified in the options dict
        (TODO: should this not always be extracted from the
        dict to prevent double refs? )

        There are several ways to specify where the data should be loaded
        from.

        none of the below parameters: look for the last data which matches the
                filtering options from the options dictionary.

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
        :param close_figs: Close the figure (do not display)
        :param options_dict: available options are:
                                -'presentation_mode'
                                -'tight_fig'
                                -'plot_init'
                                -'save_figs'
                                -'close_figs'
                                -'verbose'
                                -'auto-keys'
                                -'twoD'
                                -'timestamp_end'
                                -'msmt_label'
                                -'do_individual_traces'
                                -'exact_label_match'
        :param extract_only: Should we also do the plots?
        :param do_fitting: Should the run_fitting method be executed?
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
            self.process_data()  # binning, filtering etc
        else:
            print('There is no data processing pipe.')

    def extract_data(self):
        """
        Extracts the data specified in
            self.params_dict
            self.numeric_params
        from each timestamp in self.timestamps
        and stores it into: self.raw_data_dict
        """

        self.params_dict.update(
            {'sweep_parameter_names': 'sweep_parameter_names',
             'sweep_parameter_units': 'sweep_parameter_units',
             'measurementstring': 'measurementstring',
             'value_names': 'value_names',
             'value_units': 'value_units',
             'measured_data': 'measured_data',
             'exp_metadata':
                 'Experimental Data.Experimental Metadata'})

        self.data_dict = self.get_data_from_timestamp_list()
        self.metadata = self.data_dict.get('exp_metadata', {})
        self.metadata.update(self.get_param_value('exp_metadata', {}))
        self.data_dict['exp_metadata'] = self.metadata

        if len(self.timestamps) == 1:
            self.data_dict = self.add_measured_data(
                self.data_dict)
        else:
            temp_dict_list = []
            for i, rd_dict in enumerate(self.data_dict):
                temp_dict_list.append(
                    self.add_measured_data(rd_dict))
            self.data_dict = tuple(temp_dict_list)

        self.processing_pipe = self.get_param_value('processing_pipe')
        if self.processing_pipe is None:
            self.processing_pipe = []

    def get_param_value(self, param_name, default_value=None):
        return self.options_dict.get(param_name, self.metadata.get(
            param_name, default_value))

    def get_data_from_timestamp_list(self):
        raw_data_dict = []
        for timestamp in self.timestamps:
            raw_data_dict_ts = OrderedDict()
            folder = a_tools.get_folder(timestamp)
            raw_data_dict_ts['timestamp'] = timestamp
            raw_data_dict_ts['folder'] = folder

            # call get_params_from_hdf_file which gets values for params
            # in self.params_dict and adds them to the dictionary
            # raw_data_dict_ts
            help_func_mod.get_params_from_hdf_file(
                raw_data_dict_ts, params_dict=self.params_dict,
                numeric_params=self.numeric_params,
                folder=folder)
            raw_data_dict.append(raw_data_dict_ts)

        if len(raw_data_dict) == 1:
            raw_data_dict = raw_data_dict[0]
        return raw_data_dict

    @staticmethod
    def add_measured_data(raw_data_dict):
        if 'measured_data' in raw_data_dict and \
                'value_names' in raw_data_dict:
            measured_data = raw_data_dict.pop('measured_data')
            raw_data_dict['measured_data'] = OrderedDict()

            data = measured_data[-len(raw_data_dict['value_names']):]
            if data.shape[0] != len(raw_data_dict['value_names']):
                raise ValueError('Shape mismatch between data and '
                                 'ro channels.')
            TD = help_func_mod.get_param('TwoD', raw_data_dict,
                                         default_value=False)
            for i, ro_ch in enumerate(raw_data_dict['value_names']):
                if 'soft_sweep_points' in raw_data_dict and TD:
                    pass
                    # hsl = len(sweep_points[0][list(sweep_points[0])[0]][0])
                    # ssl = len(sweep_points[1][list(sweep_points[0])[0]][0])
                    # measured_data = np.reshape(data[i], (ssl, hsl)).T
                else:
                    measured_data = data[i]
                raw_data_dict['measured_data'][ro_ch] = measured_data
        else:
            raise ValueError('"measured_data" was not added.')
        return raw_data_dict

    def process_data(self):
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
        if len(self.processing_pipe) == 0:
            raise ValueError('Experimental metadata or options_dict must '
                             'contain "processing_pipe."')

        for node_dict in self.processing_pipe:
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
            node(self.data_dict, **node_dict)

    def analyze_fit_results(self):
        """
        Finds the node that is a class, and calls analyze_fit_results.

        ! Assumptions:
            - the pipe had ONLY ONE node that did fitting (there is only one
            'fit_dicts' entry in data_dict
            - the only node that is a class is the one that produced the fit(s)
        """
        for node_dict in self.processing_pipe:
            node = getattr(dat_proc, node_dict["node_type"])
            if isinstance(node, type):
                # the node is a class
                node_inst = node(self.data_dict, prepare_fits=False,
                                 auto=True, **node_dict)
                if hasattr(node_inst, 'analyze_fit_results'):
                    node_inst.analyze_fit_results(
                        fit_dicts=self.data_dict['fit_dicts'])
                    self.data_dict = copy.deepcopy(node_inst.data_dict)

    def prepare_plots(self):
        """
        Finds the node that is a class, and calls prepare_plots.

        ! Assumptions:
            - the pipe had ONLY ONE node that does plotting (there is only one
            'plot_dicts' entry in data_dict
            - the only node that is a class is the one that produced the plot(s)
            - this same node was also the one that did fitting (there is only
            one 'fit_dicts' entry in data_dict
        """
        for node_dict in self.processing_pipe:
            node = getattr(dat_proc, node_dict["node_type"])
            if isinstance(node, type):
                # the node is a class
                node_inst = node(self.data_dict, prepare_fits=False,
                                 auto=True, **node_dict)
                if hasattr(node_inst, 'prepare_plots'):
                    node_inst.prepare_plots(
                        fit_dicts=self.data_dict['fit_dicts'])
                    self.data_dict = copy.deepcopy(node_inst.data_dict)

    @staticmethod
    def _sort_by_axis0(arr, sorted_indices, type=None):
        '''
        Sorts the array (possibly a list of unequally long lists) by a list of indicies
        :param arr: array (possibly a list of unequally long lists)
        :param sorted_indices:  list of indicies
        :param type: the datatype of the contained values
        :return: Sorted array
        '''
        if type is None:
            return [np.array(arr[i]) for i in sorted_indices]
        else:
            return [np.array(arr[i], dtype=type) for i in sorted_indices]


def plot_scatter_errorbar(self, ax_id, xdata, ydata, xerr=None, yerr=None, pdict=None):
    pdict = pdict or {}

    pds = {
        'ax_id': ax_id,
        'plotfn': self.plot_line,
        'zorder': 10,
        'xvals': xdata,
        'yvals': ydata,
        'marker': 'x',
        'linestyle': 'None',
        'yerr': yerr,
        'xerr': xerr,
    }

    if xerr is not None or yerr is not None:
        pds['func'] = 'errorbar'
        pds['marker'] = None
        pds['line_kws'] = {'fmt': 'none'}
        if pdict.get('marker', False):
            pds['line_kws'] = {'fmt': pdict['marker']}
        else:
            ys = 0 if yerr is None else np.min(yerr) / np.max(ydata)
            xs = 0 if xerr is None else np.min(xerr) / np.max(xdata)
            if ys < 1e-2 and xs < 1e-2:
                pds['line_kws'] = {'fmt': 'o'}
    else:
        pds['func'] = 'scatter'

    pds = _merge_dict_rec(pds, pdict)

    return pds


def plot_scatter_errorbar_fit(self, ax_id, xdata, ydata, fitfunc, xerr=None, yerr=None, fitextra=0.1,
                              fitpoints=1000, pdict_scatter=None, pdict_fit=None):
    pdict_fit = pdict_fit or {}
    pds = plot_scatter_errorbar(self=self, ax_id=ax_id, xdata=xdata, ydata=ydata, xerr=xerr, yerr=yerr,
                                pdict=pdict_scatter)

    mi, ma = np.min(xdata), np.max(xdata)
    ex = (ma - mi) * fitextra
    xdata_fit = np.linspace(mi - ex, ma + ex, fitpoints)
    ydata_fit = fitfunc(xdata_fit)

    pdf = {
        'ax_id': ax_id,
        'zorder': 5,
        'plotfn': self.plot_line,
        'xvals': xdata_fit,
        'yvals': ydata_fit,
        'linestyle': '-',
        'marker': '',
    }

    pdf = _merge_dict_rec(pdf, pdict_fit)

    return pds, pdf


def _merge_dict_rec(dict_a: dict, dict_b: dict):
    for k in dict_a:
        if k in dict_b:
            if dict_a[k] is dict or dict_b[k] is dict:
                a = dict_a[k] or {}
                dict_a[k] = _merge_dict_rec(a, dict_b[k])
            else:
                dict_a[k] = dict_b[k]
    for k in dict_b:
        if k not in dict_a:
            dict_a[k] = dict_b[k]
    return dict_a


