import logging
log = logging.getLogger(__name__)

import os
import json
import lmfit
import h5py
from copy import deepcopy
import matplotlib.pyplot as plt
from collections import OrderedDict
from pycqed.utilities.general import NumpyJsonEncoder
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.analysis_v3 import helper_functions as hlp_mod
from pycqed.measurement.hdf5_data import write_dict_to_hdf5


def save_analysis_results(data_dict, ana_res_dict, **params):
    """
    Saves the analysis results.
    """
    ana_res_group_name = hlp_mod.get_param('ana_res_group_name', data_dict,
                                           default_value='Analysis results',
                                           **params)
    overwrite = hlp_mod.get_param('overwrite', data_dict, default_value=True,
                                  **params)

    timestamp = data_dict['timestamps'][-1]
    fn = a_tools.measurement_filename(a_tools.get_folder(timestamp))

    try:
        os.mkdir(os.path.dirname(fn))
    except FileExistsError:
        pass

    if params.get('verbose', False):
        log.info('Saving analysis results to %s' % fn)

    with h5py.File(fn, 'a') as data_file:
        try:
            analysis_group = data_file.create_group('Analysis')
        except ValueError:
            # If the analysis group already exists.
            analysis_group = data_file['Analysis']

        try:
            ana_res_group = \
                analysis_group.create_group(ana_res_group_name)
        except ValueError:
            # If the processed data group already exists.
            ana_res_group = analysis_group[ana_res_group_name]

        for key in ana_res_dict:
            if key in ana_res_group.keys():
                del ana_res_group[key]

            d = {key: ana_res_dict[key]}
            write_dict_to_hdf5(d, entry_point=ana_res_group,
                               overwrite=overwrite)


def save_fit_results(data_dict, fit_res_dict, **params):
    """
    Saves the fit results
    """
    timestamp = data_dict['timestamps'][-1]
    fn = a_tools.measurement_filename(a_tools.get_folder(timestamp))

    try:
        os.mkdir(os.path.dirname(fn))
    except FileExistsError:
        pass

    if params.get('verbose', False):
        log.info('Saving fitting results to %s' % fn)

    with h5py.File(fn, 'a') as data_file:
        try:
            analysis_group = data_file.create_group('Analysis')
        except ValueError:
            # If the analysis group already exists.
            analysis_group = data_file['Analysis']

        # Iterate over all the fit result dicts as not to overwrite
        # old/other analysis
        for fr_key, fit_res in fit_res_dict.items():
            try:
                fr_group = analysis_group.create_group(fr_key)
            except ValueError:
                # If the analysis sub group already exists
                # (each fr_key should be unique)
                # Delete the old group and create a new group
                # (overwrite).
                del analysis_group[fr_key]
                fr_group = analysis_group.create_group(fr_key)

            d = _convert_dict_rec(deepcopy(fit_res))
            write_dict_to_hdf5(d, entry_point=fr_group)


def save_figures(data_dict, figs, **params):

    keys_in = params.get('keys_in', 'auto')
    fmt = params.get('fmt', 'png')
    dpi = params.get('dpi', 300)
    tag_tstamp = params.get('tag_tstamp', True)
    savebase = params.get('savebase', None)
    savedir = params.get('savedir', None)
    if savedir is None:
        if isinstance(data_dict, tuple):
            savedir = data_dict[0].get('folders', '')
        else:
            savedir = data_dict.get('folders', '')

        if isinstance(savedir, list):
            savedir = savedir[-1]
        if isinstance(savedir, list):
            savedir = savedir[-1]
    if savebase is None:
        savebase = ''
    if tag_tstamp:
        if isinstance(data_dict, tuple):
            tstag = '_' + data_dict[0]['timestamps'][-1]
        else:
            tstag = '_' + data_dict['timestamps'][-1]
    else:
        tstag = ''

    if keys_in == 'auto' or keys_in is None:
        keys_in = figs.keys()

    try:
        os.mkdir(savedir)
    except FileExistsError:
        pass

    if params.get('verbose', False):
        log.info('Saving figures to %s' % savedir)

    for key in keys_in:
        if params.get('presentation_mode', False):
            savename = os.path.join(savedir, savebase + key + tstag +
                                    'presentation' + '.' + fmt)
            figs[key].savefig(savename, bbox_inches='tight',
                                   fmt=fmt, dpi=dpi)
            savename = os.path.join(savedir, savebase + key + tstag +
                                    'presentation' + '.svg')
            figs[key].savefig(savename, bbox_inches='tight', fmt='svg')
        else:
            savename = os.path.join(savedir, savebase + key + tstag
                                    + '.' + fmt)
            figs[key].savefig(savename, bbox_inches='tight',
                                   fmt=fmt, dpi=dpi)
        if params.get('close_figs', True):
            plt.close(figs[key])


def _convert_dict_rec(obj):
    try:
        # is iterable?
        for k in obj:
            obj[k] = _convert_dict_rec(obj[k])
    except TypeError:
        if isinstance(obj, lmfit.model.ModelResult):
            obj = _flatten_lmfit_modelresult(obj)
        else:
            obj = str(obj)
    return obj


def _flatten_lmfit_modelresult(model):
    assert type(model) is lmfit.model.ModelResult
    dic = OrderedDict()
    dic['success'] = model.success
    dic['message'] = model.message
    dic['params'] = {}
    for param_name in model.params:
        dic['params'][param_name] = {}
        param = model.params[param_name]
        for k in param.__dict__:
            if k == '_val':
                dic['params'][param_name]['value'] = getattr(param, k)
            else:
                if not k.startswith('_') and k not in ['from_internal', ]:
                    dic['params'][param_name][k] = getattr(param, k)
    return dic
