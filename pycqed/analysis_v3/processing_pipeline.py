import logging
log = logging.getLogger(__name__)
import sys
this_module = sys.modules[__name__]

from copy import deepcopy
from pycqed.analysis_v3 import helper_functions as hlp_mod

##################################################
#### This module creates an analysis pipeline ####
##################################################

class RawPipeline(list):
    def __init__(self, node_name=None, **node_params):
        super().__init__()
        if node_name is not None:
            node_params['node_name'] = node_name
            self.append(node_params)

    def add_node(self, node_name, **node_params):
        node_params['node_name'] = node_name
        self.append(node_params)


def create_pipeline(raw_pipeline, meas_obj_value_names_map):
    pipeline = []
    for node_params in raw_pipeline:
        if 'keys_in' in node_params:
            keys_in, keys_out, meas_obj_names, mobj_keys = \
                check_keys_mobjn(pipeline, meas_obj_value_names_map,
                                 **node_params)
            node_params['keys_in'] = keys_in
            node_params['keys_out'] = keys_out
            node_params['meas_obj_names'] = meas_obj_names
        pipeline.append(node_params)
    return pipeline


def check_keys_mobjn(pipeline, meas_obj_value_names_map, keys_in,
                     keys_out=(), meas_obj_names='all', **node_params):
    """
    Returns the explicit list of keys_in, keys_out, and meas_obj_names.
    :param pipeline: a processing pipeline (not raw!)
    :param meas_obj_value_names_map: dictionary with measurement objects as keys
        and list of readout channels as values
    :param keys_in:
        'raw': takes all channels from self._movnm for meas_obj_names
        'previous': takes the keys_out of the previous node which contain
            the channels of meas_obj_names
        'previous node_name': takes the keys_out of the previous node
            which contain the channels of meas_obj_names and the node_name
    :param keys_out:
        list or tuple of strings (can be empty). Can also be None
        If empty, populates with keys_in. Useful if keys_in have a '.' char,
        because this function will set keys_out to only what comes after '.'
    :param meas_obj_names:
        'all': returns all keys of self._movnm
        list of string containing measurement object names
    :return: keys_in, keys_out, meas_obj_names, mobj_keys

    Assumptions:
        -
    """
    prev_keys_out = []
    for d in pipeline:
        if 'keys_out' in d:
            if d['keys_out'] is not None:
                prev_keys_out += d['keys_out']

    # check keys_in
    if meas_obj_names == 'all':
        mobj_keys = hlp_mod.flatten_list(
            list(meas_obj_value_names_map.values()))
        meas_obj_names = list(meas_obj_value_names_map)
    else:
        mobj_keys = hlp_mod.flatten_list(
            [meas_obj_value_names_map[mo] for mo in meas_obj_names])

    if keys_in == 'raw':
        keys_in = mobj_keys
    elif 'previous' in keys_in:
        if len(pipeline) > 0:
            # assumes that what comes after 'previous' is separated by
            # a space
            keys_in_split = keys_in.split(' ')
            if len(keys_in_split) > 1:
                keys_in = hlp_mod.get_sublst_with_all_strings_of_list(
                    lst_to_search=hlp_mod.flatten_list(prev_keys_out),
                    lst_to_match=mobj_keys)
                keys_in = [ki for ki in keys_in if keys_in_split[-1] in ki]
            else:
                if 'keys_out' not in pipeline[-1]:
                    raise KeyError(
                        f'The previous node {pipeline[-1]["node_name"]} does '
                        f'not have the key "keys_out".')
                keys_in = hlp_mod.get_sublst_with_all_strings_of_list(
                    lst_to_search=pipeline[-1]['keys_out'],
                    lst_to_match=mobj_keys)
        else:
            raise ValueError('This is the first node in the pipeline. '
                             'keys_in cannot be "previous".')
        try:
            keys_in.sort()
        except AttributeError:
            pass

    if len(keys_in) == 0:
        raise ValueError('No "keys_in" could be determined.')

    # check keys_out
    if keys_out is not None:
        if len(keys_out) == 0:
            keys_out_container = ",".join(meas_obj_names)
            node_name = node_params['node_name']
            num_keys_out = node_params.get('num_keys_out', len(keys_in))

            # ensure all keys_in are used
            assert len(keys_in) % num_keys_out == 0
            n = len(keys_in) // num_keys_out

            keys_out = []
            for keyis in [keys_in[i*n: i*n + n] for i in range(num_keys_out)]:
                # take only what comes after '.' in keys_in and prepend
                # the keys_out_container
                node_name_repeated = False
                keyis_mod = deepcopy(keyis)
                for i, keyi in enumerate(keyis):
                    if node_name in keyi:
                        node_name_repeated = True
                        keyis_mod[i] = keyi[len(keyi.split(' ')[0])+1:]
                keyo = ','.join([keyi.split('.')[-1] for keyi in keyis_mod])
                keyo_temp = deepcopy(keyo)

                node_name_to_use = deepcopy(node_name)
                if node_name_repeated:
                    # find how many times was the node_name used
                    num_previously_used = len(
                        hlp_mod.get_sublst_with_all_strings_of_list(
                            lst_to_search=[node_name],
                            lst_to_match=prev_keys_out))
                    node_name_to_use = f'{node_name}{num_previously_used+1}'
                keyo = f'{keys_out_container}.{node_name_to_use} {keyo_temp}'

                keys_out += [keyo]

    return keys_in, keys_out, meas_obj_names, mobj_keys


class ProcessingPipeline(list):
    """
    Creates a processing pipeline for analysis_v3.
    The pipeline is a list of dictionaries
    """
    def __init__(self, meas_obj_value_names_map, node_name=None, **node_params):
        super().__init__()
        self._movnm = meas_obj_value_names_map
        if node_name is not None:
            self.add_node(node_name, **node_params)

    def add_node(self, node_name, **node_params):
        # if hasattr(this_module, 'add_' + node_name + '_node'):
        #     self.append(getattr(this_module, 'add_' + node_name + '_node')(
        #         self, self._movnm, **node_params))
        # else:
        node_params['node_name'] = node_name
        if 'keys_in' in node_params:
            keys_in, keys_out, meas_obj_names, mobj_keys = \
                check_keys_mobjn(self, self._movnm, **node_params)
            node_params['keys_in'] = keys_in
            node_params['keys_out'] = keys_out
            node_params['meas_obj_names'] = meas_obj_names
        self.append(node_params)


def add_filter_data_node(pipeline, movnm, data_filter, keys_in='previous',
                         meas_obj_names='all', keys_out=(), **params):
    keys_in, keys_out, meas_obj_names, mobj_keys = check_keys_mobjn(
        pipeline, movnm, keys_in, keys_out, meas_obj_names,
        keys_out_container=params.pop('keys_out_container', 'filter_data'),
        **params)

    return {'node_name': 'filter_data',
            'keys_in': keys_in,
            'keys_out': keys_out,
            'data_filter': data_filter,
            **params}


def add_average_data_node(pipeline, movnm, shape, averaging_axis=-1,
                          keys_in='previous', meas_obj_names='all',
                          keys_out=(), **params):
    keys_in, keys_out, meas_obj_names, mobj_keys = check_keys_mobjn(
        pipeline, movnm, keys_in, keys_out, meas_obj_names,
        keys_out_container=params.pop('keys_out_container', 'average_data'),
        **params)

    return {'node_name': 'average_data',
            'keys_in': keys_in,
            'keys_out': keys_out,
            'shape': shape,
            'averaging_axis': averaging_axis,
            **params}


def add_get_std_deviation_node(pipeline, movnm, shape, averaging_axis=-1,
                               keys_in='previous', meas_obj_names='all',
                               keys_out=(), **params):
    keys_in, keys_out, meas_obj_names, mobj_keys = check_keys_mobjn(
        pipeline, movnm, keys_in, keys_out, meas_obj_names,
        keys_out_container=params.pop('keys_out_container',
                                      'get_std_deviation'),
        **params)

    return {'node_name': 'get_std_deviation',
            'keys_in': keys_in,
            'keys_out': keys_out,
            'shape': shape,
            'averaging_axis': averaging_axis,
            **params}


def add_rotate_iq_node(pipeline, movnm, keys_in='previous',
                       meas_obj_names='all', keys_out=(), **params):

    keys_in, keys_out, meas_obj_names, mobj_keys = check_keys_mobjn(
        pipeline, movnm, keys_in, keys_out, meas_obj_names, **params)

    if keys_out is not None:
        keys_out_container = params.pop('keys_out_container', 'rotate_iq')
        keys_out = [f'{",".join(meas_obj_names)}.{keys_out_container}_' +
                    ','.join([k.split('.')[-1] for k in keys_in])]

    return {'node_name': 'rotate_iq',
            'keys_in': keys_in,
            'keys_out': keys_out,
            'meas_obj_names': meas_obj_names,
            **params}


def add_rotate_1d_array_node(pipeline, movnm, keys_in='previous',
                             meas_obj_names='all', keys_out=(), **params):
    keys_in, keys_out, meas_obj_names, mobj_keys = check_keys_mobjn(
        pipeline, movnm, keys_in, keys_out, meas_obj_names,
        keys_out_container=params.pop('keys_out_container',
                                      'rotate_1d_array'),
        **params)

    return {'node_name': 'rotate_1d_array',
            'keys_in': keys_in,
            'keys_out': keys_out,
            'meas_obj_names': meas_obj_names,
            **params}


def add_threshold_data_node(pipeline, movnm, threshold_list, threshold_map,
                            keys_in='previous', meas_obj_names='all',
                            keys_out=(), **params):
    keys_in, keys_out, meas_obj_names, mobj_keys = check_keys_mobjn(
        pipeline, movnm, keys_in, keys_out, meas_obj_names, **params)

    if keys_out is not None:
        keys_out_container = params.pop('keys_out_container',
                                        'threshold_data')
        keyo = keys_in[0] if len(keys_in) == 1 else ','.join([
            k.split('.')[-1] for k in keys_in])
        keys_out = [f'{keys_out_container}.{keyo} state {s}' for s in
                    set(threshold_map.values())]
    return {'node_name': 'threshold_data',
            'keys_in': keys_in,
            'keys_out': keys_out,
            'threshold_list': threshold_list,
            'threshold_map': threshold_map,
            **params}


def add_transform_data_node(pipeline, movnm, transform_func, keys_in='previous',
                            meas_obj_names='all', keys_out=(),
                            transform_func_kwargs=dict(), **params):
    keys_in, keys_out, meas_obj_names, mobj_keys = check_keys_mobjn(
        pipeline, movnm, keys_in, keys_out, meas_obj_names,
        keys_out_container=params.pop('keys_out_container',
                                      'transform_data'),
        **params)

    return {'node_name': 'transform_data',
            'keys_in': keys_in,
            'keys_out': keys_out,
            'transform_func': transform_func,
            'transform_func_kwargs': transform_func_kwargs,
            **params}


def add_correct_readout_node(pipeline, movnm, state_prob_mtx, keys_in='previous',
                             meas_obj_names='all', keys_out=(), **params):
    keys_in, keys_out, meas_obj_names, mobj_keys = check_keys_mobjn(
        pipeline, movnm, keys_in, keys_out, meas_obj_names,
        keys_out_container=params.pop('keys_out_container',
                                      'correct_readout'),
        **params)

    return {'node_name': 'correct_readout',
            'keys_in': keys_in,
            'keys_out': keys_out,
            'state_prob_mtx': state_prob_mtx,
            **params}


######################################
#### plot dicts preparation nodes ####
######################################

def add_prepare_1d_plot_dicts_node(pipeline, movnm, keys_in='previous',
                                   meas_obj_names='all', figure_name='',
                                   do_plotting=True, **params):
    keys_in, _, meas_obj_names, mobj_keys = check_keys_mobjn(pipeline, movnm,
        keys_in, meas_obj_names=meas_obj_names, **params)
    return {'node_name': 'prepare_1d_plot_dicts',
            'keys_in': keys_in,
            'meas_obj_names': meas_obj_names,
            'figure_name': figure_name,
            'do_plotting': do_plotting,
            **params}


def add_prepare_2d_plot_dicts_node(pipeline, movnm, keys_in='previous',
                                   meas_obj_names='all', figure_name='',
                                   do_plotting=True, **params):
    keys_in, _, meas_obj_names, mobj_keys = check_keys_mobjn(pipeline, movnm,
        keys_in, meas_obj_names=meas_obj_names, **params)
    return {'node_name': 'prepare_2d_plot_dicts',
            'keys_in': keys_in,
            'meas_obj_names': meas_obj_names,
            'figure_name': figure_name,
            'do_plotting': do_plotting,
            **params}


def add_prepare_1d_raw_data_plot_dicts_node(pipeline, movnm, keys_in='previous',
                                            meas_obj_names='all',
                                            figure_name=None, do_plotting=True,
                                            **params):
    keys_in, _, meas_obj_names, mobj_keys = check_keys_mobjn(pipeline, movnm,
        keys_in, meas_obj_names=meas_obj_names, **params)
    return {'node_name': 'prepare_1d_raw_data_plot_dicts',
            'keys_in': keys_in,
            'meas_obj_names': meas_obj_names,
            'figure_name': figure_name,
            'do_plotting': do_plotting,
            **params}


def add_prepare_2d_raw_data_plot_dicts_node(pipeline, movnm, keys_in='previous',
                                            meas_obj_names='all',
                                            figure_name=None, do_plotting=True,
                                            **params):
    keys_in, _, meas_obj_names, mobj_keys = check_keys_mobjn(pipeline, movnm,
        keys_in, meas_obj_names=meas_obj_names, **params)
    return {'node_name': 'prepare_2d_raw_data_plot_dicts',
            'keys_in': keys_in,
            'meas_obj_names': meas_obj_names,
            'figure_name': figure_name,
            'do_plotting': do_plotting,
            **params}


def add_prepare_cal_states_plot_dicts_node(pipeline, movnm, keys_in='previous',
                                           meas_obj_names='all', figure_name='',
                                           do_plotting=True, **params):
    keys_in, _, meas_obj_names, mobj_keys = check_keys_mobjn(pipeline, movnm,
        keys_in, meas_obj_names=meas_obj_names, **params)
    return {'node_name': 'prepare_cal_states_plot_dicts',
            'keys_in': keys_in,
            'meas_obj_names': meas_obj_names,
            'figure_name': figure_name,
            'do_plotting': do_plotting,
            **params}

################################
#### nodes that are classes ####
################################

def add_RabiAnalysis_node(pipeline, movnm, meas_obj_names, keys_in='previous',
                          **params):
    keys_in, _, meas_obj_names, mobj_keys = check_keys_mobjn(pipeline, movnm,
        keys_in, meas_obj_names=meas_obj_names, **params)
    return {'node_name': 'RabiAnalysis',
            'keys_in': keys_in,
            'meas_obj_names': meas_obj_names,
            **params}


def add_SingleQubitRBAnalysis_node(pipeline, movnm, meas_obj_names, 
                                   keys_in='previous', std_keys=None, **params):
    keys_in, _, meas_obj_names, mobj_keys = check_keys_mobjn(pipeline, movnm,
        keys_in, meas_obj_names=meas_obj_names, **params)
    std_keys, _, meas_obj_names, mobj_keys = check_keys_mobjn(pipeline, movnm,
        std_keys, meas_obj_names=meas_obj_names)

    return {'node_name': 'SingleQubitRBAnalysis',
            'keys_in': keys_in,
            'std_keys': std_keys,
            'meas_obj_names': meas_obj_names,
            **params}



