import logging
log = logging.getLogger(__name__)
from copy import deepcopy
from pycqed.analysis_v3 import helper_functions as help_func_mod

##################################################
#### This module creates an analysis pipeline ####
##################################################


class ProcessingPipeline(list):
    def __init__(self, meas_obj_value_names_map, node_type=None, **params):
        super().__init__()
        self.movnm = meas_obj_value_names_map
        if node_type is not None:
            # self.append(eval(('add_' + node_type + '_node')(**params)))
            self.append(getattr(self, 'add_' + node_type + '_node')(**params))

    def add_node(self, node_type, **params):
        if hasattr(self, 'add_' + node_type + '_node'):
            self.append(getattr(self, 'add_' + node_type + '_node')(**params))
        else:
            params['node_type'] = node_type
            if 'keys_in' in params:
                keys_in, keys_out, meas_obj_names, mobj_keys = \
                    self.check_keys_mobjn(**params)
                params['keys_in'] = keys_in
                if 'keys_out' in params:
                    params['keys_out'] = keys_out
                params['meas_obj_names'] = meas_obj_names
            self.append(params)

    def check_keys_mobjn(self, keys_in, keys_out=(), meas_obj_names='all',
                         keys_out_container='', update_key=False, **params):
        """
        Returns the correct list of keys_in, keys_out, and meas_obj_names.
        :param keys_in:
            'raw': takes all channels from self.movnm for meas_obj_names
            'previous': takes the keys_out of the previous node which contain
                the channels of meas_obj_names
            'previous node_name': takes the keys_out of the previous node
                which contain the channels of meas_obj_names and the node_name
        :param keys_out:
            list or tuple of strings (can be empty). Can also be None
            If empty, populates with keys_in. Useful if keys_in have a '.' char,
            because this function will set keys_out to only what comes after '.'
        :param meas_obj_names:
            'all': returns all keys of self.movnm
            list of string containing measurement object names
        :param keys_out_container:
            string. If not empty, will get appended to each key out with a '.',
            i.e. keys_out = [f'{container}.{k}' for k in keys_out]
        :param update_key:
            if any of keys_out already used in this pipeline, this flag
            specifies whether to keep this key (will result in data being
            overwritten or updated (if the data is a dict, see
            helper_func_mod.add_param)), or create unique ones
        :return: keys_in, keys_out, meas_obj_names, mobj_keys
        """
        if meas_obj_names == 'all':
            mobj_keys = help_func_mod.flatten_list_func(
                list(self.movnm.values()))
            meas_obj_names = list(self.movnm) 
        else:
            mobj_keys = help_func_mod.flatten_list_func(
                [self.movnm[mo] for mo in meas_obj_names])

        prev_keys_out = []
        for d in self:
            if 'keys_out' in d:
                prev_keys_out += d['keys_out']

        if keys_in == 'raw':
            keys_in = mobj_keys
        elif 'previous' in keys_in:
            if len(self) > 0:
                # assumes that what comes after 'previous' is separated by
                # a space
                keys_in_split = keys_in.split(' ')
                if len(keys_in_split) > 1:
                    keys_in = help_func_mod.get_sublst_with_all_strings_of_list(
                        lst_to_search=help_func_mod.flatten_list_func(
                            prev_keys_out),
                        lst_to_match=mobj_keys)
                    keys_in = [ki for ki in keys_in if keys_in_split[1] in ki]
                else:
                    if 'keys_out' not in self[-1]:
                        raise KeyError(
                            f'The previous node {self[-1]["node_type"]} does '
                            f'not have the key "keys_out".')
                    keys_in = help_func_mod.get_sublst_with_all_strings_of_list(
                        lst_to_search=self[-1]['keys_out'],
                        lst_to_match=mobj_keys)
            else:
                raise ValueError('This is the first node in the pipeline. '
                                 'keys_in cannot be "previous".')
            try:
                keys_in.sort()
            except AttributeError:
                pass

        if keys_out is not None:
            if len(keys_out) == 0:
                keys_out = []
                for keyi in keys_in:
                    # take only what comes after '.' in keys_in and prepend
                    # the keys_out_container
                    keyi_split = keyi.split('.')
                    keyo = keyi_split[0] if len(keyi_split) == 1 else \
                        keyi_split[1]
                    if len(keys_out_container) != 0:
                        # check if container.keyo was already used
                        keyo_temp = deepcopy(keyo)
                        keyo = f'{keys_out_container}.{keyo_temp}'
                        num_previously_used = len(
                            help_func_mod.get_sublst_with_all_strings_of_list(
                                lst_to_search=[keyo],
                                lst_to_match=prev_keys_out))
                        if num_previously_used > 0 and not update_key:
                            keys_out_container += f'{num_previously_used}'
                            keyo = f'{keys_out_container}.{keyo_temp}'
                    else:
                        num_previously_used = len(
                            help_func_mod.get_sublst_with_all_strings_of_list(
                                lst_to_search=[keyo],
                                lst_to_match=prev_keys_out))
                        if num_previously_used > 0 and not update_key:
                            raise ValueError(
                                f'{keyo} has already been used in this '
                                f'pipeline. Specify unique keys_out or set '
                                f'"update_keys_out=True" if you want to '
                                f'overwrite the data.')
                    keys_out += [keyo]

        return keys_in, keys_out, meas_obj_names, mobj_keys

    def add_filter_data_node(self, data_filter, keys_in='previous',
                             meas_obj_names='all', keys_out=(), **params):
        keys_in, keys_out, meas_obj_names, mobj_keys = self.check_keys_mobjn(
            keys_in, keys_out, meas_obj_names,
            keys_out_container=params.pop('keys_out_container', 'filter_data'),
            **params)

        return {'node_type': 'filter_data',
                'keys_in': keys_in,
                'keys_out': keys_out,
                'data_filter': data_filter,
                **params}

    def add_average_data_node(self, shape, averaging_axis=-1,
                              keys_in='previous', meas_obj_names='all',
                              keys_out=(), **params):
        keys_in, keys_out, meas_obj_names, mobj_keys = self.check_keys_mobjn(
            keys_in, keys_out, meas_obj_names,
            keys_out_container=params.pop('keys_out_container', 'average_data'),
            **params)

        return {'node_type': 'average_data',
                'keys_in': keys_in,
                'keys_out': keys_out,
                'shape': shape,
                'averaging_axis': averaging_axis,
                **params}

    def add_get_std_deviation_node(self, shape, averaging_axis=-1,
                                   keys_in='previous', meas_obj_names='all',
                                   keys_out=(), **params):
        keys_in, keys_out, meas_obj_names, mobj_keys = self.check_keys_mobjn(
            keys_in, keys_out, meas_obj_names,
            keys_out_container=params.pop('keys_out_container',
                                          'get_std_deviation'),
            **params)

        return {'node_type': 'get_std_deviation',
                'keys_in': keys_in,
                'keys_out': keys_out,
                'shape': shape,
                'averaging_axis': averaging_axis,
                **params}

    def add_rotate_iq_node(self, keys_in='previous',
                           meas_obj_names='all', keys_out=(), **params):
        keys_in, keys_out, meas_obj_names, mobj_keys = self.check_keys_mobjn(
            keys_in, keys_out, meas_obj_names, **params)

        if keys_out is not None:
            keys_out_container = params.pop('keys_out_container', 'rotate_iq')
            keys_out = [f'{keys_out_container}.' + ','.join([
                k.split('.')[-1] for k in keys_in])]

        return {'node_type': 'rotate_iq',
                'keys_in': keys_in,
                'keys_out': keys_out,
                'meas_obj_names': meas_obj_names,
                **params}

    def add_rotate_1d_array_node(self, keys_in='previous',
                                 meas_obj_names='all', keys_out=(), **params):
        keys_in, keys_out, meas_obj_names, mobj_keys = self.check_keys_mobjn(
            keys_in, keys_out, meas_obj_names,
            keys_out_container=params.pop('keys_out_container',
                                          'rotate_1d_array'),
            **params)

        return {'node_type': 'rotate_1d_array',
                'keys_in': keys_in,
                'keys_out': keys_out,
                'meas_obj_names': meas_obj_names,
                **params}

    def add_threshold_data_node(self, threshold_list, threshold_map,
                                keys_in='previous', meas_obj_names='all',
                                keys_out=(), **params):
        keys_in, keys_out, meas_obj_names, mobj_keys = self.check_keys_mobjn(
            keys_in, keys_out, meas_obj_names, **params)

        if keys_out is not None:
            keys_out_container = params.pop('keys_out_container',
                                            'threshold_data')
            keyo = keys_in[0] if len(keys_in) == 1 else ','.join([
                k.split('.')[-1] for k in keys_in])
            keys_out = [f'{keys_out_container}.{keyo} state {s}' for s in
                        set(threshold_map.values())]
        return {'node_type': 'threshold_data',
                'keys_in': keys_in,
                'keys_out': keys_out,
                'threshold_list': threshold_list,
                'threshold_map': threshold_map,
                **params}

    def add_transform_data_node(self, transform_func, keys_in='previous',
                                meas_obj_names='all', keys_out=(),
                                transform_func_kwargs=dict(), **params):
        keys_in, keys_out, meas_obj_names, mobj_keys = self.check_keys_mobjn(
            keys_in, keys_out, meas_obj_names,
            keys_out_container=params.pop('keys_out_container',
                                          'transform_data'),
            **params)

        return {'node_type': 'transform_data',
                'keys_in': keys_in,
                'keys_out': keys_out,
                'transform_func': transform_func,
                'transform_func_kwargs': transform_func_kwargs,
                **params}

    def add_correct_readout_node(self, state_prob_mtx, keys_in='previous',
                                 meas_obj_names='all', keys_out=(), **params):
        keys_in, keys_out, meas_obj_names, mobj_keys = self.check_keys_mobjn(
            keys_in, keys_out, meas_obj_names,
            keys_out_container=params.pop('keys_out_container',
                                          'correct_readout'),
            **params)

        return {'node_type': 'correct_readout',
                'keys_in': keys_in,
                'keys_out': keys_out,
                'state_prob_mtx': state_prob_mtx,
                **params}


    ######################################
    #### plot dicts preparation nodes ####
    ######################################

    def add_prepare_1d_plot_dicts_node(
            self, keys_in='previous', meas_obj_names='all', figure_name='',
            do_plotting=True, **params):
        keys_in, _, meas_obj_names, mobj_keys = self.check_keys_mobjn(
            keys_in, meas_obj_names=meas_obj_names, **params)
        return {'node_type': 'prepare_1d_plot_dicts',
                'keys_in': keys_in,
                'meas_obj_names': meas_obj_names,
                'figure_name': figure_name,
                'do_plotting': do_plotting,
                **params}

    def add_prepare_2d_plot_dicts_node(
            self, keys_in='previous', meas_obj_names='all', figure_name='',
            do_plotting=True, **params):
        keys_in, _, meas_obj_names, mobj_keys = self.check_keys_mobjn(
            keys_in, meas_obj_names=meas_obj_names, **params)
        return {'node_type': 'prepare_2d_plot_dicts',
                'keys_in': keys_in,
                'meas_obj_names': meas_obj_names,
                'figure_name': figure_name,
                'do_plotting': do_plotting,
                **params}

    def add_prepare_1d_raw_data_plot_dicts_node(
            self, keys_in='previous', meas_obj_names='all', figure_name=None,
            do_plotting=True, **params):
        keys_in, _, meas_obj_names, mobj_keys = self.check_keys_mobjn(
            keys_in, meas_obj_names=meas_obj_names, **params)
        return {'node_type': 'prepare_1d_raw_data_plot_dicts',
                'keys_in': keys_in,
                'meas_obj_names': meas_obj_names,
                'figure_name': figure_name,
                'do_plotting': do_plotting,
                **params}

    def add_prepare_2d_raw_data_plot_dicts_node(
            self, keys_in='previous', meas_obj_names='all', figure_name=None,
            do_plotting=True, **params):
        keys_in, _, meas_obj_names, mobj_keys = self.check_keys_mobjn(
            keys_in, meas_obj_names=meas_obj_names, **params)
        return {'node_type': 'prepare_2d_raw_data_plot_dicts',
                'keys_in': keys_in,
                'meas_obj_names': meas_obj_names,
                'figure_name': figure_name,
                'do_plotting': do_plotting,
                **params}

    def add_prepare_cal_states_plot_dicts_node(
            self, keys_in='previous', meas_obj_names='all', figure_name='',
            do_plotting=True, **params):
        keys_in, _, meas_obj_names, mobj_keys = self.check_keys_mobjn(
            keys_in, meas_obj_names=meas_obj_names, **params)
        return {'node_type': 'prepare_cal_states_plot_dicts',
                'keys_in': keys_in,
                'meas_obj_names': meas_obj_names,
                'figure_name': figure_name,
                'do_plotting': do_plotting,
                **params}

    ################################
    #### nodes that are classes ####
    ################################

    def add_RabiAnalysis_node(self, meas_obj_names, keys_in='previous', 
                              **params):
        keys_in, _, meas_obj_names, mobj_keys = self.check_keys_mobjn(
            keys_in, meas_obj_names=meas_obj_names, **params)
        return {'node_type': 'RabiAnalysis',
                'keys_in': keys_in,
                'meas_obj_names': meas_obj_names,
                **params}

    def add_SingleQubitRBAnalysis_node(self, meas_obj_names, keys_in='previous',
                                       std_keys=None, **params):
        keys_in, _, meas_obj_names, mobj_keys = self.check_keys_mobjn(
            keys_in, meas_obj_names=meas_obj_names, **params)
        std_keys, _, meas_obj_names, mobj_keys = self.check_keys_mobjn(
            std_keys, meas_obj_names=meas_obj_names)

        return {'node_type': 'SingleQubitRBAnalysis',
                'keys_in': keys_in,
                'std_keys': std_keys,
                'meas_obj_names': meas_obj_names,
                **params}



