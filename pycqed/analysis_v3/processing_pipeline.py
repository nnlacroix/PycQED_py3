import logging
log = logging.getLogger(__name__)

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
            self.append(params)

    def check_keys_mobjn(self, keys_in, keys_out=(), meas_obj_names='all',
                         keys_out_container=''):
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
        :return: keys_in, keys_out, meas_obj_names, mobj_keys
        """
        if meas_obj_names == 'all':
            mobj_keys = help_func_mod.flatten_list_func(
                list(self.movnm.values()))
            meas_obj_names = list(self.movnm) 
        else:
            mobj_keys = help_func_mod.flatten_list_func(
                [self.movnm[mo] for mo in meas_obj_names])

        if keys_in == 'raw':
            keys_in = mobj_keys
        elif 'previous' in keys_in:
            if len(self) > 0:
                # assumes that what comes after 'previous' is separated by
                # a space
                keys_in_split = keys_in.split(' ')
                if len(keys_in_split) > 1:
                    prev_keys_out = []
                    for d in self:
                        if 'keys_out' in d:
                            prev_keys_out += [d['keys_out']]
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
            
        if keys_out is not None:
            if len(keys_out) == 0:
                keys_out = []
                for keyi in keys_in:
                    keyi_split = keyi.split('.')
                    keys_out += [keyi_split[0] if len(keyi_split) == 1 else
                                 keyi_split[1]]
                if len(keys_out_container) != 0:
                    keys_out = [f'{keys_out_container}.{k}' for k in keys_out]
        return keys_in, keys_out, meas_obj_names, mobj_keys

    def add_filter_data_node(self, data_filter, keys_in='previous',
                             meas_obj_names='all', keys_out=(), **params):
        keys_in, keys_out, meas_obj_names, mobj_keys = self.check_keys_mobjn(
            keys_in, keys_out, meas_obj_names, keys_out_container='filter_data')

        return {'node_type': 'filter_data',
                'keys_in': keys_in,
                'keys_out': keys_out,
                'data_filter': data_filter,
                **params}

    def add_average_data_node(self, num_bins, keys_in='previous',
                              meas_obj_names='all', keys_out=(), **params):
        keys_in, keys_out, meas_obj_names, mobj_keys = self.check_keys_mobjn(
            keys_in, keys_out, meas_obj_names, keys_out_container='average_data')

        return {'node_type': 'average_data',
                'keys_in': keys_in,
                'keys_out': keys_out,
                'num_bins': num_bins,
                **params}

    def add_get_std_deviation_node(self, num_bins, keys_in='previous',
                                   meas_obj_names='all', keys_out=(),
                                   **params):
        keys_in, keys_out, meas_obj_names, mobj_keys = self.check_keys_mobjn(
            keys_in, keys_out, meas_obj_names,
            keys_out_container='get_std_deviation')

        return {'node_type': 'get_std_deviation',
                'keys_in': keys_in,
                'keys_out': keys_out,
                'num_bins': num_bins,
                **params}

    def add_rotate_iq_node(self, keys_in='previous',
                           meas_obj_names='all', keys_out=(), **params):
        keys_in, keys_out, meas_obj_names, mobj_keys = self.check_keys_mobjn(
            keys_in, keys_out, meas_obj_names)

        if keys_out is not None:
            keys_out = ['rotate_iq.' + ','.join(keys_in)]
        return {'node_type': 'rotate_iq',
                'keys_in': keys_in,
                'keys_out': keys_out,
                'meas_obj_names': meas_obj_names,
                **params}

    def add_rotate_1d_array_node(self, keys_in='previous',
                                 meas_obj_names='all', keys_out=(), **params):
        keys_in, keys_out, meas_obj_names, mobj_keys = self.check_keys_mobjn(
            keys_in, keys_out, meas_obj_names,
            keys_out_container='rotate_1d_array')

        return {'node_type': 'rotate_1d_array',
                'keys_in': keys_in,
                'keys_out': keys_out,
                'meas_obj_names': meas_obj_names,
                **params}

    def add_threshold_data_node(self, threshold_list, threshold_map,
                                keys_in='previous', meas_obj_names='all',
                                keys_out=(), **params):
        keys_in, keys_out, meas_obj_names, mobj_keys = self.check_keys_mobjn(
            keys_in, keys_out, meas_obj_names)

        if keys_out is not None:
            keyo = keys_in[0] if len(keys_in) == 1 else ','.join(keys_in)
            keys_out = [f'threshold_data.{keyo} state {s}' for s in
                        set(threshold_map.values())]
        return {'node_type': 'threshold_data',
                'keys_in': keys_in,
                'keys_out': keys_out,
                'threshold_list': threshold_list,
                'threshold_map': threshold_map,
                **params}


    ######################################
    #### plot dicts preparation nodes ####
    ######################################

    def add_prepare_1d_plot_dicts_node(
            self, keys_in='previous', meas_obj_names='all', fig_name='',
            do_plotting=False, **params):
        keys_in, _, meas_obj_names, mobj_keys = self.check_keys_mobjn(
            keys_in, meas_obj_names=meas_obj_names)
        return {'node_type': 'prepare_1d_plot_dicts',
                'keys_in': keys_in,
                'meas_obj_names': meas_obj_names,
                'fig_name': fig_name,
                'do_plotting': do_plotting,
                **params}

    def add_prepare_raw_data_plot_dicts_node(
            self, keys_in='previous', meas_obj_names='all', fig_name='',
            do_plotting=False, **params):
        keys_in, _, meas_obj_names, mobj_keys = self.check_keys_mobjn(
            keys_in, meas_obj_names=meas_obj_names)
        return {'node_type': 'prepare_raw_data_plot_dicts',
                'keys_in': keys_in,
                'meas_obj_names': meas_obj_names,
                'fig_name': fig_name,
                'do_plotting': do_plotting,
                **params}

    def add_prepare_cal_states_plot_dicts_node(
            self, keys_in='previous', meas_obj_names='all', fig_name='',
            do_plotting=False, **params):
        keys_in, _, meas_obj_names, mobj_keys = self.check_keys_mobjn(
            keys_in, meas_obj_names=meas_obj_names)
        return {'node_type': 'prepare_cal_states_plot_dicts',
                'keys_in': keys_in,
                'meas_obj_names': meas_obj_names,
                'fig_name': fig_name,
                'do_plotting': do_plotting,
                **params}

    ################################
    #### nodes that are classes ####
    ################################

    def add_RabiAnalysis_node(self, meas_obj_names, keys_in='previous', 
                              **params):
        keys_in, _, meas_obj_names, mobj_keys = self.check_keys_mobjn(
            keys_in, meas_obj_names=meas_obj_names)
        return {'node_type': 'RabiAnalysis',
                'keys_in': keys_in,
                'meas_obj_names': meas_obj_names,
                **params}

    def add_SingleQubitRBAnalysis_node(self, meas_obj_names, keys_in='previous',
                                       std_keys=None, **params):
        keys_in, _, meas_obj_names, mobj_keys = self.check_keys_mobjn(
            keys_in, meas_obj_names=meas_obj_names)
        std_keys, _, meas_obj_names, mobj_keys = self.check_keys_mobjn(
            std_keys, meas_obj_names=meas_obj_names)
        return {'node_type': 'SingleQubitRBAnalysis',
                'keys_in': keys_in,
                'std_keys': std_keys,
                'meas_obj_names': meas_obj_names,
                **params}



