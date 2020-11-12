import logging
log = logging.getLogger(__name__)
import re
import os
from copy import deepcopy
from pycqed.analysis_v3 import helper_functions as hlp_mod

try:
    import pygraphviz as pgv
except ModuleNotFoundError:
    log.warning('Visualizing the pipeline tree requires graphviz '
                '(http://www.graphviz.org/download/) and '
                'the module pygraphviz '
                '(conda install graphviz pygraphviz -c alubbock).')

###################################################################
#### This module creates a processing pipeline for analysis_v3 ####
###################################################################

"""
The pipeline is a list of dictionaries.
Each dictionary contains
    - NECESSARILY the key "node_name" with value being a string specifying
    the name of a processing function withing analysis_v3
    - NECESSARILY the key "keys_in" with value a list of strings that
     specify the keys that are already in data_dict that correspond to
     the data arrays to be processed by the current node.
    - NECESSARILY the key "meas_obj_names" which contains a string or a
     list of strings specifying the name(s) of the object(s) measured in
     the experiment.
     These can be for example qubits (['qb1', 'qb2', 'qb3']), or anything
     else ('TWPA', 'dummy', ['test1', 'test2'] etc.)
    - VERY OFTEN the key "keys_out" with value a list of strings
     specifiying the key names under which the data processed by the current
     node will be save in the data_dict.
    - any other keyword arguments that the current node might require

From here on I will refer to the processing functions in the pipeline as
nodes.

Instructions for use:
    Initialization
        - from a list of dicts: ProcessingPipeline(from_dict_list=dict_list)
        - without any input arguments: ProcessingPipeline()
        - or with input parameters:
         ProcessingPipeline(node_name, **node_params), where node_name is
         the name of the node, and **node_params all the parameters
         required by the node including the necessary keys described above
        ! Specify the keyword argument global_keys_out_container to prepend it 
            to all the keys_out as global_keys_out_container.keyo.
        ! For ease of use, keys_in can also be specified as
            - 'raw': the raw data corresponding to the measured object
            - 'previous': the keys_out of the previous node dictionary
             for the measured object.
            - 'previous node_name': the keys_out of the
             dictionary for the measured object which has the node_name.
             Use 'previous node_namei' where i is the i'th identical appearance
             of node_name in the pipeline for that meas_obj.
        ! keys_out do not need to be specified by the user as they will be
         automatically constructed from the measured object name and the
         keys_in
        ! use keys_out_container in the **node_params to prepend it to the 
         keys_out of that node 


            Examples:
                ProcessingPipeline('average_data',
                                    keys_in='raw',
                                    shape=(3,2),
                                    meas_obj_names='TWPA')
                ProcessingPipeline('ramsey_analysis',
                                    keys_in='previous rotate_iq',
                                    meas_obj_names=['qb1', 'qb2'])

    Adding processing node dictionaries:
        - to add more node dictionaries to the pipeline, call the "add_node"
         method with the same "node_name" and **node_params arguments as
         described above under "Initialization."
         
            Example: same as above but replace ProcessingPipeline with 
            ProcessingPipeline_instance.add_node
    
    Up to now, the pipeline is just a list of dictionaries with the
    key-value pairs as provided by the user:
        
        Example of a "raw" pipeline: 
            [{'keys_in': 'raw',
              'shape': (80, 10),
              'meas_obj_names': ['qb2'],
              'node_name': 'average_data'},
             {'keys_in': 'previous qb2.average_data',
              'shape': (10, 8),
              'averaging_axis': 0,
              'meas_obj_names': ['qb2'],
              'update_key': False,
              'node_name': 'average_data'},
             {'meas_obj_names': ['qb2'],
              'keys_out': None,
              'keys_in': 'previous qb2.average_data1',
              'std_keys': 'previous qb2.get_std_deviation1',
              'node_name': 'SingleQubitRBAnalysis'}]
    
    Creating the pipeline:
        - the analysis framework always expects keys_in to be a list of 
         keys in the data_dict, and most functions expect keys_out
        - to create the pipeline that will be used by the analysis 
         framework, the user can call: 
         ProcessingPipeline_instance(meas_obj_value_names_map), where 
         meas_obj_value_names_map is a dictionary with measured objects as keys
         and list of their corresponding readout channels as values.
         However, the analysis supports an precompiled pipeline as well, in 
         which case it will call ProcessingPipeline_instance(
         meas_obj_value_names_map).
        
        The final pipeline corresponding to the"raw" pipeline above:
             meas_obj_value_names_map = {'qb2': ['UHF1_pg w23 UHF1', 
                                                 'UHF1_pe w23 UHF1', 
                                                 'UHF1_pf w23 UHF1']}
             
             Final pipeline:
                 [{'keys_in': ['UHF1_pg w23 UHF1', 'UHF1_pe w23 UHF1', 
                               'UHF1_pf w23 UHF1'],
                  'shape': (80, 10),
                  'meas_obj_names': ['qb2'],
                  'node_name': 'average_data',
                  'keys_out': ['qb2.average_data UHF1_pg w23 UHF1',
                   'qb2.average_data UHF1_pe w23 UHF1',
                   'qb2.average_data UHF1_pf w23 UHF1']},
                 {'keys_in': ['qb2.average_data UHF1_pe w23 UHF1',
                   'qb2.average_data UHF1_pf w23 UHF1',
                   'qb2.average_data UHF1_pg w23 UHF1'],
                  'shape': (10, 8),
                  'averaging_axis': 0,
                  'meas_obj_names': ['qb2'],
                  'update_key': False,
                  'node_name': 'average_data',
                  'keys_out': ['qb2.average_data1 UHF1_pe w23 UHF1',
                   'qb2.average_data1 UHF1_pf w23 UHF1',
                   'qb2.average_data1 UHF1_pg w23 UHF1']},
                 {'meas_obj_names': ['qb2'],
                  'keys_out': None,
                  'keys_in': ['qb2.average_data1 UHF1_pe w23 UHF1',
                   'qb2.average_data1 UHF1_pf w23 UHF1',
                   'qb2.average_data1 UHF1_pg w23 UHF1'],
                  'std_keys': 'previous qb2.get_std_deviation1',
                  'node_name': 'SingleQubitRBAnalysis'}]     
                  
                  
                  
Final example where some meas_obj_names are lists: multi-file 2QB RB

meas_obj_value_names_map = {
    'qb2': ['UHF1_pg w23 UHF1', 'UHF1_pe w23 UHF1', 'UHF1_pf w23 UHF1'],
    'qb4': ['UHF1_pg w45 UHF1', 'UHF1_pe w45 UHF1', 'UHF1_pf w45 UHF1'],
    'correlation': ['correlation']}
nr_files = 10
nr_cliffs = 8
nr_seeds_per_file = 10
 
pp = pp_mod.ProcessingPipeline()
# average data for all measued objects
pp.add_node('average_data', keys_in='raw',
            shape=(nr_files*nr_cliffs, nr_seeds_per_file), 
            meas_obj_names=list(movnm)))
# average data again for all measued objects
pp.add_node('average_data', 
            keys_in=[f'previous {mobj}.average_data' for mobj in movnm],
            shape=(nr_files, nr_cliffs), 
            averaging_axis=0, 
            meas_obj_names=list(movnm))                    
# RB only for qubit2
mobj = 'qb2' 
pp.add_node('SingleQubitRBAnalysis', 
            keys_in=f'previous {mobj}.average_data1',
            std_keys=f'previous {mobj}.get_std_deviation1'
            keys_out=None, # no keys out
            meas_obj_names=mobj)
   
   
"Raw" pipeline:
    [{'keys_in': 'raw',
      'shape': (80, 10),
      'meas_obj_names': ['qb2', 'qb4', 'correlation'],
      'node_name': 'average_data'},
     {'keys_in': ['previous qb2.average_data',
       'previous qb4.average_data',
       'previous correlation.average_data'],
      'shape': (10, 8),
      'averaging_axis': 0,
      'meas_obj_names': ['qb2', 'qb4', 'correlation'],
      'node_name': 'average_data'},
     {'meas_obj_names': 'qb2',
      'keys_out': None,
      'keys_in': 'previous qb2.average_data1',
      'std_keys': 'previous qb2.get_std_deviation1',
      'node_name': 'SingleQubitRBAnalysis'}]  
      
Final pipeline:
    call pp(movnm):
    [{'keys_in': ['UHF1_pe w23 UHF1', 'UHF1_pe w45 UHF1', 'UHF1_pf w23 UHF1',
                  'UHF1_pf w45 UHF1', 'UHF1_pg w23 UHF1', 'UHF1_pg w45 UHF1',
                  'correlation'],
      'shape': (80, 10),
      'meas_obj_names': ['qb2', 'qb4', 'correlation'],
      'node_name': 'average_data',
      'keys_out': ['qb2.average_data UHF1_pe w23 UHF1',
                   'qb4.average_data UHF1_pe w45 UHF1',
                   'qb2.average_data UHF1_pf w23 UHF1',
                   'qb4.average_data UHF1_pf w45 UHF1',
                   'qb2.average_data UHF1_pg w23 UHF1',
                   'qb4.average_data UHF1_pg w45 UHF1',
                   'correlation.average_data correlation']
     },              
     {'keys_in': ['correlation.average_data correlation',
                  'qb2.average_data UHF1_pe w23 UHF1',
                  'qb2.average_data UHF1_pf w23 UHF1',
                  'qb2.average_data UHF1_pg w23 UHF1',
                  'qb4.average_data UHF1_pe w45 UHF1',
                  'qb4.average_data UHF1_pf w45 UHF1',
                  'qb4.average_data UHF1_pg w45 UHF1'],
      'shape': (10, 8),
      'averaging_axis': 0,
      'meas_obj_names': ['qb2', 'qb4', 'correlation'],
      'node_name': 'average_data',
      'keys_out': ['correlation.average_data1 correlation',
                   'qb2.average_data1 UHF1_pe w23 UHF1',
                   'qb2.average_data1 UHF1_pf w23 UHF1',
                   'qb2.average_data1 UHF1_pg w23 UHF1',
                   'qb4.average_data1 UHF1_pe w45 UHF1',
                   'qb4.average_data1 UHF1_pf w45 UHF1',
                   'qb4.average_data1 UHF1_pg w45 UHF1']
     },
     {'meas_obj_names': ['qb2'],
      'keys_out': None,
      'keys_in': ['qb2.average_data1 UHF1_pe w23 UHF1',
                  'qb2.average_data1 UHF1_pf w23 UHF1',
                  'qb2.average_data1 UHF1_pg w23 UHF1'],
      'std_keys': 'previous qb2.get_std_deviation1',
      'node_name': 'SingleQubitRBAnalysis'
     }]                             
"""


class ProcessingPipeline(list):

    def __init__(self, node_name=None, from_dict_list=None,
                 global_keys_out_container='', meaj_obj_names=None,
                 **node_params):
        """
        Creates a processing pipeline for analysis_v3.
        :param node_name: name of the processing function
        :param from_dict_list: list of dicts to be instantiated as a
            ProcessingPipeline
        :param global_keys_out_container: str specifying a container for the
            keys_out that will be prepended to all the keys_out in all the nodes
        :param node_params: keyword arguments that will be passed to the
            processing function specified by node_name
        """
        super().__init__()
        self.global_keys_out_container = global_keys_out_container
        if node_name is not None:
            if 'keys_out_container' not in node_params:
                node_params['keys_out_container'] = \
                    self.global_keys_out_container
            if meaj_obj_names is not None and \
                    'meaj_obj_names' not in node_params:
                node_params['meaj_obj_names'] = meaj_obj_names
            node_params['node_name'] = node_name
            self.append(node_params)
        elif from_dict_list is not None:
            for d in from_dict_list:
                if isinstance(d, dict):
                    # assume that dicts have the same format as this class
                    self.append(d)
                else:
                    raise ValueError('Entries in list must be dicts.')

    def __add__(self, other):
        return ProcessingPipeline.cast_init(super().__add__(other))

    def __call__(self, meas_obj_value_names_map):
        """
        Resolves the keys_in and keys_out of a raw ProcessingPipeline, if they
        exist.
        :param meas_obj_value_names_map: dict of the form
            {mobj_name: readout_ch_list}
        :return: nothing, but changes self to the resolved ProcessingPipeline.
        Adds the flag was_resolved = True to the nodes that were resolved.
        """
        fallback_pipeline = deepcopy(self)
        pipeline = deepcopy(self)
        self.clear()
        for i, node_params in enumerate(pipeline):
            if node_params.get('was_resolved', False):
                # if node was already resolved, just add it to the pipeline
                self.append(node_params)
                continue

            try:
                if 'keys_in' not in node_params:
                    self.append(node_params)
                    continue

                meas_obj_names_raw = node_params.get(
                    'meas_obj_names', list(meas_obj_value_names_map))
                if isinstance(meas_obj_names_raw, str):
                    meas_obj_names_raw = [meas_obj_names_raw]
                joint_processing = node_params.pop('joint_processing', False)
                if joint_processing:
                    meas_obj_names = [','.join(meas_obj_names_raw)]
                else:
                    meas_obj_names = meas_obj_names_raw

                for mobj_name in meas_obj_names:
                    # mobjn is a string!
                    new_node_params = deepcopy(node_params)
                    new_node_params['joint_processing'] = joint_processing
                    if joint_processing and 'num_keys_out' \
                            not in new_node_params:
                        new_node_params['num_keys_out'] = 1
                    # get keys_in and any other key in node_params that
                    # contains keys_in
                    for k, v in new_node_params.items():
                        if 'keys_in' in k:
                            keys = self.resolve_keys_in(
                                v, mobj_name, meas_obj_value_names_map,
                                node_idx=i)
                            new_node_params[k] = keys
                    # get keys_out
                    keys_out_container = new_node_params.pop(
                        'keys_out_container')
                    if len(keys_out_container) == 0:
                        keys_out_container = mobj_name
                    if new_node_params.get('add_mobjn_container', True):
                        if mobj_name not in keys_out_container:
                            keys_out_container = \
                                f'{mobj_name}.{keys_out_container}' \
                                    if len(keys_out_container) > 0 else \
                                    f'{mobj_name}'
                    else:
                        new_node_params['meas_obj_names'] = \
                            keys_out_container.split('.')[0]

                    keys_out = self.resolve_keys_out(
                        keys_out_container=keys_out_container,
                        mobj_name=mobj_name,
                        meas_obj_value_names_map=meas_obj_value_names_map,
                        **new_node_params)
                    new_node_params['keys_out_container'] = keys_out_container
                    if keys_out is not None:
                        new_node_params['keys_out'] = keys_out
                    new_node_params['meas_obj_names'] = mobj_name.split(',')
                    # add flag that this node has been resolved
                    new_node_params['was_resolved'] = True
                    self.append(new_node_params)
            except Exception as e:
                # return unresolved pipeline
                self.clear()
                [self.append(node) for node in fallback_pipeline]
                raise e

    def add_node(self, node_name, **node_params):
        """
        Adds a node to self.
        :param node_name: name of the processing function
        :param node_params: keyword arguments that will be passed to the
            processing function specified by node_name
        """
        if 'keys_out_container' not in node_params:
            node_params['keys_out_container'] = self.global_keys_out_container
        node_params['node_name'] = node_name
        self.append(node_params)

    def resolve_keys_in(self, keys_in, mobj_name, meas_obj_value_names_map,
                        node_idx=None):
        """
        Converts the raw keys_in into complete paths inside a data_dict of
        analysis_v3.
        :param keys_in: UNresolved value corresponding to the "keys_in" key in
            a node.
        :param mobj_name: name of the measured object
        :param meas_obj_value_names_map: dict of the form
            {mobj_name: readout_ch_list}
        :param node_idx: index of the current node (being resolved)
        :return: resolved keys_in
        """
        prev_keys_out = []
        for d in self:
            if 'keys_out' in d:
                if d['keys_out'] is not None:
                    prev_keys_out += d['keys_out']

        # convert keys_in to a list if it is a string such that I can iterate
        # over the keys in
        keys_in_temp = deepcopy(keys_in)
        if isinstance(keys_in_temp, str):
            keys_in_temp = [keys_in_temp]

        mobj_value_names = hlp_mod.flatten_list(
            list(meas_obj_value_names_map.values()))
        keys_in = []
        for keyi in keys_in_temp:
            if keyi in mobj_value_names or keyi in prev_keys_out:
                keys_in += [keyi]
            elif keyi == 'raw':
                keys_in += [f'{mobjn}.{movn}' for mobjn in mobj_name.split(',')
                            for movn in meas_obj_value_names_map[mobjn]]
            elif 'previous' in keyi:
                if len(self) > 0:
                    # assumes that what comes after 'previous' is separated by
                    # a space
                    keys_in_split = keyi.split(' ')
                    if len(keys_in_split) > 1:
                        for mobjn in mobj_name.split(','):
                            keys_in_suffix = ' '.join(keys_in_split[1:])
                            keys_in_suffix = f'{mobjn}.{keys_in_suffix}'
                            # keys_in += [keys_in_suffix]
                            keys_in0 = \
                                hlp_mod.get_sublst_with_all_strings_of_list(
                                    lst_to_search=hlp_mod.flatten_list(
                                        prev_keys_out),
                                    lst_to_match=mobj_value_names)
                            keys_in += [ki for ki in keys_in0 if
                                        keys_in_suffix in ki]
                    else:
                        if node_idx is None:
                            raise ValueError('Currnet node index ("node_idx") '
                                             'unknown. "keys_in" cannot be '
                                             '"previous".')
                        if 'keys_out' not in self[node_idx-1]:
                            raise KeyError(f'The previous node '
                                           f'{self[node_idx-1]["node_name"]} '
                                           f'does not have the key "keys_out".')
                        keys_in += hlp_mod.get_sublst_with_all_strings_of_list(
                            lst_to_search=self[node_idx-1]['keys_out'],
                            lst_to_match=mobj_value_names)
                else:
                    raise ValueError('The first node in the pipeline cannot '
                                     'have "keys_in" = "previous".')

        if keys_in != keys_in_temp:
            try:
                keys_in.sort()
            except AttributeError:
                pass

        if len(keys_in) == 0 or keys_in is None:
            raise ValueError(f'No "keys_in" could be determined '
                             f'for {mobj_name} in the node with index '
                             f'{node_idx} and raw "keys_in" {keys_in_temp}.')
        return keys_in

    def resolve_keys_out(self, keys_in, keys_out_container, mobj_name,
                         meas_obj_value_names_map,
                         keys_out=(), **node_params):
        """
        Creates the key_out entry in the node as complete paths inside a
        data_dict of analysis_v3.
        :param keys_in: resolved (!) value corresponding to the "keys_in" key
            in a node
        :param keys_out_container: str specifying a container for the
            keys_out that will be prepended to all the keys_out generated by
            this function
        :param mobj_name: name of the measured object
        :param meas_obj_value_names_map: dict of the form
            {mobj_name: readout_ch_list}
        :param keys_out: unresolved value corresponding to "keys_out"
        :param node_params: keyword arguments of the current node (being
            resolved) and for which the keys_in have already been resolved
        :return: resolved keys_out
        """
        if keys_out is None:
            return keys_out

        prev_keys_out = []
        for d in self:
            if 'keys_out' in d:
                if d['keys_out'] is not None:
                    prev_keys_out += d['keys_out']

        if len(keys_out) == 0:
            prev_keys_out = []
            for d in self:
                if 'keys_out' in d:
                    if d['keys_out'] is not None:
                        prev_keys_out += d['keys_out']
            node_name = node_params['node_name']
            num_keys_out = node_params.get('num_keys_out', len(keys_in))
            # ensure all keys_in are used
            assert len(keys_in) % num_keys_out == 0
            n = len(keys_in) // num_keys_out

            keys_out = []
            for keyis in [keys_in[i*n: i*n + n] for i
                          in range(num_keys_out)]:
                # check whether node_name is already in keyis
                node_name_repeated = False
                keyis_mod = deepcopy(keyis)
                for i, keyi in enumerate(keyis):
                    if node_name in keyi:
                        node_name_repeated = True
                        # take the substring in keyi that comes after the
                        # already used node_name
                        keyis_mod[i] = keyi[len(keyi.split(' ')[0])+1:]

                node_name_to_use = deepcopy(node_name)
                if node_name_repeated:
                    # find how many times the node_name was used and add
                    # that number to the node_name
                    num_previously_used = len(
                        hlp_mod.get_sublst_with_all_strings_of_list(
                            lst_to_search=prev_keys_out,
                            lst_to_match=[f'{mobj_name}.{node_name}']))
                    node_name_to_use = f'{node_name}{num_previously_used}'

                keyo = f'{keys_out_container}.{node_name_to_use}'
                if keyo in prev_keys_out:
                    # appends the keyi name
                    keyo = ','.join([keyi.split('.')[-1] for keyi
                                     in keyis_mod])
                    keyo += [f'{keys_out_container}.'
                             f'{node_name_to_use} {keyo}']
                else:
                    # Append to keyo the channel names(s) that were passed in
                    # keys_in (some nodes process the data corresponding to
                    # only one or some subset of the meas_obj readout channels.
                    suffix = []
                    for mobjn in mobj_name.split(','):
                        suffix += hlp_mod.flatten_list([re.findall(ch, k)
                            for ch in meas_obj_value_names_map[mobjn]
                            for k in [keyi.split('.')[-1]
                                      for keyi in keyis_mod]])
                    suffix = ','.join(suffix)
                    if len(suffix):
                        keyo = f'{keyo} {suffix}'
                    else:
                        # len(suffix) will be 0 if the channel name(s) for the
                        # meas_obj is a joined string of several channel names
                        # corresponding to other meas_objs. For example,
                        # if we use a correlation_object, its channel name
                        # will be for example, UHF1_raw w2 UHF1,UHF1_raw w8 UHF1
                        # corresponding to the channel names of the two
                        # correlated qubits.
                        # In this case, just append the joined channel names
                        keyo = f'{keyo} ' \
                               f'{",".join(meas_obj_value_names_map[mobj_name])}'
                keys_out += [keyo]

        return keys_out

    def get_keys_out(self, meas_obj_names, node_name, keys_out_container=''):
        """
        Find keys_out in self that contain meas_obj_names, node_name, and
         keys_out_container
        :param meas_obj_names: list of measured object names
        :param node_name: name of the node
        :param keys_out_container: container for keys_out
        :return: list of keys_out
        """
        prev_keys_out = []
        for d in self:
            if 'keys_out' in d:
                if d['keys_out'] is not None:
                    prev_keys_out += d['keys_out']

        keys_out = []
        for keyo in prev_keys_out:
            for mobjn in meas_obj_names:
                string = '.'.join([mobjn, keys_out_container, node_name]) \
                    if len(keys_out_container) else '.'.join([mobjn, node_name])
                if string in keyo:
                    keys_out += [keyo]

        if len(keys_out) == 0:
            string = '.'.join(['mobjn', keys_out_container, node_name]) \
                if len(keys_out_container) else '.'.join(['mobjn', node_name])
            raise KeyError(
                f'No keys_out were found that contain "{string}", '
                f'for mobjn in {meas_obj_names}.'
                f'Make sure you use the correct keys_out_container.')

        return keys_out

    def find_node(self, dict_to_match, strict_comparison=False):
        """
        Find and return nodes in self whose have (k, v) pairs that match
        the (k, v) pairs in dict_to_match.
        :param dict_to_match: dict that is used to specify which nodes you are
            looking for. THE ORDER IN dict_to_match MATTERS! The function will
            go through the (k, v) pairs in dict_to_match and select the
            node(s) that contain v in/ have v as the value corresponding to k.
        :param strict_comparison: whether to only look for strict equality
            node[k] == v
        :return: list of found node(s)

        Assumptions:
            - if the value to match v is list/tuple of strings, then, if
                strict_comparison == False, this function will return a match
                even when the list v is a subset of node[k]. THIS FUNCTIONALITY
                ONLY EXISTS FOR LISTS OF STRINGS.
            - !!! type(v) must match type(node[k])
        """
        nodes = self
        found = False
        for k, v in dict_to_match.items():
            if isinstance(v, dict):
                raise NotImplementedError('There is no support for searching '
                                          'for dicts inside the nodes.')
            matched_nodes = []
            for node in nodes:
                if k not in node:
                    continue
                if type(node[k]) != type(v):
                    continue

                if strict_comparison:
                    condition = (v == node[k])
                else:
                    node_v = deepcopy(node[k])
                    v_temp = deepcopy(v)

                    if isinstance(node_v, str):
                        node_v = [node_v]
                    if isinstance(v_temp, str):
                        v_temp = [v_temp]

                    if hasattr(v_temp, '__iter__') and isinstance(v_temp[0], str):
                        condition = (len(
                            hlp_mod.get_sublst_with_all_strings_of_list(
                                node_v, v_temp)) > 0)
                    else:
                        condition = (v == node[k])
                matched_nodes += [node] if condition else []

            if len(matched_nodes):
                nodes = matched_nodes
                found = True
            else:
                print(f'No nodes found to have {k} = {v}')

        if not found:
            return []
        else:
            return nodes

    def show(self, meas_obj_value_names_map=None,
             save_name=None, save_folder=None, fmt='png'):
        """
        Produces a dependency tree of the ProcessingPipeline in self, using
        graphviz and pygraphviz.
        :param meas_obj_value_names_map: dict of the form
            {mobj_name: readout_ch_list}
        :param save_name: str with the name of the file that will be saved if
            this parameter is not None
        :param save_folder: path to where the file will be saved
        :param fmt: file format (png, pdf)
        :return: a pygraphviz.AGraph instance
        """
        pipeline = self
        if not any([node.get('was_resolved', False) for node in pipeline]):
            if meas_obj_value_names_map is None:
                raise ValueError('Please provide a resolved pipeline or the '
                                 'meas_obj_value_names_map.')
            pipeline = deepcopy(self)
            pipeline(meas_obj_value_names_map)

        G = pgv.AGraph(directed=True, dpi=600)
        node_names = [(f'{node.get("keys_out_container", "")} ' if
                       node.get("keys_out_container", "") else '') +
                      f'{node["node_name"]}' for node in pipeline]
        G.add_nodes_from(node_names)
        for node in pipeline[1:]:
            if 'keys_in' in node:
                prefix = (f'{node.get("keys_out_container", "")} ' if
                          node.get("keys_out_container", "") else '')
                node_name = f'{prefix}{node["node_name"]}'
                matched_nodes = pipeline.find_node(
                    dict_to_match={'keys_out': node['keys_in']})
                for n in matched_nodes:
                    prefix = (f'{n.get("keys_out_container", "")} ' if
                              n.get("keys_out_container", "") else '')
                    n_name = f'{prefix}{n["node_name"]}'
                    G.add_edge(n_name, node_name)
        G.layout('dot')

        if save_name is not None:
            if save_folder is None:
                save_folder = os.getcwd()
            G.draw(f'{save_folder}\{save_name}.{fmt}')

        return G

    @staticmethod
    def cast_init(processing_pipeline):
        """
        Recreates a ProcessingPipeline object from a string representation of
        a ProcessingPipeline instance, or a list of dicts.
        Avoids having "eval" statements throughout the codebase.
        Args:
            processing_pipeline: string representation of ProcessingPipeline
                instance, or a list of dicts

        Returns: ProcessingPipeline object
        """
        if isinstance(processing_pipeline, str):
            return ProcessingPipeline(from_dict_list=eval(processing_pipeline))
        else:
            return ProcessingPipeline(from_dict_list=processing_pipeline)