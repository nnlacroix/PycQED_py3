import logging
log = logging.getLogger(__name__)
# import sys
# this_module = sys.modules[__name__]

from copy import deepcopy
from pycqed.analysis_v3 import helper_functions as hlp_mod

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
        - without any input arguments: ProcessingPipeline()
        - or with input parameters:
         ProcessingPipeline(node_name, **node_params), where node_name is
         the name of the node, and **node_params all the parameters
         required by the node including the necessary keys described above
        ! For ease of use, keys_in can also be specified as
            - 'raw': the raw data corresponding to the measured object
            - 'previous': the keys_out of the previous node dictionary
             for the measured object.
            - 'previous measured_object_name.node_name': the keys_out of the
             dictionary for the measured object which has the node_name
        ! keys_out do not need to be specified by the user as they will be
         automatically constructed from the measured object name and the
         keys_in

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
 
pp = ppmod.ProcessingPipeline()
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
    """
    Creates a processing pipeline for analysis_v3.
    """
    def __init__(self, node_name=None, from_dict_list=None,
                 global_keys_out_container='', **node_params):
        super().__init__()
        self.global_keys_out_container = global_keys_out_container
        if node_name is not None:
            node_params['keys_out_container'] = self.global_keys_out_container
            node_params['node_name'] = node_name
            self.append(node_params)
        elif from_dict_list is not None:
            for d in from_dict_list:
                if isinstance(d, dict):
                    # assume that dicts have the same format as this class
                    self.append(d)
                else:
                    raise ValueError('Entries in list must be dicts.')

    def __call__(self, meas_obj_value_names_map):
        fallback_pipeline = deepcopy(self)
        pipeline = deepcopy(self)
        self.clear()
        for i, node_params in enumerate(pipeline):
            if node_params.get('was_resolved', False):
                # if node was already resolved, just add it
                self.append(node_params)
                continue

            try:
                if 'keys_in' not in node_params:
                    self.append(node_params)
                    continue
                meas_obj_names_raw = node_params['meas_obj_names']
                if isinstance(meas_obj_names_raw, str):
                    meas_obj_names_raw = [meas_obj_names_raw]
                joint_processing = node_params.pop('joint_processing', False)
                if joint_processing:
                    meas_obj_names = [','.join(meas_obj_names_raw)]
                    mobj_value_names = hlp_mod.flatten_list(
                        meas_obj_value_names_map.values())
                else:
                    meas_obj_names = meas_obj_names_raw

                for mobj_name in meas_obj_names:
                    # mobjn is a string!
                    new_node_params = deepcopy(node_params)
                    new_node_params['meas_obj_names'] = mobj_name
                    # get the value names corresponding to the measued
                    # object name
                    if not joint_processing:
                        mobj_value_names = meas_obj_value_names_map[mobj_name]
                    # get keys_in and any other key in node_params that
                    # contains keys_in
                    for k, v in new_node_params.items():
                        if 'keys_in' in k:
                            keys = self.process_keys_in(
                                v, mobj_name, mobj_value_names,
                                node_idx=i)
                            new_node_params[k] = keys
                    # get keys_out
                    keys_out_container = new_node_params.pop(
                        'keys_out_container', mobj_name)
                    if mobj_name not in keys_out_container:
                        keys_out_container = \
                            f'{mobj_name}.{keys_out_container}' \
                                if len(keys_out_container) > 0 else \
                                f'{mobj_name}'
                    keys_out = self.process_keys_out(
                        keys_out_container=keys_out_container,
                        **new_node_params)
                    new_node_params['keys_out_container'] = keys_out_container
                    if keys_out is not None:
                        new_node_params['keys_out'] = keys_out
                    # add flag that this node has been resolved
                    new_node_params['was_resolved'] = True
                    self.append(new_node_params)
            except Exception as e:
                # return unresolved pipeline
                self.clear()
                [self.append(node) for node in fallback_pipeline]
                raise e

    def add_node(self, node_name, **node_params):
        node_params['keys_out_container'] = self.global_keys_out_container
        node_params['node_name'] = node_name
        self.append(node_params)

    def process_keys_in(self, keys_in, mobj_name, mobj_value_names,
                        node_idx=None):
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

        keys_in = []
        for keyi in keys_in_temp:
            if keyi in mobj_value_names or keyi in prev_keys_out:
                keys_in += [keyi]
            elif keyi == 'raw':
                keys_in += mobj_value_names
            elif 'previous' in keyi:
                if len(self) > 0:
                    # assumes that what comes after 'previous' is separated by
                    # a space
                    keys_in_split = keyi.split(' ')
                    if len(keys_in_split) > 1:
                        keys_in_suffix = ' '.join(keys_in_split[1:])
                        keys_in_suffix = f'{mobj_name}.{keys_in_suffix}'
                        keys_in0 = hlp_mod.get_sublst_with_all_strings_of_list(
                            lst_to_search=hlp_mod.flatten_list(prev_keys_out),
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
        try:
            keys_in.sort()
        except AttributeError:
            pass

        if len(keys_in) == 0 or keys_in is None:
            raise ValueError(f'No "keys_in" could be determined '
                             f'for {mobj_name} in the node with index '
                             f'{node_idx} and raw "keys_in" {keys_in_temp}.')
        return keys_in

    def process_keys_out(self, keys_in, keys_out_container, keys_out=(),
                         **node_params):
        if keys_out is None:
            return keys_out

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
                    # find how many times was the node_name used and add
                    # 1 to that
                    num_previously_used = len(
                        hlp_mod.get_sublst_with_all_strings_of_list(
                            lst_to_search=[node_name],
                            lst_to_match=prev_keys_out))
                    node_name_to_use = f'{node_name}{num_previously_used+1}'

                keyo = ','.join([keyi.split('.')[-1] for keyi
                                 in keyis_mod])
                keys_out += [f'{keys_out_container}.'
                             f'{node_name_to_use} {keyo}']
        return keys_out