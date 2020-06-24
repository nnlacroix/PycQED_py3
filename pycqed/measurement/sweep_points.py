import logging
log = logging.getLogger(__name__)
from collections import OrderedDict
from copy import deepcopy

class SweepPoints(list):
    """
    This class is used to create sweep points for any measurement.
    The SweepPoints object is a list of dictionaries of the form:
        [
            # 1st sweep dimension
            {param_name0: (values, unit, label),
             param_name1: (values, unit, label),
            ...
             param_nameN: (values, unit, label)},

            # 2nd sweep dimension
            {param_name0: (values, unit, label),
             param_name1: (values, unit, label),
            ...
             param_nameN: (values, unit, label)},

             .
             .
             .

            # D-th sweep dimension
            {param_name0: (values, unit, label),
             param_name1: (values, unit, label),
            ...
             param_nameN: (values, unit, label)},
        ]

    Example how to use this class to create a 2D sweep for 3 qubits, where
    the first (hard) sweep is over amplitudes and the :

    sp = SweepPoints()
    for qb in ['qb1', 'qb2', 'qb3']:
        sp.add_sweep_parameter(f'lengths_{qb}', np.linspace(10e-9, 1e-6, 80),
        's', 'Pulse length, $L$')
    sp.add_sweep_dimension()
    for qb in ['qb1', 'qb2', 'qb3']:
        sp.add_sweep_parameter(f'amps_{qb}', np.linspace(0, 1, 20),
        'V', 'Pulse amplitude, $A$')
    """
    def __init__(self, param_name=None, values=None, unit='', label=None,
                 from_dict_list=None):
        super().__init__()
        if param_name is not None and values is not None:
            if label is None:
                label = param_name
            self.append({param_name: (values, unit, label)})
        elif from_dict_list is not None:
            for d in deepcopy(from_dict_list):
                if len(d) == 0 or isinstance(list(d.values())[0], tuple):
                    # assume that dicts have the same format as this class
                    self.append(d)
                else:
                    # import from a list of sweep dicts in the old format
                    self.append({k: (v['values'],
                                     v.get('unit',''),
                                     v.get('label', k))
                                 for k, v in d.items()})

    def add_sweep_parameter(self, param_name, values, unit='', label=None):
        if label is None:
            label = param_name
        if len(self) == 0:
            self.append({param_name: (values, unit, label)})
        else:
            self[-1].update({param_name: (values, unit, label)})

    def add_sweep_dimension(self):
        self.append(dict())

    def get_sweep_dimension(self, dimension='all'):
        """
        Returns the sweep dict of the sweep dimension specified by dimension.
        :param dimension: int > 0 specifying a sweep dimension or
            the string 'all'
        :return: self if dimension == 'all', else self[dimension-1]
        """
        if dimension == 'all':
            return self
        else:
            assert dimension > 0, 'Dimension must be > 0.'
            if len(self) < dimension:
                raise ValueError(f'Dimension {dimension} not found.')
            return self[dimension-1]

    def get_sweep_params_description(self, param_names, dimension='all'):
        """
        Get the sweep tuples for the sweep parameters param_names if they are
        found in the sweep dimension dict specified by dimension.
        :param param_names: string or list of strings corresponding to keys in
            the dictionaries in self
        :param dimension: see docstring for get_sweep_dimension
        :return:
            If the param_names are found in self or self[dimension]:
            if param_names is string: string with the sweep tuples of each
                param_names in the sweep dimension dict specified by dimension.
            if param_names is list: list with the property of each
                param_names in the sweep dimension dict specified by dimension.
            if param_names is None: string corresponding to the
                first sweep parameter in the sweep dimension dict
            If none of param_names are found, raises KeyError.
        """
        sweep_points_dim = self.get_sweep_dimension(dimension)
        is_list = True
        if not isinstance(param_names, list):
            param_names = [param_names]
            is_list = False
        sweep_param_values = []
        if isinstance(sweep_points_dim, list):
            for sweep_dim_dict in sweep_points_dim:
                for pn in param_names:
                    if pn in sweep_dim_dict:
                        sweep_param_values += [sweep_dim_dict[pn]]
        else:  # it is a dict
            for pn in param_names:
                if pn in sweep_points_dim:
                    sweep_param_values += [sweep_points_dim[pn]]

        if len(sweep_param_values) == 0:
            s = "sweep points" if dimension == "all" else f'sweep dimension ' \
                                                          f'{dimension}'
            raise KeyError(f'{param_names} not found in {s}.')

        if is_list:
            return sweep_param_values
        else:
            return sweep_param_values[0]

    def get_sweep_params_property(self, property, dimension, param_names=None):
        """
        Get a property of the sweep parameters param_names in self.
        :param property: str with the name of a sweep param property. Can be
            "values", "unit", "label."
        :param dimension: int > 0 specifying a sweep dimension
        :param param_names: None, or string or list of strings corresponding to
            keys in the sweep dimension specified by dimension
        :return:
            if param_names is string: the property of the sweep parameter
                specified in param_names in the sweep dimension dict specified
                by dimension.
            if param_names is list: list with the property of each
                param_names in the sweep dimension dict specified by dimension.
            if param_names is None: property corresponding to the
                first sweep parameter in the sweep dimension dict
        """
        assert isinstance(dimension, int), 'Dimension must be an integer > 0.'
        properties_dict = {'values': 0, 'unit': 1, 'label': 2}
        if param_names is None:
            if len(self.get_sweep_dimension(dimension)) == 0:
                return [] if property == 'values' else ''
            else:
                return next(iter(self.get_sweep_dimension(
                dimension).values()))[properties_dict[property]]
        else:
            # FIXME: probably we need somethign like the above
            #   if len(self.get_sweep_dimension(dimension)) == 0:
            # also here
            if isinstance(param_names, list):
                return [pnd[properties_dict[property]] for pnd in
                        self.get_sweep_params_description(param_names,
                                                          dimension)]
            else:
                return self.get_sweep_params_description(
                    param_names, dimension)[properties_dict[property]]

    def get_meas_obj_sweep_points_map(self, measured_objects):
        """
        Assumes the order of params in each sweep dimension corresponds to
        the order of keys in keys_list

        :param measured_objects: list of strings to be used as keys in the
            returned dictionary. These are the measured object names
        :return: {keys[k]: list(d)[k] for d in self for k in measured_objects}
        """
        if not isinstance(measured_objects, list):
            measured_objects = [measured_objects]
        for i, mobj in enumerate(measured_objects):
            if hasattr(mobj, 'name'):
                measured_objects[i] = mobj.name

        if len(measured_objects) != len(self[0]):
            raise ValueError('The number of keys and number of sweep '
                             'parameters do not match.')

        sweep_points_map = OrderedDict()
        for i, key in enumerate(measured_objects):
            sweep_points_map[key] = [list(d)[i] for d in self]

        return sweep_points_map

    def update(self, sweep_points):
        while len(self) < len(sweep_points):
            self.add_sweep_dimension()
        for d, u in zip(self, sweep_points):
            d.update(u)
