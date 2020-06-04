import logging
log = logging.getLogger(__name__)
from collections import OrderedDict


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
            for d in from_dict_list:
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

    def get_meas_obj_sweep_points_map(self, measured_objects):
        """
        Assumes the order of params in each sweep dimension corresponds to
        the order of keys in keys_list

        :param measured_objects: list of strings to be used as keys in the
            returned dictionary. These are the measured object names
        :return: {keys[k]: list(d)[k] for d in self for k in measured_objects}
        """

        if len(measured_objects) != len(self[0]):
            raise ValueError('The number of keys and number of sweep '
                             'parameters do not match.')

        sweep_points_map = OrderedDict()
        for i, key in enumerate(measured_objects):
            sweep_points_map[key] = [list(d)[i] for d in self]

        return sweep_points_map


