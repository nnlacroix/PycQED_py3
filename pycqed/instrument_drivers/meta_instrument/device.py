"""
The Device class is intended to be used for two main tasks:
    * store two-qubit gate parameters
    * run multi-qubit standard experiments

The structure is chosen to resemble the one of the QuDev_transmon class. As such, the two-qubit gate parameters
are stored as instrument parameters of the device, as is the case for single-qubit gates for the QuDev_transmon
class.

* add_2qb_gate *
New two-qubit gates can be added using the add_2qb_gate(gate_name, pulse_type) method. It takes the gate name and the
pulse type intended to be used for the gate as input. It scans the pulse_library.py file for the provided pulse type.
Using the new pulse_params() method of the pulse, the relevant two qubit gate parameters can be added for each connected
qubit.

* get_operation_dict *
As for the QuDev_transmon class the Device class has the ability to return a dictionary of all device operations
(single- and two-qubit) in the form of a dictionary, using the get_operation_dict method.

* multi-qubit experiments *
For regularly used methods we place wrapper functions calling methods from the multi_qubit_module. This list is readily
extended by further methods from the multi_qubit_module or other modules.
"""

# General imports
import logging
from copy import deepcopy

import pycqed.measurement.multi_qubit_module as mqm
import pycqed.instrument_drivers.meta_instrument.qubit_objects.QuDev_transmon as qdt
import pycqed.measurement.waveform_control.pulse as bpl
from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import (ManualParameter, InstrumentRefParameter)
from qcodes.utils import validators as vals

log = logging.getLogger(__name__)


class Device(Instrument):
    def __init__(self, name, qubits, connectivity_graph, **kw):
        """
        Instantiates device instrument and adds its parameters.

        Args:
            name (str): name of the device
            qubits (list of QudevTransmon or names of QudevTransmon objects): qubits of the device
            connectivity_graph: list of elements of the form [qb1, qb2] with qb1 and qb2 QudevTransmon objects or names
                         thereof. qb1 and qb2 should be physically connected on the device.
        """
        super().__init__(name, **kw)

        qb_names = [qb if isinstance(qb, str) else qb.name for qb in qubits]
        qubits = [qb if not isinstance(qb, str) else self.find_instrument(qb) for qb in qubits]
        connectivity_graph = [[qb1 if isinstance(qb1, str) else qb1.name,
                               qb2 if isinstance(qb2, str) else qb2.name] for [qb1, qb2] in connectivity_graph]

        for qb in qubits:
            setattr(self, qb.name, qb)

        self.add_parameter('qubits',
                           vals=vals.Lists(),
                           initial_value=qubits,
                           parameter_class=ManualParameter)
        self.add_parameter('qb_names',
                           vals=vals.Lists(),
                           initial_value=qb_names,
                           parameter_class=ManualParameter)

        self._operations = {}  # dictionary containing dictionaries of operations with parameters

        # Instrument reference parameters
        self.add_parameter('instr_mc',
                           parameter_class=InstrumentRefParameter)
        self.add_parameter('instr_pulsar',
                           parameter_class=InstrumentRefParameter)
        self.add_parameter('instr_dc_source',
                           parameter_class=InstrumentRefParameter)
        self.add_parameter('instr_trigger',
                           parameter_class=InstrumentRefParameter)

        self.add_parameter('connectivity_graph',
                           vals=vals.Lists(),
                           label="Qubit Connectivity Graph",
                           docstring="Stores the connections between the qubits "
                                     "in form of a list of lists [qbi_name, qbj_name]",
                           parameter_class=ManualParameter,
                           initial_value=connectivity_graph
                           )
        self.add_parameter('last_calib',
                           vals=vals.Strings(),
                           initial_value='',
                           docstring='stores timestamp of last calibration',
                           parameter_class=ManualParameter)

        self.add_parameter('operations',
                           docstring='a list of operations on the device, without single QB operations.',
                           get_cmd=self._get_operations)
        self.add_parameter('two_qb_gates',
                           vals=vals.Lists(),
                           initial_value=[],
                           docstring='stores all two qubit gate names',
                           parameter_class=ManualParameter)

        # Pulse preparation parameters
        default_prep_params = dict(preparation_type='wait',
                                   post_ro_wait=1e-6, reset_reps=1)

        self.add_parameter('preparation_params', parameter_class=ManualParameter,
                           initial_value=default_prep_params, vals=vals.Dict())

    # General Class Methods

    def add_operation(self, operation_name):
        """
        Adds the name of an operation to the operations dictionary.

        Args:
            operation_name (str): name of the operation
        """

        self._operations[operation_name] = {}

    def add_pulse_parameter(self, operation_name, parameter_name, argument_name,
                            initial_value=None, **kw):
        """
        Adds a pulse parameter to an operation. Makes sure that parameters are not duplicated.
        Adds the pulse parameter to the device instrument.

        Args:
            operation_name (tuple): name of operation in format (gate_name, qb1, qb2)
            parameter_name (str): name of parameter
            argument_name (str): name of the argument that is added to the operations dict
            initial_value: initial value of parameter
        """
        if parameter_name in self.parameters:
            raise KeyError(
                'Duplicate parameter name {}'.format(parameter_name))

        if operation_name in self.operations().keys():
            self._operations[operation_name][argument_name] = parameter_name
        else:
            raise KeyError('Unknown operation {}, add '.format(operation_name) +
                           'first using add operation')

        self.add_parameter(parameter_name,
                           initial_value=initial_value,
                           parameter_class=ManualParameter, **kw)

    def _get_operations(self):
        """
        Private method that is used as getter function for operations parameter
        """
        return self._operations

    def get_operation_dict(self, operation_dict=None):
        """
        Returns the operations dictionary of the device and qubits, combined with the input
        operation_dict.

        Args:
            operation_dict (dict): input dictionary the operations should be added to

        Returns:
            operation_dict (dict): dictionary containing both qubit and device operations

        """
        if operation_dict is None:
            operation_dict = dict()

        # add 2qb operations
        two_qb_operation_dict = {}
        for op_tag, op in self.operations().items():
            # op_tag is the tuple (gate_name, qb1, qb2) and op the dictionary of the
            # operation

            # Add both qubit combinations to operations dict
            # Still return a string instead of tuple as keys to be consisten
            # with QudevTransmon class
            for op_name in [op_tag[0] + ' ' + op_tag[1] + ' ' + op_tag[2],
                            op_tag[0] + ' ' + op_tag[2] + ' ' + op_tag[1]]:
                two_qb_operation_dict[op_name] = {}
                for argument_name, parameter_name in op.items():
                    two_qb_operation_dict[op_name][argument_name] = \
                        self.get(parameter_name)

        operation_dict.update(two_qb_operation_dict)

        # add sqb operations
        for qb in self.qubits():
            operation_dict.update(qb.get_operation_dict())

        return operation_dict

    def get_qb(self, qb_name):
        """
        Wrapper: Returns the qubit instance with name qb_name

        Args:
            qb_name (str): name of the qubit
        Returns:
            qubit instrument with name qubit_name

        """
        return self.find_instrument(qb_name)

    def get_pulse_par(self, gate_name, qb1, qb2, param):
        """
        Returns the value of a two qubit gate parameter.

        Args:
            gate_name (str): Name of the gate
            qb1 (str, QudevTransmon): Name of one qubit
            qb2 (str, QudevTransmon): Name of other qubit
            param (str): name of parameter
        Returns:
            the value of the parameter
        """

        if isinstance(qb1, qdt.QuDev_transmon):
            qb1_name = qb1.name
        else:
            qb1_name = qb1

        if isinstance(qb2, qdt.QuDev_transmon):
            qb2_name = qb2.name
        else:
            qb2_name = qb2

        try:
            self.get(f'{gate_name}_{qb1_name}_{qb2_name}_{param}')
        except KeyError:
            try:
                self.get(f'{gate_name}_{qb2_name}_{qb1_name}_{param}')
            except KeyError:
                raise ValueError(f'Parameter {param} for the gate '
                                 f'{gate_name} {qb1_name} {qb2_name} '
                                 f'does not exist!')

    def get_prep_params(self, qb_list):
        """
        Returns the preparation paramters for all qubits in qb_list.

        Args:
            qb_list (list): list of qubit names or objects

        Returns:
            dictionary of preparation parameters
        """

        for i, qb in enumerate(qb_list):
            if isinstance(qb, str):
                qb_list[i] = self.get_qb(qb)

        # threshold_map has to be updated for all qubits
        thresh_map = {}
        for prep_params in [qb.preparation_params() for qb in qb_list]:
            if 'threshold_mapping' in prep_params:
                thresh_map.update(prep_params['threshold_mapping'])

        prep_params = deepcopy(self.preparation_params())
        prep_params['threshold_mapping'] = thresh_map

        return prep_params

    def set_pulse_par(self, gate_name, qb1, qb2, param, value):
        """
        Sets a value to a two qubit gate parameter.

        Args:
            gate_name (str): Name of the gate
            qb1 (str, QudevTransmon): Name of one qubit
            qb2 (str, QudevTransmon): Name of other qubit
            param (str): name of parameter
            value: value of parameter
        """

        if isinstance(qb1, qdt.QuDev_transmon):
            qb1_name = qb1.name
        else:
            qb1_name = qb1

        if isinstance(qb2, qdt.QuDev_transmon):
            qb2_name = qb2.name
        else:
            qb2_name = qb2

        try:
            self.set(f'{gate_name}_{qb1_name}_{qb2_name}_{param}', value)
        except KeyError:
            try:
                self.set(f'{gate_name}_{qb2_name}_{qb1_name}_{param}', value)
            except KeyError:
                raise ValueError(f'Parameter {param} for the gate '
                                 f'{gate_name} {qb1_name} {qb2_name} '
                                 f'does not exist!')

    def check_connection(self, qubit_a, qubit_b, connectivity_graph=None, raise_exception=True):
        """
        Checks whether two qubits are connected.

        Args:
            qubit_a (str, QudevTransmon): Name of one qubit
            qubit_b (str, QudevTransmon): Name of other qubit
            connectivity_graph: custom connectivity graph. If None device graph will be used.
            raise_exception (Bool): flag whether an error should be raised if qubits are not connected.
        """

        if connectivity_graph is None:
            connectivity_graph = self.connectivity_graph()
        # convert qubit object to name if necessary
        if not isinstance(qubit_a, str):
            qubit_a = qubit_a.name
        if not isinstance(qubit_b, str):
            qubit_b = qubit_b.name
        if [qubit_a, qubit_b] not in connectivity_graph and [qubit_b, qubit_a] not in connectivity_graph:
            if raise_exception:
                raise ValueError(f'Qubits {[qubit_a, qubit_b]}  are not connected!')
            else:
                log.warning('Qubits are not connected!')
                # TODO: implement what we want in case of swap (e.g. determine shortest path of swaps)

    def add_2qb_gate(self, gate_name, pulse_type='BufferedNZFLIPPulse'):
        """
        Method to add a two qubit gate with name gate_name with parameters for
        all connected qubits. The parameters including their default values are taken
        for the Class pulse_type in pulse_library.py.

        Args:
            gate_name (str): Name of gate
            pulse_type (str): Two qubit gate class from pulse_library.py
        """

        # add gate to list of two qubit gates
        self.set('two_qb_gates', self.get('two_qb_gates') + [gate_name])

        # find pulse module
        pulse_func = None
        for module in bpl.pulse_libraries:
            try:
                pulse_func = getattr(module, pulse_type)
            except AttributeError:
                pass
        if pulse_func is None:
            raise KeyError('pulse_type {} not recognized'.format(pulse_type))

        # for all connected qubits add the operation with name gate_name
        for [qb1, qb2] in self.connectivity_graph():
            op_name = (gate_name, qb1, qb2)
            par_name = f'{gate_name}_{qb1}_{qb2}'
            self.add_operation(op_name)

            # get default pulse params for the pulse type
            params = pulse_func.pulse_params()

            for param, init_val in params.items():
                self.add_pulse_parameter(op_name, par_name + '_' + param, param,
                                         initial_value=init_val)

            # Update flux pulse channels
            for qb, c in zip([qb1, qb2], ['channel', 'channel2']):
                if c in params:
                    channel = self.get_qb(qb).flux_pulse_channel()
                    if channel == '':
                        raise ValueError(f'No flux pulse channel defined for {qb}!')
                    else:
                        self.set_pulse_par(gate_name, qb1, qb2, c, channel)

    # Wrapper functions for Device algorithms #

    def measure_J_coupling(self, qbm, qbs, freqs, cz_pulse_name, **kwargs):

        """
        Wrapper function for the multi_qubit_module method measure_J_coupling.
        """

        mqm.measure_J_coupling(qbm, qbs, freqs, cz_pulse_name, **kwargs)

    def measure_tomography(self, qubits, prep_sequence, state_name, **kwargs):
        """
        Wrapper function for the multi_qubit_module method measure_two_qubit_randomized_benchmarking.
        """

        mqm.measure_tomography(self, qubits, prep_sequence, state_name, **kwargs)

    def measure_two_qubit_randomized_benchmarking(self, qb1, qb2, cliffords, nr_seeds, cz_pulse_name, **kwargs):
        """
        Wrapper function for the multi_qubit_module method measure_two_qubit_randomized_benchmarking.
        """

        mqm.measure_two_qubit_randomized_benchmarking(self, qb1, qb2, cliffords, nr_seeds, cz_pulse_name, **kwargs)

    def measure_chevron(self, qbc, qbt, hard_sweep_params, soft_sweep_params, cz_pulse_name, **kwargs):
        '''
        Wrapper function for the multi_qubit_module method measure_chevron.
        '''

        mqm.measure_chevron(self, qbc, qbt, hard_sweep_params, soft_sweep_params, cz_pulse_name, **kwargs)

    def measure_cphase(self, qbc, qbt, soft_sweep_params, cz_pulse_name, **kwargs):
        '''
        Wrapper function for the multi_qubit_module method measure_cphase.
        '''

        mqm.measure_cphase(self, qbc, qbt, soft_sweep_params, cz_pulse_name, **kwargs)

    def measure_dynamic_phases(self, qbc, qbt, cz_pulse_name, **kwargs):
        """
        Wrapper function for the multi_qubit_module method measure_dynamic_phase.
        """

        mqm.measure_dynamic_phases(self, qbc, qbt, cz_pulse_name, **kwargs)
