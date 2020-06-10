import time
import logging

from qcodes import Instrument
from qcodes.instrument.parameter import InstrumentRefParameter
from qcodes.utils.validators import Enum, Ints, Numbers

class QudevDisplacer(Instrument):
    """
    A meta-instrument containing the two sub-modules that make up a
    displacer board along with routines for optimizing the parameters
    for a TWPA
    """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

        # Add instrument references
        self.add_parameter('attenuator',
                           parameter_class=InstrumentRefParameter)
        self.add_parameter('phase_shifter',
                           parameter_class=InstrumentRefParameter)

        # Add displacer board control parameters
        self.add_parameter(
            'attenuator_position',
            label='Attenuator Position',
            unit='',
            get_cmd=self.attenuator.instrument.position_normalized,
            set_cmd=self.attenuator.instrument.position_normalized,
            docstring=('Attenuator motor position (normalized)'
                       'Min value: 0.0'
                       'Max value: 1.0'))

        self.add_parameter(
            'phase_shifter_position',
            label='Phase Shifter Position',
            unit='',
            get_cmd=self.phase_shifter.instrument.position_normalized,
            set_cmd=self.phase_shifter.instrument.position_normalized,
            docstring=('Phase shifter motor position (normalized)'
                       'Min value: 0.0'
                       'Max value: 1.0'))