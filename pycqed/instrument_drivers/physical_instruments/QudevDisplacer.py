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
        self.add_parameter('instr_attenuator',
                           parameter_class=InstrumentRefParameter)
        self.add_parameter('instr_phase_shifter',
                           parameter_class=InstrumentRefParameter)

        # Add displacer board control parameters
        self.add_parameter('attenuator_position',
                           label='Attenuator Position',
                           unit='',
                           get_cmd=)