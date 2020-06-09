import logging
import numpy as np

from qcodes import VisaInstrument
from qcodes.utils.validators import Enum, Ints, Numbers

"""
Requires
    pyVISA-py
    PySerial
"""

log = logging.getLogger(__name__)


class NanotecSMI33(VisaInstrument):
    """
    This is a driver for the Nanotec SMI33 Motor controller

    Status: not tested

    Only a subset of all features have been included
    """
    def __init__(self, name: str, address, controller_id: str, **kwargs) -> None:
        """
        Args:
            name: The name of this instance
            address: The address of the controller
            controller_id: The id of the motor which can be either '*' or a number
                between 1 and 254
            kwargs: Additional keyword arguments
        """
        super().__init__(name, address, terminator='\r', visalib='@py',
                         **kwargs)

        # Set correct serial parameters
        self.visa_handle.baud_rate = 115200
        self.visa_handle.read_termination = '\r'
        self.visa_handle.write_termination = '\r'

        if controller_id not in [str(i) for i in range(1, 255)] + ['*']:
            raise ValueError('controller_id must be * or a number from 1 to 254')
        self.controller_id = controller_id
        self._start_character = '#'

        self.add_parameter(
            'acceleration',
            label='Acceleration',
            unit='',
            get_cmd=(lambda cmd='b': self.build_get_string(cmd)),
            set_cmd=(lambda x, cmd='b': self.build_set_string(cmd, x)),
            get_parser=(lambda x, cmd='b': int(self.parse_cmd_response(
                x,
                self.build_get_string(cmd)))),
            max_val_age=0,
            vals=Ints(min_value=1,
                      max_value=65535),
            docstring=('Acceleration rate'
                       'Min value: 1'
                       'Max value: 65535'))

        self.add_parameter(
            'acceleration_mode',
            label='Acceleration Mode',
            unit='',
            get_cmd=(lambda cmd=':ramp_mode': self.build_get_string(cmd)),
            set_cmd=(lambda x, cmd=':ramp_mode': self.build_set_string(cmd, x)),
            get_parser=(lambda x, cmd=':ramp_mode': int(self.parse_cmd_response(
                x,
                self.build_get_string(cmd)))),
            max_val_age=0,
            val_mapping={'Trapezoidal': 0,
                         'Sinusoidal': 1,
                         'Jerk-free': 2},
            docstring='Set the acceleration mode')

        self.add_parameter(
            'acceleration_jerk',
            label='Acceleration Jerk',
            unit='',
            get_cmd=(lambda cmd=':b': self.build_get_string(cmd)),
            set_cmd=(lambda x, cmd=':b': self.build_set_string(cmd, x)),
            get_parser=(lambda x, cmd=':b': int(self.parse_cmd_response(
                x,
                self.build_get_string(cmd)))),
            vals=Ints(min_value=1,
                      max_value=100000000),
            docstring=('Acceleration jerk'
                       'Min value: 1'
                       'Max value: 100000000'))

        self.add_parameter(
            'braking',
            label='Breaking',
            unit='',
            get_cmd=(lambda cmd='B': self.build_get_string(cmd)),
            set_cmd=(lambda x, cmd='B': self.build_set_string(cmd, x)),
            get_parser=(lambda x, cmd='B': int(self.parse_cmd_response(
                x,
                self.build_get_string(cmd)))),
            max_val_age=0,
            vals=Ints(min_value=0,
                      max_value=65535),
            docstring=('Braking rate'
                       'Min value: 1'
                       'Max value: 65535'))

        self.add_parameter(
            'braking_jerk',
            label='Braking Jerk',
            unit='',
            get_cmd=(lambda cmd=':B': self.build_get_string(cmd)),
            set_cmd=(lambda x, cmd=':B': self.build_set_string(cmd, x)),
            get_parser=(lambda x, cmd=':B': int(self.parse_cmd_response(
                x,
                self.build_get_string(cmd)))),
            vals=Ints(min_value=1,
                      max_value=100000000),
            docstring=('Breaking jerk'
                       'Min value: 1'
                       'Max value: 100000000'))

        self.add_parameter(
            'continuation_record',
            label='Continuation Record',
            unit='',
            get_cmd=(lambda cmd='N': self.build_get_string(cmd)),
            set_cmd=(lambda x, cmd='N': self.build_set_string(cmd, x)),
            get_parser=(lambda x, cmd='N': int(self.parse_cmd_response(
                x,
                self.build_get_string(cmd)))),
            vals=Ints(min_value=0,
                      max_value=32),
            docstring=('Record in EEPROM to continue with after finishing'
                       'the current record.'
                       'Min value is 0'
                       'Max value is 32'
                       'If the value is set to 0, this setting has no'
                       'effect.'))

        self.add_parameter(
            'command_response',
            label='Command Response',
            unit='',
            get_cmd=False,
            set_cmd=(lambda x, cmd='|': self.build_set_string(cmd, x)),
            val_mapping={'Disabled': 0,
                         'Enabled': 1},
            docstring=('Enable or disable command response'
                       'If disabled, the controller will obey all commands'
                       'that are sent without responding.'
                       'Note that this disables response to any commands'
                       'including those reading out other settings'))

        self.add_parameter(
            'digital_input_1_function',
            label='Digital Input 1 Function',
            unit='',
            get_cmd=(lambda cmd=':port_in_a': self.build_get_string(cmd)),
            set_cmd=(lambda x, cmd=':port_in_a':
                     self.build_set_string(cmd, x)),
            get_parser=(lambda x, cmd=':port_in_a':
                        int(self.parse_cmd_response(
                            x,
                            self.build_get_string(cmd)))),
            max_val_age=0,
            val_mapping={'UserDefined': 0,
                         'StartRecord/ErrorReset': 1,
                         'RecordSelectBit0': 2,
                         'RecordSelectBit1': 3,
                         'RecordSelectBit2': 4,
                         'RecordSelectBit3': 5,
                         'RecordSelectBit4': 6,
                         'ExternalLimitSwitch': 7,
                         'Trigger': 8,
                         'Direction': 9,
                         'Enable': 10,
                         'Clock': 11,
                         'ClockDirectionMode1': 12,
                         'ClockDirectionMode2': 13},
            docstring='Function of digital input 1')

        self.add_parameter(
            'digital_input_2_function',
            label='Digital Input 2 Function',
            unit='',
            get_cmd=(lambda cmd=':port_in_b': self.build_get_string(cmd)),
            set_cmd=(lambda x, cmd=':port_in_b':
                     self.build_set_string(cmd, x)),
            get_parser=(lambda x, cmd=':port_in_b':
                        int(self.parse_cmd_response(
                            x,
                            self.build_get_string(cmd)))),
            max_val_age=0,
            val_mapping={'UserDefined': 0,
                         'StartRecord/ErrorReset': 1,
                         'RecordSelectBit0': 2,
                         'RecordSelectBit1': 3,
                         'RecordSelectBit2': 4,
                         'RecordSelectBit3': 5,
                         'RecordSelectBit4': 6,
                         'ExternalLimitSwitch': 7,
                         'Trigger': 8,
                         'Direction': 9,
                         'Enable': 10,
                         'Clock': 11,
                         'ClockDirectionMode1': 12,
                         'ClockDirectionMode2': 13},
            docstring='Function of digital input 2')

        self.add_parameter(
            'digital_input_3_function',
            label='Digital Input 3 Function',
            unit='',
            get_cmd=(lambda cmd=':port_in_c': self.build_get_string(cmd)),
            set_cmd=(lambda x, cmd=':port_in_c':
                     self.build_set_string(cmd, x)),
            get_parser=(lambda x, cmd=':port_in_c':
                        int(self.parse_cmd_response(
                            x,
                            self.build_get_string(cmd)))),
            max_val_age=0,
            val_mapping={'UserDefined': 0,
                         'StartRecord/ErrorReset': 1,
                         'RecordSelectBit0': 2,
                         'RecordSelectBit1': 3,
                         'RecordSelectBit2': 4,
                         'RecordSelectBit3': 5,
                         'RecordSelectBit4': 6,
                         'ExternalLimitSwitch': 7,
                         'Trigger': 8,
                         'Direction': 9,
                         'Enable': 10,
                         'Clock': 11,
                         'ClockDirectionMode1': 12,
                         'ClockDirectionMode2': 13},
            docstring='Function of digital input 3')

        self.add_parameter(
            'digital_input_4_function',
            label='Digital Input 4 Function',
            unit='',
            get_cmd=(lambda cmd=':port_in_d': self.build_get_string(cmd)),
            set_cmd=(lambda x, cmd=':port_in_d':
                     self.build_set_string(cmd, x)),
            get_parser=(lambda x, cmd=':port_in_d':
                        int(self.parse_cmd_response(
                            x,
                            self.build_get_string(cmd)))),
            max_val_age=0,
            val_mapping={'UserDefined': 0,
                         'StartRecord/ErrorReset': 1,
                         'RecordSelectBit0': 2,
                         'RecordSelectBit1': 3,
                         'RecordSelectBit2': 4,
                         'RecordSelectBit3': 5,
                         'RecordSelectBit4': 6,
                         'ExternalLimitSwitch': 7,
                         'Trigger': 8,
                         'Direction': 9,
                         'Enable': 10,
                         'Clock': 11,
                         'ClockDirectionMode1': 12,
                         'ClockDirectionMode2': 13},
            docstring='Function of digital input 4')

        self.add_parameter(
            'digital_input_5_function',
            label='Digital Input 5 Function',
            unit='',
            get_cmd=(lambda cmd=':port_in_e': self.build_get_string(cmd)),
            set_cmd=(lambda x, cmd=':port_in_e':
                     self.build_set_string(cmd, x)),
            get_parser=(lambda x, cmd=':port_in_e':
                        int(self.parse_cmd_response(
                            x,
                            self.build_get_string(cmd)))),
            max_val_age=0,
            val_mapping={'UserDefined': 0,
                         'StartRecord/ErrorReset': 1,
                         'RecordSelectBit0': 2,
                         'RecordSelectBit1': 3,
                         'RecordSelectBit2': 4,
                         'RecordSelectBit3': 5,
                         'RecordSelectBit4': 6,
                         'ExternalLimitSwitch': 7,
                         'Trigger': 8,
                         'Direction': 9,
                         'Enable': 10,
                         'Clock': 11,
                         'ClockDirectionMode1': 12,
                         'ClockDirectionMode2': 13},
            docstring='Function of digital input 5')

        self.add_parameter(
            'digital_input_6_function',
            label='Digital Input 6 Function',
            unit='',
            get_cmd=(lambda cmd=':port_in_f': self.build_get_string(cmd)),
            set_cmd=(lambda x, cmd=':port_in_f':
                     self.build_set_string(cmd, x)),
            get_parser=(lambda x, cmd=':port_in_f':
                        int(self.parse_cmd_response(
                            x,
                            self.build_get_string(cmd)))),
            max_val_age=0,
            val_mapping={'UserDefined': 0,
                         'StartRecord/ErrorReset': 1,
                         'RecordSelectBit0': 2,
                         'RecordSelectBit1': 3,
                         'RecordSelectBit2': 4,
                         'RecordSelectBit3': 5,
                         'RecordSelectBit4': 6,
                         'ExternalLimitSwitch': 7,
                         'Trigger': 8,
                         'Direction': 9,
                         'Enable': 10,
                         'Clock': 11,
                         'ClockDirectionMode1': 12,
                         'ClockDirectionMode2': 13},
            docstring='Function of digital input 6')

        self.add_parameter(
            'digital_input_7_function',
            label='Digital Input 7 Function',
            unit='',
            get_cmd=(lambda cmd=':port_in_g': self.build_get_string(cmd)),
            set_cmd=(lambda x, cmd=':port_in_g':
                     self.build_set_string(cmd, x)),
            get_parser=(lambda x, cmd=':port_in_g':
                        int(self.parse_cmd_response(
                            x,
                            self.build_get_string(cmd)))),
            max_val_age=0,
            val_mapping={'UserDefined': 0,
                         'StartRecord/ErrorReset': 1,
                         'RecordSelectBit0': 2,
                         'RecordSelectBit1': 3,
                         'RecordSelectBit2': 4,
                         'RecordSelectBit3': 5,
                         'RecordSelectBit4': 6,
                         'ExternalLimitSwitch': 7,
                         'Trigger': 8,
                         'Direction': 9,
                         'Enable': 10,
                         'Clock': 11,
                         'ClockDirectionMode1': 12,
                         'ClockDirectionMode2': 13},
            docstring='Function of digital input 7')

        self.add_parameter(
            'digital_input_8_function',
            label='Digital Input 8 Function',
            unit='',
            get_cmd=(lambda cmd=':port_in_h': self.build_get_string(cmd)),
            set_cmd=(lambda x, cmd=':port_in_h':
                     self.build_set_string(cmd, x)),
            get_parser=(lambda x, cmd=':port_in_h':
                        int(self.parse_cmd_response(
                            x,
                            self.build_get_string(cmd)))),
            max_val_age=0,
            val_mapping={'UserDefined': 0,
                         'StartRecord/ErrorReset': 1,
                         'RecordSelectBit0': 2,
                         'RecordSelectBit1': 3,
                         'RecordSelectBit2': 4,
                         'RecordSelectBit3': 5,
                         'RecordSelectBit4': 6,
                         'ExternalLimitSwitch': 7,
                         'Trigger': 8,
                         'Direction': 9,
                         'Enable': 10,
                         'Clock': 11,
                         'ClockDirectionMode1': 12,
                         'ClockDirectionMode2': 13},
            docstring='Function of digital input 8')

        self.add_parameter(
            'digital_output_1_function',
            label='Digital Output 1 Function',
            unit='',
            get_cmd=(lambda cmd=':port_out_a': self.build_get_string(cmd)),
            set_cmd=(lambda x, cmd=':port_out_a':
                     self.build_set_string(cmd, x)),
            get_parser=(lambda x, cmd=':port_out_a':
                        int(self.parse_cmd_response(
                            x,
                            self.build_get_string(cmd)))),
            max_val_age=0,
            val_mapping={'UserDefined': 0,
                         'Ready': 1,
                         'Running': 2},
            docstring='Function of digital output 1')

        self.add_parameter(
            'digital_output_2_function',
            label='Digital Output 2 Function',
            unit='',
            get_cmd=(lambda cmd=':port_out_b': self.build_get_string(cmd)),
            set_cmd=(lambda x, cmd=':port_out_b':
                     self.build_set_string(cmd, x)),
            get_parser=(lambda x, cmd=':port_out_a':
                        int(self.parse_cmd_response(
                            x,
                            self.build_get_string(cmd)))),
            max_val_age=0,
            val_mapping={'UserDefined': 0,
                         'Ready': 1,
                         'Running': 2},
            docstring='Function of digital output 2')

        self.add_parameter(
            'digital_output_3_function',
            label='Digital Output 3 Function',
            unit='',
            get_cmd=(lambda cmd=':port_out_c': self.build_get_string(cmd)),
            set_cmd=(lambda x, cmd=':port_out_c':
                     self.build_set_string(cmd, x)),
            get_parser=(lambda x, cmd=':port_out_c':
                        int(self.parse_cmd_response(
                            x,
                            self.build_get_string(cmd)))),
            max_val_age=0,
            val_mapping={'UserDefined': 0,
                         'Ready': 1,
                         'Running': 2},
            docstring='Function of digital output 3')

        self.add_parameter(
            'digital_output_4_function',
            label='Digital Output 4 Function',
            unit='',
            get_cmd=(lambda cmd=':port_out_d': self.build_get_string(cmd)),
            set_cmd=(lambda x, cmd=':port_out_d':
                     self.build_set_string(cmd, x)),
            get_parser=(lambda x, cmd=':port_out_d':
                        int(self.parse_cmd_response(
                            x,
                            self.build_get_string(cmd)))),
            max_val_age=0,
            val_mapping={'UserDefined': 0,
                         'Ready': 1,
                         'Running': 2},
            docstring='Function of digital output 4')

        self.add_parameter(
            'digital_output_5_function',
            label='Digital Output 5 Function',
            unit='',
            get_cmd=(lambda cmd=':port_out_e': self.build_get_string(cmd)),
            set_cmd=(lambda x, cmd=':port_out_e':
                     self.build_set_string(cmd, x)),
            get_parser=(lambda x, cmd=':port_out_e':
                        int(self.parse_cmd_response(
                            x,
                            self.build_get_string(cmd)))),
            max_val_age=0,
            val_mapping={'UserDefined': 0,
                         'Ready': 1,
                         'Running': 2},
            docstring='Function of digital output 5')

        self.add_parameter(
            'digital_output_6_function',
            label='Digital Output 6 Function',
            unit='',
            get_cmd=(lambda cmd=':port_out_f': self.build_get_string(cmd)),
            set_cmd=(lambda x, cmd=':port_out_f':
                     self.build_set_string(cmd, x)),
            get_parser=(lambda x, cmd=':port_out_f':
                        int(self.parse_cmd_response(
                            x,
                            self.build_get_string(cmd)))),
            max_val_age=0,
            val_mapping={'UserDefined': 0,
                         'Ready': 1,
                         'Running': 2},
            docstring='Function of digital output 6')

        self.add_parameter(
            'digital_output_7_function',
            label='Digital Output 7 Function',
            unit='',
            get_cmd=(lambda cmd=':port_out_g': self.build_get_string(cmd)),
            set_cmd=(lambda x, cmd=':port_out_g':
                     self.build_set_string(cmd, x)),
            get_parser=(lambda x, cmd=':port_out_g':
                        int(self.parse_cmd_response(
                            x,
                            self.build_get_string(cmd)))),
            max_val_age=0,
            val_mapping={'UserDefined': 0,
                         'Ready': 1,
                         'Running': 2},
            docstring='Function of digital output 7')

        self.add_parameter(
            'digital_output_8_function',
            label='Digital Output 8 Function',
            unit='',
            get_cmd=(lambda cmd=':port_out_h': self.build_get_string(cmd)),
            set_cmd=(lambda x, cmd=':port_out_h':
                     self.build_set_string(cmd, x)),
            get_parser=(lambda x, cmd=':port_out_h':
                        int(self.parse_cmd_response(
                            x,
                            self.build_get_string(cmd)))),
            max_val_age=0,
            val_mapping={'UserDefined': 0,
                         'Ready': 1,
                         'Running': 2},
            docstring='Function of digital output 8')

        self.add_parameter(
            'direction',
            label='Direction',
            unit='',
            get_cmd=(lambda cmd='d': self.build_get_string(cmd)),
            set_cmd=(lambda x, cmd='d': self.build_set_string(cmd, x)),
            get_parser=(lambda x, cmd='d': int(self.parse_cmd_response(
                x,
                self.build_get_string(cmd)))),
            max_val_age=0,
            val_mapping={'Left': 0,
                         'Right': 1},
            docstring='Direction of rotation (left or right)')

        self.add_parameter(
            'direction_change_on_repeat',
            label='Direction Change on Repeat',
            unit='',
            get_cmd=(lambda cmd='t': self.build_get_string(cmd)),
            set_cmd=(lambda x, cmd='t': self.build_set_string(cmd, x)),
            get_parser=(lambda x, cmd='t': int(self.parse_cmd_response(
                x,
                self.build_get_string(cmd)))),
            val_mapping={False: 0,
                         True: 1},
            docstring='Change direction of rotation on repeat')

        self.add_parameter(
            'error_correction',
            label='Error Correction',
            unit='',
            get_cmd=(lambda cmd='U': self.build_get_string(cmd)),
            set_cmd=(lambda x, cmd='U': self.build_set_string(cmd, x)),
            get_parser=(lambda x, cmd='U': int(self.parse_cmd_response(
                x,
                self.build_get_string(cmd)))),
            val_mapping={'Off': 0,
                         'CorrectionAfterTravel': 1,
                         'CorrectionDuringTravel': 2},
            docstring=('Error correction mode'
                       'Can be Off, CorrectionAfterTravel, or'
                       'CorrectionDuringTravel. CorrectionDuringTravel'
                       'is included only for compatibility reasons and'
                       'is implemented by CorrectionAfterTravel.'))

        self.add_parameter(
            'firmware_version',
            label='Firmware Version',
            unit='',
            get_cmd=(lambda cmd='v': self.build_get_string(cmd)),
            set_cmd=False,
            get_parser=(lambda x, cmd='v': self.parse_cmd_response(
                x,
                self.build_get_string(cmd))),
            docstring='Firmware version')

        self.add_parameter(
            'input_debounce_time',
            label='Input Debounce Time',
            unit='ms',
            get_cmd=(lambda cmd='K': self.build_get_string(cmd)),
            set_cmd=(lambda x, cmd='K': self.build_set_string(cmd, x)),
            get_parser=(lambda x, cmd='K': int(self.parse_cmd_response(
                x,
                self.build_get_string(cmd)))),
            vals=Ints(min_value=0,
                      max_value=250),
            docstring=('Debounce time for inputs in ms'
                       'Min value is 0'
                       'Max value is 250'))

        self.add_parameter(
            'io_input_mask',
            label='IO Input Mask',
            unit='',
            get_cmd=(lambda cmd='L': self.build_get_string(cmd)),
            set_cmd=(lambda x, cmd='L': self.build_set_string(cmd, x)),
            get_parser=(lambda x, cmd='L': int(self.parse_cmd_response(
                x,
                self.build_get_string(cmd)))),
            vals=Ints(min_value=0,
                      max_value=196671),
            docstring=('Set the IO input mask (32 bits)'
                       'If a bit of the mask is 1, the input or output is used'
                       'by the controller. If the bit is 0, the input or output'
                       'is available to the user.'
                       'Bit 0: Input 1'
                       'Bit 1: Input 2'
                       'Bit 2: Input 3'
                       'Bit 3: Input 4'
                       'Bit 4: Input 5'
                       'Bit 5: Input 6'
                       'Bit 16: Output 1'
                       'Bit 17: Output 2'
                       'Invalid masks are discarded even if echoed by'
                       'the controller.'))

        self.add_parameter(
            'io_output_mask',
            label='IO Output Mask',
            unit='',
            get_cmd=(lambda cmd='Y': self.build_get_string(cmd)),
            set_cmd=(lambda x, cmd='Y': self.build_set_string(cmd, x)),
            get_parser=(lambda x, cmd='Y': int(self.parse_cmd_response(
                x,
                self.build_get_string(cmd)))),
            vals=Ints(min_value=0,
                      max_value=196671),
            docstring=('Set the IO output mask (32 bits)'
                       'If a bit of the mask is 1, the input or output is used'
                       'by the controller (as long as it free in the IO input'
                       'mask. If the bit is 0, the input or output'
                       'is available to the user.'
                       'Bit 0: Input 1'
                       'Bit 1: Input 2'
                       'Bit 2: Input 3'
                       'Bit 3: Input 4'
                       'Bit 4: Input 5'
                       'Bit 5: Input 6'
                       'Bit 16: Output 1'
                       'Bit 17: Output 2'
                       'Invalid masks are discarded even if echoed by'
                       'the controller.'))

        self.add_parameter(
            'io_polarity',
            label='IO Polarity',
            unit='',
            get_cmd=(lambda cmd='h': self.build_get_string(cmd)),
            set_cmd=(lambda x, cmd='h': self.build_set_string(cmd, x)),
            get_parser=(lambda x, cmd='h': int(self.parse_cmd_response(
                x,
                self.build_get_string(cmd)))),
            max_val_age=0,
            vals=Ints(min_value=0,
                      max_value=196671),
            docstring=('Set the IO polarity mask (32 bits)'
                       'If a bit of the mask is 1, the polarity is retained.'
                       'If the bit is 0, the polarity is reversed.'
                       'Bit 0: Input 1'
                       'Bit 1: Input 2'
                       'Bit 2: Input 3'
                       'Bit 3: Input 4'
                       'Bit 4: Input 5'
                       'Bit 5: Input 6'
                       'Bit 16: Output 1'
                       'Bit 17: Output 2'
                       'Invalid masks are discarded even if echoed by'
                       'the controller.'))

        self.add_parameter(
            'limit_switch_behavior',
            label='Limit Switch Behavior',
            unit='',
            get_cmd=(lambda cmd='l': self.build_get_string(cmd)),
            set_cmd=(lambda x, cmd='l': self.build_set_string(cmd, x)),
            get_parser=(lambda x, cmd='l': int(self.parse_cmd_response(
                x,
                self.build_get_string(cmd)))),
            max_val_age=0,
            vals=Ints(min_value=0,
                      max_value=4294967295),
            docstring=('Set the limit switch behavior'
                       '16-bit mask'))

        self.add_parameter(
            'maximum_frequency',
            label='Maximum Frequency',
            unit='Steps/s',
            get_cmd=(lambda cmd='o': self.build_get_string(cmd)),
            set_cmd=(lambda x, cmd='o': self.build_set_string(cmd, x)),
            get_parser=(lambda x, cmd='o': int(self.parse_cmd_response(
                x,
                self.build_get_string(cmd)))),
            vals=Ints(min_value=1,
                      max_value=1000000),
            docstring=('Maximum frequency in steps per second'
                       'Min value: 1'
                       'Max value: 1000000'
                       'Maximum value depends on stepping mode'))

        self.add_parameter(
            'maximum_frequency2',
            label='Maximum Frequency 2',
            unit='Steps/s',
            get_cmd=(lambda cmd='n': self.build_get_string(cmd)),
            set_cmd=(lambda x, cmd='n': self.build_set_string(cmd, x)),
            get_parser=(lambda x, cmd='n': int(self.parse_cmd_response(
                x,
                self.build_get_string(cmd)))),
            vals=Ints(min_value=1,
                      max_value=1000000),
            docstring=('Maximum frequency 2 in steps per second'
                       'Min value: 1'
                       'Max value: 1000000'
                       'Maximum value depends on stepping mode.'
                       'This speed is only used in flag positioning mode'))

        self.add_parameter(
            'minimum_frequency',
            label='Minimum Frequency',
            unit='Steps/s',
            get_cmd=(lambda cmd='u': self.build_get_string(cmd)),
            set_cmd=(lambda x, cmd='u': self.build_set_string(cmd, x)),
            get_parser=(lambda x, cmd='u': int(self.parse_cmd_response(
                x,
                self.build_get_string(cmd)))),
            vals=Ints(min_value=1,
                      max_value=160000),
            docstring=('Minimum frequency in steps per second'
                       'Min value: 1'
                       'Max value: 160000'
                       'Motor starts moving at this speed at the beginning'
                       'of a record and then accelerates at the set rate'
                       'to the maximum frequency'))

        self.add_parameter(
            'motor_referenced',
            label='Motor Referenced',
            unit='',
            get_cmd=(lambda cmd=':is_referenced': self.build_get_string(cmd)),
            set_cmd=False,
            get_parser=(lambda x, cmd=':is_referenced':
                        bool(self.parse_cmd_response(
                            x,
                            self.build_get_string(cmd)))),
            max_val_age=0,
            docstring=('Set the limit switch behavior'
                       '16-bit mask'))

        self.add_parameter(
            'pause',
            label='Pause',
            unit='ms',
            get_cmd=(lambda cmd='P': self.build_get_string(cmd)),
            set_cmd=(lambda x, cmd='P': self.build_set_string(cmd, x)),
            get_parser=(lambda x, cmd='P': int(self.parse_cmd_response(
                x,
                self.build_get_string(cmd)))),
            vals=Ints(min_value=0,
                      max_value=65535),
            docstring=('Pause time between current record and continuation'
                       'record in ms.'
                       'Min value is 0'
                       'Max value is 65535'
                       'Has no effect if the current record specifies no'
                       'continuation record'))

        self.add_parameter(
            'phase_current',
            label='Phase Current',
            unit='%',
            get_cmd=(lambda cmd='i': self.build_get_string(cmd)),
            set_cmd=(lambda x, cmd='i': self.build_set_string(cmd, x)),
            get_parser=(lambda x, cmd='i': int(self.parse_cmd_response(
                x,
                self.build_get_string(cmd)))),
            max_val_age=0,
            vals=Ints(min_value=0,
                      max_value=150),
            docstring=('Set the phase current (in %)'
                       'Min value: 0'
                       'Max value: 150'
                       'Should not be set above 100'))

        self.add_parameter(
            'phase_current_standstill',
            label='Phase Current at Standstill',
            unit='%',
            get_cmd=(lambda cmd='r': self.build_get_string(cmd)),
            set_cmd=(lambda x, cmd='r': self.build_set_string(cmd, x)),
            get_parser=(lambda x, cmd='r': int(self.parse_cmd_response(
                x,
                self.build_get_string(cmd)))),
            max_val_age=0,
            vals=Ints(min_value=0,
                      max_value=150),
            docstring=('Set the phase current at standstill (in %)'
                       'Min value: 0'
                       'Max value: 150'
                       'Should not be set above 100'))

        self.add_parameter(
            'position',
            label='Position',
            unit='Steps',
            get_cmd=(lambda cmd='C': self.build_get_string(cmd)),
            set_cmd=False,
            get_parser=(lambda x, cmd='C': int(self.parse_cmd_response(
                x,
                self.build_get_string(cmd)))),
            max_val_age=0,
            docstring=('Current motor position relative to last reference'
                       'run'))

        self.add_parameter(
            'positioning_mode',
            label='Positioning Mode',
            unit='',
            get_cmd=(lambda cmd='p': self.build_get_string(cmd)),
            set_cmd=(lambda x, cmd='p': self.build_set_string(cmd, x)),
            get_parser=(lambda x, cmd='p': int(self.parse_cmd_response(
                x,
                self.build_get_string(cmd)))),
            val_mapping={'Relative': 1,
                         'Absolute': 2,
                         'InternalReferenceRun': 3,
                         'ExternalReferenceRun': 4,
                         'Speed': 5,
                         'Flag': 6,
                         'ClockManualLeft': 7,
                         'ClockManualRight': 8,
                         'ClockIntRefRun': 9,
                         'ClockExtRefRun': 10,
                         'AnalogSpeed': 11,
                         'Joystick': 12,
                         'AnalogPosition': 13,
                         'HWReference': 14,
                         'Torque': 15,
                         'CLQuickTest': 16,
                         'ClTest': 17,
                         'CLAutotune': 18,
                         'CLQuickTest2': 19},
            docstring='Positioning Mode')

        self.add_parameter(
            'quickstop',
            label='Quick Stop',
            unit='',
            get_cmd=(lambda cmd='H': self.build_get_string(cmd)),
            set_cmd=(lambda x, cmd='H': self.build_set_string(cmd, x)),
            get_parser=(lambda x, cmd='H': int(self.parse_cmd_response(
                x,
                self.build_get_string(cmd)))),
            vals=Ints(min_value=0,
                      max_value=8000),
            docstring=('Quickstop rate'
                       'Min value: 0'
                       'Max value: 8000'
                       'A value of 0 corresponds to an abrupt stop.'))

        self.add_parameter(
            'repetitions',
            label='Repetitions',
            unit='',
            get_cmd=(lambda cmd='W': self.build_get_string(cmd)),
            set_cmd=(lambda x, cmd='W': self.build_set_string(cmd, x)),
            get_parser=(lambda x, cmd='W': int(self.parse_cmd_response(
                x,
                self.build_get_string(cmd)))),
            vals=Ints(min_value=0,
                      max_value=254),
            docstring=('Repetitions of the current record'
                       'A value of zero means infinite repetitions'))

        self.add_parameter(
            'reset_position_error',
            label='Reset Position Error',
            unit='Steps',
            get_cmd=False,
            set_cmd=(lambda x, cmd='D': self.build_set_string(cmd, x)),
            vals=Ints(min_value=-100000000,
                      max_value=+100000000),
            docstring=('Reset position error.'
                       'Used to clear speed errors and set position to'
                       'the given value'))

        self.add_parameter(
            'reverse_clearance',
            label='Reverse Clearance',
            unit='Steps',
            get_cmd=(lambda cmd='z': self.build_get_string(cmd)),
            set_cmd=(lambda x, cmd='z': self.build_set_string(cmd, x)),
            get_parser=(lambda x, cmd='z': int(self.parse_cmd_response(
                x,
                self.build_get_string(cmd)))),
            vals=Ints(min_value=0,
                      max_value=9999),
            max_val_age=0,
            docstring=('Set the reverse clearance (in steps)'
                       'This is the number of steps that is added to'
                       'a movement command when changing directions to'
                       'compensate for play in the system'
                       'Min value: 0'
                       'Max value: 9999'))

        self.add_parameter(
            'status',
            label='Status',
            unit='',
            get_cmd=(lambda cmd='$': self.build_get_string(cmd)),
            set_cmd=False,
            get_parser=(lambda x, cmd='$': int(self.parse_cmd_response(
                x,
                self.build_get_string(cmd)))),
            max_val_age=0,
            docstring=('Controller status, 8-bit mask'
                       'Bit 0 == 1 means the controller is ready'
                       'Bit 1 == 1 means that zero position has been reached'
                       'Bit 2 == 1 means that there has been a position error'
                       'Bit 3 == 1 means that input 1 was set while the'
                       'controller was ready'
                       'Bit 4 is always set to 1'
                       'Bit 5 is always set to 0'
                       'Bit 6 is always set to 1'
                       'Bit 7 is always set to 0'))

        self.add_parameter(
            'step_mode',
            label='Step Mode',
            unit='',
            get_cmd=(lambda cmd='g': self.build_get_string(cmd)),
            set_cmd=(lambda x, cmd='g': self.build_set_string(cmd, x)),
            get_parser=(lambda x, cmd='g': int(self.parse_cmd_response(
                x,
                self.build_get_string(cmd)))),
            vals=Enum(1, 2, 4, 5, 8, 10, 16, 32, 64, 254, 255),
            docstring=('Step mode'
                       'Must be one of '
                       '[1, 2, 4, 5, 8, 10, 16, 32, 64, 254, 255]'
                       'Values from 1 to 64 set the number of microsteps'
                       'per step. Value 254 selects the feed rate mode'
                       'and value 255 selects the adaptive step mode.'))

        self.add_parameter(
            'travel_distance',
            label='Travel Distance',
            unit='Steps',
            get_cmd=(lambda cmd='s': self.build_get_string(cmd)),
            set_cmd=(lambda x, cmd='s': self.build_set_string(cmd, x)),
            get_parser=(lambda x, cmd='s': int(self.parse_cmd_response(
                x,
                self.build_get_string(cmd)))),
            vals=Ints(min_value=-100000000,
                      max_value=+100000000),
            docstring=('Travel distance in steps'
                       'Min value: -100000000'
                       'Max value: +100000000'
                       'Can only be positive when in relative positioning mode'
                       'In absolute positioning mode, this sets the target'
                       'position'))

        self.connect_message()

    def build_get_string(self, parameter: str) -> str:
        if parameter[0] == ':' and parameter[1] not in ['b', 'B']:
            return self._start_character + self.controller_id + str(parameter)
        else:
            return self._start_character + self.controller_id + 'Z' + str(parameter)

    def build_set_string(self, parameter: str, value: str = '') -> str:
        if parameter[0] == ':' and parameter[1] not in ['b', 'B']:
            return (self._start_character + self.controller_id + str(parameter)
                    + '=' + str(value))
        else:
            return(self._start_character + self.controller_id + str(parameter)
                   + str(value))

    def get_idn(self):
        info = self.ask_raw(self._start_character + self.controller_id + 'Zv')
        info = info.split(' ')[1].rstrip('\r').split('_')
        return {'vendor': 'Nanotec',
                'model': info[0],
                'serial': '',
                'firmware': info[2]}

    def load_record_from_eeprom(self, index: int) -> None:
        """
        Load a record from the EEPROM
        Args
            index: integer between 1 and 32
        :return:
        """
        self.write(self.build_set_string('y', str(index)))

    def parse_cmd_response(self, response: str, command: str) -> str:
        # long commands
        if command[0] == ':':
            if response == self.controller_id + ':?':
                raise ValueError(f'Unknown command {command}')
            return response[len(self.controller_id) + str(command):]
        # normal commands
        else:
            if response == (self._start_character + self.controller_id
                            + str(command) + '?'):
                raise ValueError(f'Unknown command {command}')
            return response[len(self._start_character + self.controller_id
                            + str(command) + str(self.controller_id)):]

    def save_record_to_eeprom(self, index: int) -> None:
        """
        Save the current record to the EEPROM
        Args
            index: integer between 1 and 32
        :return:
        """
        self.write(self.build_set_string('>', str(index)))

    def start_motor(self):
        self.write(self.build_set_string('A'))

    def stop_motor(self, ramp='Quickstop'):
        types = {
            'Quickstop': 0,
            'Brake': 1,
        }
        # TODO: add error checking to ramp type
        self.write(self.build_set_string('S', str(types[ramp])))
