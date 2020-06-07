import numpy as np
import serial


class NanotecSMI33():
    def __init__(self, port, id, baudrate=115200, timeout=0.5):
        self.baudrate = baudrate
        if id in range(1,254) or id == '*':
            self.id = id
        else:
            #TODO: warn about incorrect ID
            pass
        self.port = port
        self.timeout = timeout
        self._buffer_length = 128
        self._line_terminator = '\r'
        self._start_character = '#'


        #TODO: set error correction mode (U) = 0

    def _check_bit(self, byte, bit_number):
        return byte & 1 << bit_number != 0

    def query(self, command):
        #TODO: Check command and raise exception if incorrect
        with serial.Serial(self.port,
                           self.baudrate,
                           timeout=self.timeout) as ser:
            message = bytes(  self._start_character
                            + self.id
                            + command
                            + self.line_terminator)
            ser.write(message)
            response = ser.read_until(self._line_terminator,
                                      self._buffer_length)
            if response == message + b'?':
                #TODO: unknown command
                pass
            # Remove ID from response
            return response[1:]

    def read_parameter(self, parameter):
        #TODO: Check parameter and raise exception if incorrect
        command = 'Z' + parameter
        response = self.query(command)
        # Remove command from response
        value = response[len(command):]
        return value

    def read_acceleration_ramp(self):
        return np.uint16(self.read_parameter('b'))

    def read_bldc_current_time_constant(self, time_constant=0):
        """Read current time constant for BLDC motors
        time_constant is defined in ms
        """
        return int(self.read_parameter(':itime'))

    def read_bldc_peak_current(self, current=0):
        """Read peak current for BLDC motors
        current is specified in percent (0-150)
        """
        return int(self.read_parameter(':ipeak'))

    def read_brake_ramp(self):
        return np.uint16(self.read_parameter('B'))

    def read_current_record(self):
        return self.read_parameter('|')

    def read_errors(self):
        for index in range(32):
            parameter = str(index) + 'E'
            response = self.read_parameter(parameter)
            # {0x00: None,
            #  0x01: 'LOWVOLTAGE',
            #  0x02: 'TEMP',
            #  0x04: 'TMC',
            #  0x08: 'EE',
            #  0x10: 'QEI',
            #  0x20: 'INTERNAL',
            #  }

    def read_firmware_info(self):
        #TODO: parse fields
        return self.read_parameter('v')

    def read_limit_switch_behavior(self):
        #TODO: parse fields
        return self.read_parameter('l')

    def read_maximum_frequency(self):
        return np.uint32(self.read_parameter('o'))

    def read_motor_type(self, motor_type='stepper'):
        """Sets type of motor"""
        motors = {
            0: 'stepper',
            1: 'BLDCwHall',
            2: 'BLDCwHallwEnc',
        }
        return motors(int(self.read_parameter(':CL_motor_type')))

    def read_phase_current(self):
        """Reads the phase current
        current is specified in percent (0-150)"""
        return int(self.read_parameter('i'))

    def read_phase_current_standstill(self):
        """Reads the phase current at standstill
        current is specified in percent (0-150)"""
        return int(self.read_parameter('r'))

    def read_position(self):
        return int(self.read_parameter('C'))

    def read_position_mode(self):
        return self.read_parameter('p')

    def read_quickstop_ramp(self):
        return self.read_parameter('H')

    def read_record(self, record=None):
        """Reads record from EEPROM
        if record is None, reads the currently-loaded settings
        otherwise, if record in range(1,32), reads record from EEPROM"""
        if record is None:
            return self.read_parameter('|')
        else:
            return self.read_parameter(str(record) + '|')


    def read_repetition_number(self):
        return self.read_parameter('W')

    def read_rotation_direction(self):
        #TODO: return left or right rather than number
        directions = {
            0: 'left',
            1: 'right',
        }
        return directions[self.read_parameter('d')]


    def read_status(self):
        response = self.read_parameter('$')
        status = response[0]
        # Check controller status
        if _check_bit(status, 0):
            #TODO error has occurred
            pass
        # Check if zero position reached
        if _check_bit(status, 1):
            pass
        # Check if position error occurred
        if _check_bit(status, 2):
            pass

    def read_step_mode(self):
        return int(self.read_parameter('g'))

    def read_travel_distance(self):
        return int(self.read_parameter('s'))

    def reset_position_error(self, value=None):
        self.write_parameter('D', value)

    def set_acceleration_ramp(self, value):
        """Set acceleration ramp
        Acceleration in Hz/ms = (3000.0/sqrt(float)<parameter>))-11.7)"""
        self.write_parameter('b', np.uint16(value))

    def set_bldc_current_time_constant(self, time_constant=0):
        """Set current time constant for BLDC motors
        time_constant is defined in ms
        """
        self.write_parameter(':itime', str(np.uint16(time_constant)))

    def set_bldc_peak_current(self, current=0):
        """Set peak current for BLDC motors
        current is specified in percent (0-150)
        """
        self.write_parameter(':ipeak',str(np.uint8(current)))

    def set_brake_ramp(self, value=0):
        """Set brake ramp
        To convert to Hz/ms, see formula in set_acceleration_ramp
        A value of 0 (default) means that the braking rate equals the
        acceleration rate
        """
        self.write_parameter('B', np.uint16(value))

    def set_command_response(self, response='enabled'):
        """Enable/disable controller responses to commands"""
        responses = {
            'disabled': 0,
            'enabled': 1,
        }
        self.write('|', responses[response])

    def set_error_correction_mode(self, mode):
        values = {
            'Off': 0,
            'CorrectionAfterTravel': 1,
            'CorrectionDuringTravel': 2
        }
        if mode not in values.keys():
            #TODO print error message
            pass
        command = values['mode']
        self.write_parameter(command)

    # def set_keyword_parameter(self, keyword, value):
    #     command = keyword + '=' + str(value)
    #     response = self.query(command)
    #     if response != command:
    #         #TODO: error
    #         pass

    def set_maximum_frequency(self, frequency):
        """Set maximum frequency (steps per second)"""
        self.write_parameter('o', np.uint32(frequency))

    def set_motor_type(self, motor_type='stepper'):
        """Sets type of motor"""
        motors = {
            'stepper': 0,
            'BLDCwHall': 1,
            'BLDCwHallwEnc': 2,
        }
        self.write_parameter(':CL_motor_type', motors[motor_type])

    def set_phase_current(self, current):
        """Sets the phase current
        current is specified in percent (0-150)
        """
        self.write_parameter('i', str(np.uint8(current)))

    def set_phase_current_standstill(self, current):
        """Sets the phase current at standstill
        current is specified in percent (0-150)
        """
        self.write_parameter('r', str(np.uint8(current)))

    def set_position_mode(self, mode=1):
        """Set the positioning mode
        Defaults to 1 (relative positioning)
        """
        self.write_parameter('p', np.uint8(mode))

    def set_quickstop_ramp(self, value):
        """Set quickstop ramp
        0 means abrupt stop"""
        self.write_parameter('H', np.uint16(value))

    def set_repetition_number(self, repetitions=1):
        """Set number of repetitions of current record
        Typically 1 repetition
        0 equivalent to infinite repetitions
        """
        self.write_parameter('W', np.uint8(repetitions))

    def set_rotation_direction(self, direction):
        direction_values = {
            'Left': 0,
            'Right': 1,
        }
        #TODO: check that direction is in values
        self.write_parameter('d', direction_values[direction])

    def set_step_mode(self, value):
        """Set number of microsteps per step
        Must be in [1, 2, 4, 5, 8, 10, 16, 32, 64, 254, 255]
        254 and 255 have special meanings, see Nanotec documentation
        """
        if value not in [1, 2, 4, 5, 8, 10, 16, 32, 64, 254, 255]:
            #TODO: value error
            pass
        self.write_parameter('g', value)

    def set_travel_distance(self, value):
        """Set travel distance in (micro-)steps
        In relative mode, only positive values are allowed"""
        self.write_parameter('s', int(value))

    def start_motor(self):
        self.write_parameter('A')

    def stop_motor(self, ramp='Quickstop'):
        types = {
            'Quickstop': 0,
            'Brake': 1,
        }
        #TODO: add error checking to ramp type
        self.write_parameter('S', types[ramp])

    def write(self, command, value=None):
        """Write to controller without checking for a response"""
        #TODO: Check command and raise exception if incorrect
        if value is not None:
            command += value
        with serial.Serial(self.port,
                           self.baudrate,
                           timeout=self.timeout) as ser:
            message = bytes(  self._start_character
                            + self.id
                            + command
                            + self.line_terminator)
            ser.write(message)

    def write_parameter(self, parameter, value=None):
        #TODO: Check parameter and raise exception if incorrect
        command = parameter
        if value is not None:
            if parameter[0] == ':':
                command += '='
            command += str(value)
        response = self.query(command)
        if response != command:
            #TODO error occurred
            pass




"""
'#' + command + '\r'

'$' read status
'A' start motor
'b' set acceleration ramp
'B' set the brake ramp
'C' read out position
'd' set direction of rotation
'D' clear errors
'E' errors
'g' set step mode
'H' set quickstop ramp
'I' set the limit switch behavior
'm' set drive address
'n' set maximum frequency 2
'o' set maximum frequency
'p' set the positioning mode
's' set the travel distance
'S' stops the motor
't' set change of direction for repeat modes?
'U' set error correction mode
'u' set minimum frequency
'v' read firmware version
'Z + command' read from motor
"""