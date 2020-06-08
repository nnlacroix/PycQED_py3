import time
import logging

import NanotecSMI33

class QudevDisplacer(NanotecSMI33):
    def __init__(self, name: str, address: str, id: str = '*', **kwargs) -> None:
        super().__init__(name, address, id, **kwargs)
        self.initialize()

    def initialize(self, reverse_clearance: int = 0):
        self.command_response('Enabled')
        self.firmware_version()
        self.phase_current(20)
        self.phase_current_standstill(0)
        self.limit_switch_behavior(int(0b0010010000100010))
        self.io_polarity(int(0b110000000000011111))
        self.digital_input_6_function('ExternalLimitSwitch')
        self.error_correction('Off')
        self.reverse_clearance(reverse_clearance)
        self.acceleration_mode('Jerk-free')
        self.acceleration_jerk(1)
        self.braking_jerk(100000000)
        self.step_mode(2)

        previous_direction = self.direction()
        if self.limit_switch_on():
            for i in range(1,3):
                if previous_direction == 'Left':
                    direction = 'Right'
                else:
                    direction = 'Left'
                self.escape_bound(direction, steps=30*i)
                previous_direction = self.direction()

    def escape_limit(self, direction, steps):
        # Configure motor
        self.reset_position_error(0)
        self.command_response('Disabled')
        self.positioning_mode('Relative')
        self.travel_distance(steps)
        self.minimum_frequency(30)
        self.maximum_frequency(100)
        self.maximum_frequency2(100)
        self.acceleration(65535)
        self.braking(65535)
        self.quickstop(0)
        self.direction(direction)
        self.direction_change_on_repeat(False)
        self.repetitions(1)
        self.pause(0)
        self.continuation_record(0)
        self.command_response('Enabled')
        # Start motor
        self.start_motor()
        time.sleep(3)

    def find_limits(self):
        pass

    def limit_switch_on(self) -> Bool:
        return self.io_output_mask() & 32 > 0

    def safety_check(self):
        self.command_response('Enabled')
        error_correction = self.error_correction()
        input_6_function = self.digital_input_6_function()
        input_config = self.io_input_mask()
        io_polarity = self.io_polarity()
        limit_switch_config = self.limit_switch_behavior()
        microsteps = self.step_mode()
        phase_current = self.phase_current()
        standstill_current = self.phase_current_standstill()
        assert error_correction == 'Off', f'Error correction not disabled'
        assert input_6_function == 'ExternalLimitSwitch',\
            'Input 6 not configured for external limit switch'
        assert input_config & 32 == 0,\
            'Bit 5 of IO output mask not set to 0'
        assert io_polarity & 32 == 0,\
            'Bit 5 of IO polarity not set to 0'
        assert microsteps in [1, 2, 4, 8], 'Unusual number of microsteps'
        assert limit_switch_config == 0b10010000100010, \
            f'Limit switch not correctly configured (should be {9250:b})'
        assert 1 <= phase_current <= 20,\
            f'Phase current not between 0 and 20% ({phase_current}%)'
        assert 0 <= standstill_current <= 1,\
            f'Standstill current not between 0 and 1% ({standstill_current}%)'





