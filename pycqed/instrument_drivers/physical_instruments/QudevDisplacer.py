import time
import logging

from pycqed.instrument_drivers.physical_instruments.NanotecSMI33\
    import NanotecSMI33


class QudevDisplacer(NanotecSMI33):
    def __init__(self, name: str, address: str, controller_id: str = '*', **kwargs) -> None:
        super().__init__(name, address, controller_id, **kwargs)
        self._lower_bound = None
        self._upper_bound = None
        self._is_initialized = False
        self.initialize()

    def _drive_absolute(self, position: int, direction: str,
                        speed: int = 600) -> None:
        """
        Drive the motor in absolute mode
        :param position:
        :param direction:
        :param speed:
        :return:
        """
        self.command_response('Disabled')
        self.positioning_mode('Absolute')
        self.travel_distance(position)
        self.minimum_frequency(30)
        self.maximum_frequency(speed)
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
        self.start_motor()

    def drive_motor(self, position: int, speed: int = 600,
                    minimum_steps_to_limit: int = 50,
                    mode: str = 'Normal',
                    clearance_steps: int = 30) -> None:
        """
        Drive the motor to a given position (in steps)
        Works in either Normal mode where it drives directly to the target
        in terms of number of steps or in LeftToRight mode where it always
        approaches the target position from below
        :param position:
        :param speed:
        :param minimum_steps_to_limit:
        :param mode:
        :param clearance_steps:
        :return:
        """
        self.safety_check()
        if (position < self._lower_bound + minimum_steps_to_limit or
            position > self._upper_bound - minimum_steps_to_limit):
            pass
            # TODO: add warning
        position = max(position, self._lower_bound + minimum_steps_to_limit)
        position = min(position, self._upper_bound - minimum_steps_to_limit)
        if mode == 'Normal':
            if position < self.position():
                direction = 'Right'
            else:
                direction = 'Left'
            self._drive_abosolute(position, direction, speed)
            # wait until limit is reached or controller is finished
            self.wait_until_status(5)
        elif mode == 'LeftToRight':
            if (position - clearance_steps) < self.position():
                direction = 'Right'
            else:
                direction = 'Left'
            self._drive_absolute(position - clearance_steps, direction, speed)
            self.wait_until_status(5)
            if direction == 'Left':
                direction = 'Right'
            else:
                direction = 'Left'
            self._drive_absolute(position, direction, speed)
            self.wait_until_status(5)

    def _escape_limit(self, direction: str, steps: int) -> None:
        # Configure motor
        self.reset_position_error(0)
        # TODO: should have a with ... construct to always reenable
        #       command response
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
        # Wait until controller reaches limit
        # (LabVIEW code had a hard coded wait of 3s)
        self.wait_until_status(4)

    def _find_limits(self):
        """
        Drives the motor to the lower limit in referenced mode and then
        commands the motor to drive towards the upper limit until it
        reaches the limit switch
        :return:
        """
        # Prepare first record
        self.command_response('Disabled')
        self.positioning_mode('ExternalReferenceRun')
        self.travel_distance(100)
        self.minimum_frequency(30)
        self.maximum_frequency(250)
        self.maximum_frequency2(250)
        self.acceleration(6000)
        self.braking(1)
        self.quickstop(0)
        self.direction('Right')
        self.direction_change_on_repeat(False)
        self.repetitions(1)
        self.pause(200)
        self.continuation_record(2)
        self.save_record_to_eeprom(1)
        self.command_response('Enabled')
        # Prepare second record
        self.command_response('Disabled')
        self.positioning_mode('Relative')
        self.travel_distance(100000000)
        self.minimum_frequency(30)
        self.maximum_frequency(250)
        self.maximum_frequency2(250)
        self.acceleration(6000)
        self.braking(1)
        self.quickstop(0)
        self.direction('Left')
        self.direction_change_on_repeat(False)
        self.repetitions(254)
        self.pause(200)
        self.continuation_record(0)
        self.save_record_to_eeprom(2)
        self.command_response('Enabled')
        # Prepare third record
        self.command_response('Disabled')
        self.positioning_mode('Relative')
        self.travel_distance(300)
        self.minimum_frequency(30)
        self.maximum_frequency(250)
        self.maximum_frequency2(250)
        self.acceleration(6000)
        self.braking(65535)
        self.quickstop(0)
        self.distance('Right')
        self.direction_change_on_repeat(False)
        self.repetitions(1)
        self.pause(200)
        self.continuation_record(0)
        self.save_record_to_eeprom(3)
        self.command_response('Enabled')
        # Start sequence
        self.load_record_from_eeprom(1)
        self.start_motor()

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

        # Escape the limit if we are currently at one
        previous_direction = self.direction()
        if self.limit_switch_on():
            for i in range(1,3):
                if previous_direction == 'Left':
                    direction = 'Right'
                else:
                    direction = 'Left'
                self.escape_limit(direction, steps=30*i)
                previous_direction = self.direction()

        # Find the limits by driving to the upper limit in external reference
        # run mode and then driving to lower limit in relative mode until
        # the limit switch is reached and the motor stops
        self.find_limits()
        # Uses record 3 set in find_limits to travel away from the limit
        self.travel_away_from_limit()
        self.limit_switch_behavior(0b10010000100010)
        self._is_initialized = True

    def limit_switch_on(self) -> bool:
        """
        Check whether limit switch is activated
        :return:
        """
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

    def _travel_away_from_limit(self) -> None:
        """
        Travel away from the limit after finding the limit
        :return:
        """
        self._lower_bound = 0
        self._upper_bound = self.position()
        self.reset_position_error(0)
        # Set limit switch behavior to correct value
        self.limit_switch_behavior(0b100010000100010)
        # Reload record 3 from EEPROM (set in find_limits)
        self.load_record_from_eeprom(3)
        self.start()
        # wait until controller finishes moving or the limit is reached
        self.wait_until_status(5)

    def wait_until_status(self, mask: int = 5, timeout: int = 30) -> None:
        """
        Wait until controller has a status given by the mask
        :param mask:
        :param timeout:
        :return:
        """
        self.command_response('Enabled')
        t0 = time.time()
        while (time.time() - t0) < timeout:
            if self.status() & mask > 0:
                break
            else:
                time.sleep(0.1)






