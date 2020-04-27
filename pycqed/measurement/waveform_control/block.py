import logging
from copy import deepcopy

log = logging.getLogger(__name__)

class Block:
    """
    A block is a building block for a Quantum Algorithm Experiment.
            :param block_start_position: position of the block start. Defaults to
            before first pulse in list. Position should be changed in case the
            first pulse played is not the first one in the list.
        :param block_end_position: position of the block end. Defaults to after
            last pulse. If given, block_end_position MUST take into account the
            block_start i.e. the block start is added to the pulses before inserting
            the block end at its defined location.
    """
    counter = 0
    INSIDE_BLOCKINFO_NAME = "BlockInfo"

    def __init__(self, block_name, pulse_list:list):
        self.name = block_name
        self.pulses = deepcopy(pulse_list)

    def build(self, ref_point="end", ref_point_new="start",
              ref_pulse='previous_pulse', block_delay=0, name=None,
               **kwargs):
        """
        Adds the block shell recursively through the pulse list.
        Returns the flattened pulse list
        :param ref_point: to which point of the previous pulse should this block
        be referenced to. Defaults to "end".
        :param ref_point_new: which point of this block is reference to the
            ref_point. Currently only supports "start".
        :param ref_pulse: to which pulse in the list is this block referenced to.
            Defaults to "previous_pulse" in list.
        :param block_delay: delay before the start of the block
        :return:
        """
        if ref_point_new != "start":
            raise NotImplementedError("For now can only refer blocks to 'start'")
        if name is None:
            name = self.name + (f"_{self.counter}" if self.counter > 0 else "")
            self.counter += 1

        block_start = {"name": f"start",
                       "pulse_type": "VirtualPulse",
                       "pulse_delay": block_delay,
                       "ref_pulse": ref_pulse,
                       "ref_point": ref_point}
        block_start.update(kwargs.get("block_start", {}))
        block_end = {"name": f"end",
                     "pulse_type": "VirtualPulse"}
        block_end.update(kwargs.get("block_end", {}))
        pulses_built = deepcopy(self.pulses)

        # check if block_start/end  specified by user
        block_start_specified = False
        block_end_specified = False
        for p in pulses_built:
            if p.get("name", None) == "start":
                block_start = p #save reference
                block_start_specified = True
            elif p.get("name", None) == "end":
                block_end = p
                block_end_specified = True
        # add them if not specified
        if not block_start_specified:
            pulses_built = [block_start] + pulses_built
        if not block_end_specified:
            pulses_built = pulses_built + [block_end]

        for p in pulses_built:
            # if a dictionary wrapping a block is found, compile the inner block.
            if p.get("pulse_type", None) == self.INSIDE_BLOCKINFO_NAME:
                # p needs to have a block key
                assert 'block' in p, f"Inside block {p.get('name', 'Block')} " \
                    f"requires a key 'block' which refers to the uncompiled " \
                    f"block object."
                inside_block = p.pop('block')
                inside_block_pulses = inside_block.build(**p)
                # add all pulses of the inside block to the outer block
                pulses_built.extend(inside_block_pulses)

        # prepend block name to reference pulses and pulses names
        for p in pulses_built:
            # if the pulse has a name, prepend the blockname to it
            if p.get("name", None) is not None:
                p['name'] = name + "-|-" + p['name']

            ref_pulse = p.get("ref_pulse", "previous_pulse")
            p_is_block_start = self._is_block_start(p, block_start)

            # rename ref pulse within the block if not a special name
            escape_names = ("previous_pulse", "segment_start")
            if ref_pulse not in escape_names and not p_is_block_start:
                p['ref_pulse'] = name + "-|-" + p['ref_pulse']

        return pulses_built

    def _is_shell(self, pulse, block_start, block_end):
        """
        Checks, based on the pulse name, whether a pulse belongs to the block shell.
        That is, if the pulse name is the same as the name of the block start or end.
        A simple equivalence p == block_start or p == p_end does not work as pulse
        could be a deepcopy of block_start, which would return False in the above
        expressions.
        Args:
            pulse (dict): pulse to check.
            block_start (dict): dictionary of the block start
            block_end (dict): dictionary of the block end
        Returns: whether pulse is a shell dictionary (bool)
        """
        return self._is_block_start(pulse, block_start) \
               or self._is_block_end(pulse, block_end)

    def _is_block_start(self, pulse, block_start):
        """
        Checks, based on the pulse name, whether a pulse belongs to the block shell.
        That is, if the pulse name is the same as the name of the block start or end.
        A simple equivalence p == block_start or p == p_end does not work as pulse
        could be a deepcopy of block_start, which would return False in the above
        expressions.
        Args:
            pulse (dict): pulse to check.
            block_start (dict): dictionary of the block start
        Returns: whether pulse is a the block start dictionary (bool)
        """
        return pulse.get('name', None) == block_start['name']


    def _is_block_end(self, pulse, block_end):
        """
        Checks, based on the pulse name, whether a pulse belongs to the block shell.
        That is, if the pulse name is the same as the name of the block start or end.
        A simple equivalence p == block_start or p == p_end does not work as pulse
        could be a deepcopy of block_start, which would return False in the above
        expressions.
        Args:
            pulse (dict): pulse to check.
            block_end (dict): dictionary of the block end
        Returns: whether pulse is a the block end dictionary (bool)
        """
        return pulse.get('name', None) == block_end['name']


    def extend(self, additional_pulses):
        self.pulses.extend(additional_pulses)

    def __add__(self, other_block):
        return Block(f"{self.name}_{other_block.name}",
                     self.pulses + other_block.pulses)

    def __len__(self):
        return len(self.pulses)

    def __repr__(self):
        string_repr = f"---- {self.name} ----\n"
        for i, p in enumerate(self.pulses):
            string_repr += f"{i}: " + repr(p) + "\n"
        return string_repr