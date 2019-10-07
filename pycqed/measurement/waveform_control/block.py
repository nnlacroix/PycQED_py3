import logging
from copy import deepcopy

log = logging.getLogger(__name__)

class Block:
    #counter = 0 for now cannot really reuse a block
    INSIDE_BLOCKINFO_NAME = "BlockInfo"

    def __init__(self, block_name, pulse_list:list):
        self.name = block_name
        self.pulses = deepcopy(pulse_list)

    def build(self, ref_point="end", ref_point_new="start",
              ref_pulse='previous_pulse', block_delay=0, block_start_position=0,
              block_end_position=None):
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
        :param block_start_position: position of the block start. Defaults to
            before first pulse in list. Position should be changed in case the
            first pulse played is not the first one in the list.
        :param block_end_position: position of the block end. Defaults to after
            last pulse.
        :return:
        """
        if ref_point_new != "start":
            raise NotImplementedError("For now can only refer blocks to 'start'")

        block_instance_name = f"{self.name}"
        block_start = {"name": f"start",
                       "pulse_type": "VirtualPulse",
                       "pulse_delay": block_delay,
                       "ref_point": ref_point,
                       "ref_pulse": ref_pulse}
        block_end = {"name": f"end",
                     "pulse_type": "VirtualPulse"}

        # insert starting block at defined position
        self.pulses.insert(block_start_position, block_start)

        # same for end
        if block_end_position is None:
            block_end_position = len(self.pulses)
        self.pulses.insert(block_end_position, block_end)

        pulses_built = []
        for p in self.pulses:
            # if a block is found inside a block, build the lower level
            # block assuming first dictionary in list are the arguments
            # to build the block.
            if isinstance(p, Block):
                if len(p.pulses) == 0:
                    raise ValueError(
                        f"Blocks inside blocks must at least contain a "
                        f"dictionary with parameters to build the block but "
                        f"block {p.name} inside block {self.name} is empty.")
                inside_block_params = deepcopy(p.pulses[0])
                if inside_block_params.get("pulse_type", None) != \
                    self.INSIDE_BLOCKINFO_NAME:
                    log.warning(f"First dict in {p.name} inside {self.name}"
                                f"is not a Block information dictionary because "
                                f"its pulse_type is not "
                                f"{self.INSIDE_BLOCKINFO_NAME}. {p.name} "
                                f"will be build with default parameters")
                    inside_block_pulses = p.build()
                else:
                    del p.pulses[0] #info to build block should not stay in block
                    inside_block_pulses = p.build(**inside_block_params)
                # add all pulses of the inside block to the outer block
                pulses_built.extend(inside_block_pulses)
                continue

            # else if the pulse has a name, prepend the blockname to it
            elif p.get("name", None) is not None:
                p['name'] = block_instance_name + "_" + p['name']
            pulses_built.append(p)

        return pulses_built

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