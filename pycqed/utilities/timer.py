import numpy as np
import datetime as dt
import logging
from collections import OrderedDict

from pycqed.measurement.hdf5_data import write_dict_to_hdf5

log = logging.getLogger(__name__)

import functools


class Timer(OrderedDict):
    HDF_GRP_NAME = "Timers"
    NAME_CKPT_START = "start"
    NAME_CKPT_END = "end"

    def __init__(self, name="timer", fmt="%Y-%m-%d %H:%M:%S.%f", name_separator=".",
                 verbose=False, auto_start=True, timer=None):
        self.fmt = fmt
        self.name = name
        self.name_separator = name_separator
        self.verbose = verbose
        if timer is not None:
            super().__init__(timer)
        if auto_start and timer is not None:
            self.checkpoint(self.NAME_CKPT_START)

    @staticmethod
    def from_string(timer):
        pass

    def __call__(self, func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # print(args)
            if hasattr(args[0], "timer"):
                args[0].timer.checkpoint(func.__qualname__ + self.name_separator + self.NAME_CKPT_START)
            else:
                log.warning(f'Using @Timer decorator on {args[0]} but {args[0]} has no .timer attribute.'
                            'Time will not be logged.')
            output = func(*args, **kwargs)
            if hasattr(args[0], "timer"):
                args[0].timer.checkpoint(func.__qualname__ + self.name_separator + self.NAME_CKPT_END)
            return output

        return wrapper

    def __enter__(self):
        if self.get(self.NAME_CKPT_START, None) is None:
            # overwrite auto_start because when used in "with" statement, start must happen at beginning
            self.checkpoint(self.NAME_CKPT_START)
        if self.verbose:
            lvl = log.level
            log.setLevel(logging.INFO)
            log.info(f'Start of {self.name}: {self[self.NAME_CKPT_START].get_start()}')
            log.setLevel(lvl)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.checkpoint(self.NAME_CKPT_END)

        if self.verbose:
            lvl = log.level
            log.setLevel(logging.INFO)
            log.info(f'End of {self.name}: {self[self.NAME_CKPT_END].get_end()}. Duration: {self.duration(return_type="str")}')
            log.setLevel(lvl)

    def checkpoint(self, name, value=None, log_every_x=1):
        if name not in self:
            self[name] = Checkpoint(name, fmt=self.fmt, log_every_x=log_every_x)
        else:
            self[name].log_time()

    def duration(self, keys=None, return_type="seconds"):
        if keys is None:
            keys = (self.NAME_CKPT_START, self.NAME_CKPT_END)
        try:
            duration = self[keys[1]].get_end() - self[keys[0]].get_start()
            if return_type == "seconds":
                return duration.total_seconds()
            elif return_type == "time_delta":
                return duration
            elif return_type == "str":
                return str(duration)
            else:
                raise ValueError(f'return_type={return_type} not understood.')
        except KeyError as ke:
            log.error(f"Could not find key in timer: {ke}. Available keys: {self.keys()}")

    def save(self, data_object, group_name=None):
        '''
        Saves metadata on the MC (such as timings)
        '''
        if group_name is None:
            group_name = self.name
        set_grp = data_object.create_group(group_name)
        d = {k: repr(v) for k, v in self.items()}
        write_dict_to_hdf5(d, entry_point=set_grp,
                               overwrite=False)

class Checkpoint(list):
    def __init__(self, name, checkpoints=(), log_every_x=1, fmt="%Y-%m-%d %H:%M:%S.%f",
                 min_timedelta=0, verbose=False):
        super().__init__()
        self.extend(checkpoints)
        self.name = name
        self.fmt = fmt
        self.log_every_x = log_every_x
        self.counter = 0
        self.min_timedelta = min_timedelta
        self.log_time()
        self.verbose = verbose

    def get_start(self):
        return self[0]

    def get_end(self):
        return self[-1]

    def active(self):
        if len(self) > 0 and \
                (dt.datetime.now() - self[-1]).total_seconds() < self.min_timedelta:
            #             (dt.datetime.now() - dt.datetime.strptime(self[-1], self.fmt)).total_seconds() < self.min_timedelta:

            return False
        else:
            return True

    def log_time(self, value=None):
        if self.active():
            if self.counter % self.log_every_x == 0:
                if value is None:
                    value = dt.datetime.now()  # .strftime(self.fmt)
                self.counter += 1
                self.append(value)

    def duration(self, ref=None, return_type="seconds"):
        if ref is None:
            ref = self[0]
        duration = self[-1] - ref
        if return_type == "seconds":
            return duration.total_seconds()
        elif return_type == "time_delta":
            return duration
        elif return_type == "str":
            return str(duration)
        else:
            raise ValueError(f'return_type={return_type} not understood.')

    #     def __enter__(self):
    #         if self.verbose:
    #             lvl = log.level
    #             log.setLevel(logging.INFO)
    #             log.info(f'Start of checkpoint {self.name}: {self[0]}.')
    #             log.setLevel(lvl)
    # #         self.log_time()
    #         return self

    #     def __exit__(self, exc_type, exc_val, exc_tb):
    #         self.log_time()

    #         if self.verbose:
    #             lvl = log.level
    #             log.setLevel(logging.INFO)
    #             log.info(f'End of checkpoint {self.name}: {self[-1]}. Duration: {self.duration(return_type="str")}')
    #             log.setLevel(lvl)

    def __str__(self):
        return "['" + "', '".join(dt.datetime.strftime(pt, self.fmt) for pt in self) + "']"

    def __repr__(self):
        return self.__str__()