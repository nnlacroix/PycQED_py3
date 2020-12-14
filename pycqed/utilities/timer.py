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
                 verbose=False, auto_start=True, **kwargs):
        self.fmt = fmt
        self.name = name
        self.name_separator = name_separator
        self.verbose = verbose
        # timer should not start logging when initializing with previous values
        if len(kwargs):
            auto_start = False

        # initialize previous checkpoints
        for ckpt_name, values in kwargs.items():
            if isinstance(values, str):
                values = eval(values)
            try:
                self.checkpoint(ckpt_name, values=values, log_init=False)
            except Exception as e:
                log.warning(f'Could not initialize checkpoint {ckpt_name}. Skipping.')
        if auto_start:
            self.checkpoint(self.NAME_CKPT_START)

    @staticmethod
    def from_dict(timer_dict, **kwargs):
        kwargs.update(dict(auto_start=False))
        tm = Timer(**kwargs)
        for ckpt_name, values in timer_dict.items():
            tm.checkpoint(ckpt_name, values=values, log_init=False)
        return tm

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

    def checkpoint(self, name, values=(), log_every_x=1, log_init=True):
        if name not in self:
            self[name] = Checkpoint(name, values=values, fmt=self.fmt, log_every_x=log_every_x, log_init=log_init)
        else:
            self[name].log_time()

    def duration(self, keys=None, return_type="seconds"):
        if keys is None:
            keys = (self.find_earliest()[0], self.find_latest()[0])
        try:
            duration = self[keys[1]].get("latest")[1] - self[keys[0]].get("earliest")[1]
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

    def find_keys(self, query="", mode='endswith'):
        """
        Finds keys of checkpoints based on query. Defaults to returning all keys.
        Args:
            query (str): string to search
            mode (str): 'exact', 'contains', 'startswith', 'endswith'.
                Decides which string matching function to use.

        Returns:
            matching_keys (list)
        """
        assert mode in ('exact', 'contains', 'startswith', 'endswith'), \
            f"Unknown mode: {mode}"
        matches = []
        for s in self:
            match_func = dict(exact=s.__eq__, contains=s.__contains__,
                              startswith=s.startswith, endswith=s.endswith)
            if match_func[mode](query):
                matches.append(s)
        return matches

    def find_earliest(self, after=None):
        if after is None:
            after = dt.datetime(1900, 1, 1)
        earliest_val = ()
        for i, ckpt_name in enumerate(self):
            earliest_val_ckpt = self[ckpt_name].get("earliest")
            if len(earliest_val_ckpt):
                if after < earliest_val_ckpt[1] and len(earliest_val) == 0:
                    #initialize earliest value with encountered first value after "after"
                    earliest_val = (ckpt_name,) + earliest_val_ckpt
                if after < earliest_val_ckpt[1] < earliest_val[2]:
                    earliest_val = (ckpt_name,) + earliest_val_ckpt
        return earliest_val

    def find_latest(self, before=None):
        if before is None:
            before = dt.datetime(9999, 1, 1) # in case our code is still used in the year 9998.
        latest_val = ()
        for i, ckpt_name in enumerate(self):
            latest_val_ckpt = self[ckpt_name].get("latest") # (index in ckpt, value)
            if len(latest_val_ckpt):
                if before > latest_val_ckpt[1] and len(latest_val) == 0:
                    #initialize earliest value with encountered first value after "after"
                    latest_val = (ckpt_name,) + latest_val_ckpt
                if before > latest_val_ckpt[1] > latest_val[2]:
                    latest_val = (ckpt_name,) + latest_val_ckpt
        return latest_val

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
    def __init__(self, name, values=(), log_every_x=1, fmt="%Y-%m-%d %H:%M:%S.%f",
                 min_timedelta=0, verbose=False, log_init=True):
        super().__init__()
        self.name = name
        self.fmt = fmt
        self.log_every_x = log_every_x
        self.counter = 0
        self.min_timedelta = min_timedelta
        self.verbose = verbose

        for v in values:
            if isinstance(v, str):
                v = dt.datetime.strptime(v, self.fmt)
            self.append(v)
        if log_init:
            self.log_time()

    def get_start(self):
        return self[0]

    def get_end(self):
        return self[-1]

    def get(self, which="latest"):
        """
        Convenience function to get lastest, earliest of nth checkpoint value (and index).
        Returns empty tuple if self is empty.
        Args:
            which (str or int): "earliest" returns value with earliest datetime,
                "latest" with the latest datetime, and integer n returns the nth datetime value
                 (starting from earliest)

        Returns:
            (index, datetime_value)

        """
        if which == "latest":
            which_ind = -1
        elif which == "earliest":
            which_ind = 0
        elif isinstance(which, int):
            which_ind = which
        else:
            raise ValueError(f'which not in ("latest", "earliest", integer): {which}')
        argsorted = np.argsort(self)[::-1] # from recent to latest
        if len(argsorted):
            return argsorted[which_ind], self[which_ind]
        else:
            return ()

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