import scipy as sp

from qcodes import validators as vals
from qcodes.instrument.parameter import ManualParameter


def mwg_with_lo_calibration_template(mwg_class):
    """

    Structure of the calibration data:
        {name: (param, freqs, cal_vals)}

    """

    class MWGWithLOCalibration(mwg_class):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.add_parameter('lo_cal_data',
                               vals=vals.Dict(),
                               parameter_class=ManualParameter,
                               initial_value=dict())
            self.add_parameter('lo_cal_interp_kind',
                               vals=vals.Enum(
                                   'linear', 'nearest', 'zero', 'slinear',
                                   'quadratic', 'cubic', 'previous', 'next'),
                               parameter_class=ManualParameter,
                               initial_value='linear')

            self.frequency.set_parser = \
                lambda val, lo_cal_data=self.lo_cal_data, \
                       lo_cal_interp_kind=self.lo_cal_interp_kind : \
                   self.lo_calib(val, lo_cal_data, lo_cal_interp_kind)

        @staticmethod
        def lo_calib(val, lo_cal_data, lo_cal_interp_kind):
            for par, freqs, cal_vals in lo_cal_data().values():
                par(float(sp.interpolate.interp1d(
                    freqs, cal_vals, kind=lo_cal_interp_kind(),
                    fill_value=(min(cal_vals), max(cal_vals)))(val)))
            return val

        # def frequency(self, val=None):
        #     # note that super() does not work for this class construction
        #     # (seems to work in __init__, though)
        #     # This is why we use self.get and self.set here.
        #     if val is None:
        #         return self.get('frequency')
        #     self.set('frequency', val)
        #
        #     for par, freqs, cal_vals in self.lo_cal_data().values():
        #         par(float(sp.interpolate.interp1d(
        #             freqs, cal_vals, kind=self.lo_cal_interp_kind(),
        #             fill_value=(min(cal_vals), max(cal_vals)))(val)))

    return MWGWithLOCalibration
