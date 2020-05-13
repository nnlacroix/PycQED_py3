"""
Library containing various pulse shapes.
"""

import sys
import numpy as np
import scipy as sp
from pycqed.measurement.waveform_control import pulse

pulse.pulse_libraries.add(sys.modules[__name__])


class SSB_DRAG_pulse(pulse.Pulse):
    """In-phase Gaussian pulse with derivative quadrature and SSB modulation.

    Modulation and mixer predistortion added with `apply_modulation` function.

    Args:
        name (str): Name of the pulse, used for referencing to other pulses in a
            sequence. Typically generated automatically by the `Segment` class.
        element_name (str): Name of the element the pulse should be played in.
        I_channel (str): In-phase output channel name.
        Q_channel (str): Quadrature output channel name.
        codeword (int or 'no_codeword'): The codeword that the pulse belongs in.
            Defaults to 'no_codeword'.
        amplitude (float): Pulse amplitude in Volts. Defaults to 0.1 V.
        sigma (float): Pulse width standard deviation in seconds. Defaults to
            250 ns.
        nr_sigma (float): Pulse clipping length in units of pulse sigma. Total
            pulse length will be `nr_sigma*sigma`. Defaults to 4.
        motzoi (float): Amplitude of the derivative quadrature in units of
            pulse sigma. Defautls to 0.
        mod_frequency (float): Pulse modulation frequency in Hz. Defaults to
            1 MHz.
        phase (float): Pulse modulation phase in degrees. Defaults to 0.
        phaselock (bool): The phase reference time is the start of the algorithm
            if True and the middle of the pulse otherwise. Defaults to True.
        alpha (float): Ratio of the I_channel and Q_channel output. Defaults to
            1.
        phi_skew (float): Phase offset between I_channel and Q_channel, in
            addition to the nominal 90 degrees. Defaults to 0.
    """

    def __init__(self, name, element_name, I_channel, Q_channel,
                 codeword='no_codeword', amplitude=0.1, sigma=250e-9,
                 nr_sigma=4, motzoi=0, mod_frequency=1e6, phase=0,
                 phaselock=True, alpha=1, phi_skew=0, **kw):
        self.name = name
        self.element_name = element_name
        self.codeword = codeword

        self.I_channel = I_channel
        self.Q_channel = Q_channel

        self.amplitude = amplitude
        self.sigma = sigma
        self.nr_sigma = nr_sigma
        self.motzoi = motzoi

        self.mod_frequency = mod_frequency
        self.phase = phase
        self.phaselock = phaselock
        self.alpha = alpha
        self.phi_skew = phi_skew

    @property
    def channels(self):
        return [c for c in [self.I_channel, self.Q_channel] if c is not None]

    @property
    def length(self):
        return self.sigma*self.nr_sigma

    def chan_wf(self, channel, tvals):
        half = self.nr_sigma * self.sigma / 2
        tc = self.algorithm_time() + half

        gauss_env = np.exp(-0.5 * (tvals - tc)**2 / self.sigma**2)
        gauss_env -= np.exp(-0.5 * half**2 / self.sigma**2)
        gauss_env *= self.amplitude * (tvals - tc >= -half) * (
                tvals - tc < half)
        deriv_gauss_env = -self.motzoi * (tvals - tc) * gauss_env / self.sigma

        I_mod, Q_mod = apply_modulation(
            gauss_env, deriv_gauss_env, tvals, self.mod_frequency,
            phase=self.phase, phi_skew=self.phi_skew, alpha=self.alpha,
            tval_phaseref=0 if self.phaselock else tc)

        if channel == self.I_channel:
            return I_mod
        elif channel == self.Q_channel:
            return Q_mod
        else:
            return np.zeros_like(tvals)

    def hashables(self, tstart, channel):
        if channel not in self.channels:
            return []
        hashlist = [type(self), self.algorithm_time() - tstart]
        hashlist += [channel == self.I_channel, self.amplitude, self.sigma]
        hashlist += [self.nr_sigma, self.motzoi, self.mod_frequency]
        phase = self.phase
        phase += 360 * self.phaselock * self.mod_frequency * (
                self.algorithm_time() + self.nr_sigma * self.sigma / 2)
        hashlist += [self.alpha, self.phi_skew, phase]
        return hashlist


class BufferedSquarePulse(pulse.Pulse):
    def __init__(self,
                 element_name,
                 channel=None,
                 channels=None,
                 name='buffered square pulse',
                 **kw):
        super().__init__(name, element_name)
        if channel is None and channels is None:
            raise ValueError('Must specify either channel or channels')
        elif channels is None:
            self.channels.append(channel)
        else:
            self.channels = channels
        self.amplitude = kw.pop('amplitude', 0)

        self.pulse_length = kw.pop('pulse_length', 0)
        self.buffer_length_start = kw.pop('buffer_length_start', 0)
        self.buffer_length_end = kw.pop('buffer_length_end', 0)
        self.gaussian_filter_sigma = kw.pop('gaussian_filter_sigma', 0)
        self.length = self.pulse_length + self.buffer_length_start + \
                      self.buffer_length_end
        self.codeword = kw.pop('codeword', 'no_codeword')

    def chan_wf(self, chan, tvals):
        if self.gaussian_filter_sigma == 0:
            wave = np.ones_like(tvals) * self.amplitude
            wave *= (tvals >= tvals[0] + self.buffer_length_start)
            wave *= (tvals <
                     tvals[0] + self.buffer_length_start + self.pulse_length)
            return wave
        else:
            tstart = tvals[0] + self.buffer_length_start
            tend = tvals[0] + self.buffer_length_start + self.pulse_length
            scaling = 1 / np.sqrt(2) / self.gaussian_filter_sigma
            wave = 0.5 * (sp.special.erf(
                (tvals - tstart) * scaling) - sp.special.erf(
                (tvals - tend) * scaling)) * self.amplitude
            return wave

    def hashables(self, tstart, channel):
        if channel not in self.channels:
            return []
        hashlist = [type(self), self.algorithm_time() - tstart]
        hashlist += [self.amplitude, self.pulse_length]
        hashlist += [self.buffer_length_start, self.buffer_length_end]
        hashlist += [self.gaussian_filter_sigma]
        return hashlist


class BufferedCZPulse(pulse.Pulse):
    def __init__(self,
                 channel,
                 element_name,
                 aux_channels_dict=None,
                 name='buffered CZ pulse',
                 **kw):
        super().__init__(name, element_name)

        self.channel = channel
        self.aux_channels_dict = aux_channels_dict
        self.channels = [self.channel]
        if self.aux_channels_dict is not None:
            self.channels += list(self.aux_channels_dict)

        self.amplitude = kw.pop('amplitude', 0)
        self.frequency = kw.pop('frequency', 0)
        self.phase = kw.pop('phase', 0.)

        self.pulse_length = kw.pop('pulse_length', 0)
        self.buffer_length_start = kw.pop('buffer_length_start', 0)
        self.buffer_length_end = kw.pop('buffer_length_end', 0)
        self.extra_buffer_aux_pulse = kw.pop('extra_buffer_aux_pulse', 5e-9)
        self.gaussian_filter_sigma = kw.pop('gaussian_filter_sigma', 0)
        self.length = self.pulse_length + self.buffer_length_start + \
                      self.buffer_length_end
        self.codeword = kw.pop('codeword', 'no_codeword')

    def chan_wf(self, chan, tvals):
        amp = self.amplitude
        buffer_start = self.buffer_length_start
        buffer_end = self.buffer_length_end
        pulse_length = self.pulse_length
        if chan != self.channel:
            amp = self.aux_channels_dict[chan]
            buffer_start -= self.extra_buffer_aux_pulse
            buffer_end -= self.extra_buffer_aux_pulse
            pulse_length += 2 * self.extra_buffer_aux_pulse

        if self.gaussian_filter_sigma == 0:
            wave = np.ones_like(tvals) * amp
            wave *= (tvals >= tvals[0] + buffer_start)
            wave *= (tvals < tvals[0] + buffer_start + pulse_length)
        else:
            tstart = tvals[0] + buffer_start
            tend = tvals[0] + buffer_start + pulse_length
            scaling = 1 / np.sqrt(2) / self.gaussian_filter_sigma
            wave = 0.5 * (sp.special.erf(
                (tvals - tstart) * scaling) - sp.special.erf(
                (tvals - tend) * scaling)) * amp
        t_rel = tvals - tvals[0]
        wave *= np.cos(
            2 * np.pi * (self.frequency * t_rel + self.phase / 360.))
        return wave

    def hashables(self, tstart, channel):
        if channel not in self.channels:
            return []
        hashlist = [type(self), self.algorithm_time() - tstart]

        amp = self.amplitude
        buffer_start = self.buffer_length_start
        buffer_end = self.buffer_length_end
        pulse_length = self.pulse_length
        if channel != self.channel:
            amp = self.aux_channels_dict[channel]
            buffer_start -= self.extra_buffer_aux_pulse
            buffer_end -= self.extra_buffer_aux_pulse
            pulse_length += 2 * self.extra_buffer_aux_pulse

        hashlist += [amp, pulse_length, buffer_start, buffer_end]
        hashlist += [self.gaussian_filter_sigma]
        hashlist += [self.frequency, self.phase % 360]
        return hashlist


class BufferedCZPulseEffectiveTime(pulse.Pulse):
    def __init__(self,
                 channel,
                 element_name, chevron_func,
                 aux_channels_dict=None,
                 name='buffered CZ pulse effective time',
                 **kw):
        super().__init__(name, element_name)

        self.channel = channel
        self.aux_channels_dict = aux_channels_dict
        self.channels = [self.channel]
        if self.aux_channels_dict is not None:
            self.channels += list(self.aux_channels_dict)

        self.amplitude = kw.pop('amplitude', 0)
        self.frequency = kw.pop('frequency', 0)
        self.phase = kw.pop('phase', 0.)

        self.chevron_func = chevron_func
        # length rescaled to have a "straight" chevron -- parameter set by user
        self.pulse_length = kw.pop('pulse_length', 0)
        # physical length of pulse, computed using the model in chevron_func,
        # which is the true length of the pulse
        self.pulse_physical_length = self.chevron_func(self.amplitude,
                                                       self.pulse_length)
        self.buffer_length_start = kw.pop('buffer_length_start', 0)
        self.buffer_length_end = kw.pop('buffer_length_end', 0)
        self.extra_buffer_aux_pulse = kw.pop('extra_buffer_aux_pulse', 5e-9)
        self.gaussian_filter_sigma = kw.pop('gaussian_filter_sigma', 0)
        self.length = self.pulse_physical_length + self.buffer_length_start + \
                      self.buffer_length_end
        self.codeword = kw.pop('codeword', 'no_codeword')

    def chan_wf(self, chan, tvals):
        amp = self.amplitude
        buffer_start = self.buffer_length_start
        buffer_end = self.buffer_length_end
        pulse_physical_length = self.pulse_physical_length
        if chan != self.channel:
            amp = self.aux_channels_dict[chan]
            buffer_start -= self.extra_buffer_aux_pulse
            buffer_end -= self.extra_buffer_aux_pulse
            pulse_physical_length += 2 * self.extra_buffer_aux_pulse

        if self.gaussian_filter_sigma == 0:
            wave = np.ones_like(tvals) * amp
            wave *= (tvals >= tvals[0] + buffer_start)
            wave *= (tvals < tvals[0] + buffer_start + pulse_physical_length)
        else:
            tstart = tvals[0] + buffer_start
            tend = tvals[0] + buffer_start + pulse_physical_length
            scaling = 1 / np.sqrt(2) / self.gaussian_filter_sigma
            wave = 0.5 * (sp.special.erf(
                (tvals - tstart) * scaling) - sp.special.erf(
                (tvals - tend) * scaling)) * amp
        t_rel = tvals - tvals[0]
        wave *= np.cos(
            2 * np.pi * (self.frequency * t_rel + self.phase / 360.))
        return wave

    def hashables(self, tstart, channel):
        if channel not in self.channels:
            return []
        hashlist = [type(self), self.algorithm_time() - tstart]

        amp = self.amplitude
        buffer_start = self.buffer_length_start
        buffer_end = self.buffer_length_end
        pulse_physical_length = self.pulse_physical_length
        if channel != self.channel:
            amp = self.aux_channels_dict[channel]
            buffer_start -= self.extra_buffer_aux_pulse
            buffer_end -= self.extra_buffer_aux_pulse
            pulse_physical_length += 2 * self.extra_buffer_aux_pulse

        hashlist += [amp, pulse_physical_length, buffer_start, buffer_end]
        hashlist += [self.gaussian_filter_sigma]
        hashlist += [self.frequency, self.phase % 360]
        return hashlist


class NZBufferedCZPulse(pulse.Pulse):
    def __init__(self, channel, element_name, aux_channels_dict=None,
                 name='NZ buffered CZ pulse', **kw):
        super().__init__(name, element_name)

        self.channel = channel
        self.aux_channels_dict = aux_channels_dict
        self.channels = [self.channel]
        if self.aux_channels_dict is not None:
            self.channels += list(self.aux_channels_dict)

        self.amplitude = kw.pop('amplitude', 0)  # of first half
        self.alpha = kw.pop('alpha', 1)  # this will be applied to 2nd half
        self.pulse_length = kw.pop('pulse_length', 0)
        self.length1 = self.alpha * self.pulse_length / (self.alpha + 1)

        self.buffer_length_start = kw.pop('buffer_length_start', 0)
        self.buffer_length_end = kw.pop('buffer_length_end', 0)
        self.extra_buffer_aux_pulse = kw.pop('extra_buffer_aux_pulse', 5e-9)
        self.gaussian_filter_sigma = kw.pop('gaussian_filter_sigma', 0)
        self.length = self.pulse_length + self.buffer_length_start + \
                      self.buffer_length_end

        # these are here so that we can use the CZ pulse dictionary that is
        # created by add_CZ_pulse in QuDev_transmon.py
        self.frequency = kw.pop('frequency', 0)
        self.phase = kw.pop('phase', 0.)
        self.codeword = kw.pop('codeword', 'no_codeword')

        self.codeword = kw.pop('codeword', 'no_codeword')

    def chan_wf(self, chan, tvals):
        amp1 = self.amplitude
        amp2 = -self.amplitude * self.alpha
        buffer_start = self.buffer_length_start
        buffer_end = self.buffer_length_end
        pulse_length = self.pulse_length
        l1 = self.length1
        if chan != self.channel:
            amp1 = self.aux_channels_dict[chan] * amp1
            amp2 = -amp1 * self.alpha
            buffer_start -= self.extra_buffer_aux_pulse
            buffer_end -= self.extra_buffer_aux_pulse
            pulse_length += 2 * self.extra_buffer_aux_pulse
            l1 = self.alpha * pulse_length / (self.alpha + 1)

        if self.gaussian_filter_sigma == 0:
            wave1 = np.ones_like(tvals) * amp1
            wave1 *= (tvals >= tvals[0] + buffer_start)
            wave1 *= (tvals < tvals[0] + buffer_start + l1)

            wave2 = np.ones_like(tvals) * amp2
            wave2 *= (tvals >= tvals[0] + buffer_start + l1)
            wave2 *= (tvals < tvals[0] + buffer_start + pulse_length)

            wave = wave1 + wave2
        else:
            tstart = tvals[0] + buffer_start
            tend = tvals[0] + buffer_start + l1
            tend2 = tvals[0] + buffer_start + pulse_length
            scaling = 1 / np.sqrt(2) / self.gaussian_filter_sigma
            wave = 0.5 * (amp1 * sp.special.erf((tvals - tstart) * scaling) -
                          amp1 * sp.special.erf((tvals - tend) * scaling) +
                          amp2 * sp.special.erf((tvals - tend) * scaling) -
                          amp2 * sp.special.erf((tvals - tend2) * scaling))
        return wave

    def hashables(self, tstart, channel):
        if channel not in self.channels:
            return []
        hashlist = [type(self), self.algorithm_time() - tstart]

        amp = self.amplitude
        buffer_start = self.buffer_length_start
        buffer_end = self.buffer_length_end
        pulse_length = self.pulse_length
        if channel != self.channel:
            amp = self.aux_channels_dict[channel]
            buffer_start -= self.extra_buffer_aux_pulse
            buffer_end -= self.extra_buffer_aux_pulse
            pulse_length += 2 * self.extra_buffer_aux_pulse

        hashlist += [amp, pulse_length, buffer_start, buffer_end]
        hashlist += [self.gaussian_filter_sigma, self.alpha]
        return hashlist


class NZMartinisGellarPulse(pulse.Pulse):
    def __init__(self, channel, element_name, wave_generation_func,
                 aux_channels_dict=None,
                 name='NZMartinisGellarPulse', **kw):
        super().__init__(name, element_name)

        self.channel = channel
        self.aux_channels_dict = aux_channels_dict
        self.channels = [self.channel]
        if self.aux_channels_dict is not None:
            self.channels += list(self.aux_channels_dict)

        self.theta_f = kw.pop('theta_f', np.pi / 2)
        self.alpha = kw.pop('alpha', 1)  # this will be applied to 2nd half
        self.pulse_length = kw.pop('pulse_length', 0)

        self.buffer_length_start = kw.pop('buffer_length_start', 0)
        self.buffer_length_end = kw.pop('buffer_length_end', 0)
        self.extra_buffer_aux_pulse = kw.pop('extra_buffer_aux_pulse', 0e-9)
        self.length = self.pulse_length + self.buffer_length_start + \
                      self.buffer_length_end

        self.wave_generation_func = wave_generation_func
        self.qbc_freq = kw.pop('qbc_freq', 0)
        self.qbt_freq = kw.pop('qbt_freq', 0)
        self.anharmonicity = kw.pop('anharmonicity', 0)
        self.J = kw.pop('J', 0)
        self.loop_asym = kw.pop('loop_asym', 0)
        self.dv_dphi = kw.pop('dv_dphi', 0)
        self.lambda_2 = kw.pop('lambda_2', 0)
        self.codeword = kw.pop('codeword', 'no_codeword')

    def chan_wf(self, chan, tvals):

        dv_dphi = self.dv_dphi
        if chan != self.channel:
            dv_dphi *= self.aux_channels_dict[chan]

        params_dict = {
            'pulse_length': self.pulse_length,
            'theta_f': self.theta_f,
            'qbc_freq': self.qbc_freq,
            'qbt_freq': self.qbt_freq,
            'anharmonicity': self.anharmonicity,
            'J': self.J,
            'dv_dphi': dv_dphi,
            'loop_asym': self.loop_asym,
            'lambda_2': self.lambda_2,
            'alpha': self.alpha,
            'buffer_length_start': self.buffer_length_start
        }
        return self.wave_generation_func(tvals, params_dict)

    def hashables(self, tstart, channel):
        if channel not in self.channels:
            return []
        hashlist = [type(self), self.algorithm_time() - tstart]
        hashlist += [self.pulse_length, self.theta_f, self.qbc_freq]
        hashlist += [self.qbt_freq, self.anharmonicity, self.J, self.dv_dphi]
        hashlist += [self.loop_asym, self.lambda_2, self.alpha]
        hashlist += [self.buffer_length_start, hash(self.wave_generation_func)]
        return hashlist


class GaussFilteredCosIQPulse(pulse.Pulse):
    def __init__(self,
                 I_channel,
                 Q_channel,
                 element_name,
                 name='gauss filtered cos IQ pulse',
                 **kw):
        super().__init__(name, element_name)

        self.I_channel = I_channel
        self.Q_channel = Q_channel
        self.channels = [self.I_channel, self.Q_channel]

        self.amplitude = kw.pop('amplitude', 0)
        self.mod_frequency = kw.pop('mod_frequency', 0)
        self.phase = kw.pop('phase', 0.)
        self.phi_skew = kw.pop('phi_skew', 0.)
        self.alpha = kw.pop('alpha', 1.)

        self.pulse_length = kw.pop('pulse_length', 0)
        self.gaussian_filter_sigma = kw.pop('gaussian_filter_sigma', 0)
        self.nr_sigma = kw.pop('nr_sigma', 5)
        self.phase_lock = kw.pop('phase_lock', False)
        self.length = self.pulse_length + \
                      self.gaussian_filter_sigma * self.nr_sigma
        self.codeword = kw.pop('codeword', 'no_codeword')

    def chan_wf(self, chan, tvals, **kw):
        if self.gaussian_filter_sigma == 0:
            wave = np.ones_like(tvals) * self.amplitude
            wave *= (tvals >= tvals[0])
            wave *= (tvals < tvals[0] + self.pulse_length)
        else:
            tstart = tvals[0] + 0.5 * self.gaussian_filter_sigma * self.nr_sigma
            tend = tstart + self.pulse_length
            scaling = 1 / np.sqrt(2) / self.gaussian_filter_sigma
            wave = 0.5 * (sp.special.erf(
                (tvals - tstart) * scaling) - sp.special.erf(
                (tvals - tend) * scaling)) * self.amplitude
        I_mod, Q_mod = apply_modulation(
            wave,
            np.zeros_like(wave),
            tvals,
            mod_frequency=self.mod_frequency,
            phase=self.phase,
            phi_skew=self.phi_skew,
            alpha=self.alpha,
            tval_phaseref=0 if self.phase_lock else self.algorithm_time())

        if chan == self.I_channel:
            return I_mod
        if chan == self.Q_channel:
            return Q_mod

    def hashables(self, tstart, channel):
        if channel not in self.channels:
            return []
        hashlist = [type(self), self.algorithm_time() - tstart]
        hashlist += [channel == self.I_channel, self.amplitude]
        hashlist += [self.mod_frequency, self.gaussian_filter_sigma]
        hashlist += [self.nr_sigma, self.pulse_length]
        phase = self.phase
        phase += 360 * (not self.phase_lock) * self.mod_frequency \
                 * self.algorithm_time()
        hashlist += [self.alpha, self.phi_skew, phase]
        return hashlist


class GaussFilteredCosIQPulseMultiChromatic(pulse.Pulse):
    def __init__(self,
                 I_channel,
                 Q_channel,
                 element_name,
                 name='gauss filtered cos IQ pulse multi chromatic',
                 **kw):
        super().__init__(name, element_name)

        self.I_channel = I_channel
        self.Q_channel = Q_channel
        self.channels = [self.I_channel, self.Q_channel]

        self.amplitude = kw.pop('amplitude', 0)
        self.mod_frequency = kw.pop('mod_frequency', [0])
        if np.ndim(self.mod_frequency) != 1:
            raise ValueError("MultiChromatic Pulse requires a list or 1D array "
                             f"of frequencies. Instead {self.mod_frequency} "
                             f"was given")
        self.phase = kw.pop('phase', 0.)
        self.phi_skew = kw.pop('phi_skew', 0.)
        self.alpha = kw.pop('alpha', 1.)

        self.pulse_length = kw.pop('pulse_length', 0)
        self.gaussian_filter_sigma = kw.pop('gaussian_filter_sigma', 0)
        self.nr_sigma = kw.pop('nr_sigma', 5)
        self.phase_lock = kw.pop('phase_lock', False)
        self.length = self.pulse_length + \
                      self.gaussian_filter_sigma * self.nr_sigma
        self.codeword = kw.pop('codeword', 'no_codeword')

        params = dict(amplitude=self.amplitude,
                      phase=self.phase,
                      phi_skew=self.phi_skew,
                      alpha=self.alpha)
        for pname, p in params.items():

            if np.ndim(p) == 0:
                setattr(self, pname, len(self.mod_frequency) * [p])
            elif len(p) != len(self.mod_frequency):
                raise ValueError(f"Received {len(p)} {pname}  but expected "
                                 f"{len(self.mod_frequency)} (number of frequencies)")

    def chan_wf(self, chan, tvals, **kw):
        I_mods, Q_mods = np.zeros_like(tvals), np.zeros_like(tvals)
        for a, ph, f, phi, alpha in zip(self.amplitude, self.phase,
                                        self.mod_frequency, self.phi_skew,
                                        self.alpha):
            if self.gaussian_filter_sigma == 0:
                wave = np.ones_like(tvals) * a
                wave *= (tvals >= tvals[0])
                wave *= (tvals < tvals[0] + self.pulse_length)
            else:
                tstart = tvals[
                             0] + 0.5 * self.gaussian_filter_sigma * self.nr_sigma
                tend = tstart + self.pulse_length
                scaling = 1 / np.sqrt(2) / self.gaussian_filter_sigma
                wave = 0.5 * (sp.special.erf(
                    (tvals - tstart) * scaling) - sp.special.erf(
                    (tvals - tend) * scaling)) * a
            I_mod, Q_mod = apply_modulation(
                wave,
                np.zeros_like(wave),
                tvals,
                mod_frequency=f,
                phase=ph,
                phi_skew=phi,
                alpha=alpha,
                tval_phaseref=0 if self.phase_lock else self.algorithm_time())
            I_mods += I_mod
            Q_mods += Q_mod
        if chan == self.I_channel:
            return I_mods
        if chan == self.Q_channel:
            return Q_mods

    def hashables(self, tstart, channel):
        if channel not in self.channels:
            return []
        hashlist = [type(self), self.algorithm_time() - tstart]
        hashlist += [channel == self.I_channel]
        hashlist += list(self.amplitude)
        hashlist += self.mod_frequency
        hashlist += [self.gaussian_filter_sigma]
        hashlist += [self.nr_sigma, self.pulse_length]
        phase = [p + 360 * (not self.phase_lock) * f * self.algorithm_time() \
                 for p, f in zip(self.phase, self.mod_frequency)]
        hashlist += self.alpha
        hashlist += self.phi_skew
        hashlist += phase
        return hashlist


class VirtualPulse(pulse.Pulse):
    def __init__(self, name, element_name, **kw):
        super().__init__(name, element_name)
        self.codeword = kw.pop('codeword', 'no_codeword')
        self.pulse_length = kw.pop('pulse_length', 0)
        self.length = self.pulse_length

    def chan_wf(self, chan, tvals):
        return {}

    def hashables(self, tstart, channel):
        return []


class SquarePulse(pulse.Pulse):
    def __init__(self, element_name, channel=None, channels=None,
                 name='square pulse', **kw):
        super().__init__(name, element_name)
        if channel is None and channels is None:
            raise ValueError('Must specify either channel or channels')
        elif channels is None:
            self.channel = channel  # this is just for convenience, internally
            # this is the part the sequencer element wants to communicate with
            self.channels.append(channel)
        else:
            self.channels = channels
        self.amplitude = kw.pop('amplitude', 0)
        self.length = kw.pop('length', 0)
        self.codeword = kw.pop('codeword', 'no_codeword')

    def chan_wf(self, chan, tvals):
        return np.ones(len(tvals)) * self.amplitude

    def hashables(self, tstart, channel):
        if channel not in self.channels:
            return []
        hashlist = [type(self), self.algorithm_time() - tstart]
        hashlist += [self.amplitude, self.length]
        return hashlist


class CosPulse(pulse.Pulse):
    def __init__(self, channel, element_name, name='cos pulse', **kw):
        super().__init__(name, element_name)

        self.channel = channel  # this is just for convenience, internally
        self.channels.append(channel)
        # this is the part the sequencer element wants to communicate with
        self.frequency = kw.pop('frequency', 1e6)
        self.amplitude = kw.pop('amplitude', 0.)
        self.length = kw.pop('length', 0.)
        self.phase = kw.pop('phase', 0.)
        self.codeword = kw.pop('codeword', 'no_codeword')

    def chan_wf(self, chan, tvals):
        return self.amplitude * np.cos(2 * np.pi *
                                       (self.frequency * tvals +
                                        self.phase / 360.))

    def hashables(self, tstart, channel):
        if channel not in self.channels:
            return []
        hashlist = [type(self), self.algorithm_time() - tstart]
        hashlist += [self.amplitude, self.length, self.frequency]
        hashlist += [(self.phase + self.frequency * tstart * 360) % 360.]
        return hashlist


def apply_modulation(ienv, qenv, tvals, mod_frequency,
                     phase=0., phi_skew=0., alpha=1., tval_phaseref=0.):
    """
    Applies single sideband modulation, requires tvals to make sure the
    phases are correct.

    The modulation and predistortion is calculated as
    [I_mod] = [cos(phi_skew)  sin(phi_skew)] [ cos(wt)  sin(wt)] [I_env]
    [Q_mod]   [0              1/alpha      ] [-sin(wt)  cos(wt)] [Q_env],
    where wt = 360 * mod_frequency * (tvals - tval_phaseref) + phase

    The output is normalized such that the determinatnt of the transformation
    matrix is +-1.

    Args:
        ienv (np.ndarray): In-phase envelope waveform.
        qenv (np.ndarray): Quadrature envelope waveform.
        tvals (np.ndarray): Sample start times in seconds.
        mod_frequency (float): Modulation frequency in Hz.
        phase (float): Phase of modulation in degrees. Defaults to 0.
        phi_skew (float): Phase offset between I_channel and Q_channel, in
            addition to the nominal 90 degrees. Defaults to 0.
        alpha (float): Ratio of the I_channel and Q_channel output.
            Defaults to 1.
        tval_phaseref: The reference time in seconds for calculating phase.
            Defaults to 0.

    Returns:
        np.ndarray, np.ndarray: The predistorted and modulated outputs.
    """
    phi = 360 * mod_frequency * (tvals - tval_phaseref) + phase
    phii = phi + phi_skew
    phiq = phi + 90

    # k = 1 / np.cos(np.pi * phi_skew / 180) #  old normalization
    k = np.sqrt(np.abs(alpha / np.cos(np.deg2rad(phi_skew))))

    imod = k * (ienv * np.cos(np.deg2rad(phii)) +
                qenv * np.sin(np.deg2rad(phii)))
    qmod = k * (ienv * np.cos(np.deg2rad(phiq)) +
                qenv * np.sin(np.deg2rad(phiq))) / alpha

    return imod, qmod
