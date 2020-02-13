import numpy as np
from copy import deepcopy
try:
    from math import gcd
except:  # Moved to math in python 3.5, this is to be 3.4 compatible
    from fractions import gcd
from pycqed.measurement.waveform_control import pulse
from pycqed.measurement.waveform_control import sequence
from pycqed.measurement.waveform_control import pulsar as ps
from pycqed.measurement.pulse_sequences.standard_elements import multi_pulse_elt
from pycqed.measurement.pulse_sequences.single_qubit_tek_seq_elts import \
    sweep_pulse_params, add_preparation_pulses, pulse_list_list_seq
from pycqed.measurement.pulse_sequences.multi_qubit_tek_seq_elts import \
    generate_mux_ro_pulse_list

from importlib import reload
reload(pulse)
from pycqed.measurement.waveform_control import pulse_library
reload(pulse_library)

import logging
log = logging.getLogger(__name__)


def get_pulse_dict_from_pars(pulse_pars):
    '''
    Returns a dictionary containing pulse_pars for all the primitive pulses
    based on a single set of pulse_pars.
    Using this function deepcopies the pulse parameters preventing accidently
    editing the input dictionary.

    input args:
        pulse_pars: dictionary containing pulse_parameters
    return:
        pulses: dictionary of pulse_pars dictionaries
    '''
    pi_amp = pulse_pars['amplitude']
    pi2_amp = pulse_pars['amplitude']*pulse_pars['amp90_scale']

    pulses = {'I': deepcopy(pulse_pars),
              'X180': deepcopy(pulse_pars),
              'mX180': deepcopy(pulse_pars),
              'X90': deepcopy(pulse_pars),
              'mX90': deepcopy(pulse_pars),
              'Y180': deepcopy(pulse_pars),
              'mY180': deepcopy(pulse_pars),
              'Y90': deepcopy(pulse_pars),
              'mY90': deepcopy(pulse_pars)}

    pulses['I']['amplitude'] = 0
    pulses['mX180']['amplitude'] = -pi_amp
    pulses['X90']['amplitude'] = pi2_amp
    pulses['mX90']['amplitude'] = -pi2_amp
    pulses['Y180']['phase'] = 90
    pulses['mY180']['phase'] = 90
    pulses['mY180']['amplitude'] = -pi_amp

    pulses['Y90']['amplitude'] = pi2_amp
    pulses['Y90']['phase'] = 90
    pulses['mY90']['amplitude'] = -pi2_amp
    pulses['mY90']['phase'] = 90

    return pulses


def Ramsey_with_flux_pulse_meas_seq(thetas, qb, X90_separation, verbose=False,
                                    upload=True, return_seq=False,
                                    cal_points=False):
    '''
    Performs a Ramsey with interleaved Flux pulse

    Timings of sequence
           <----- |fluxpulse|
        |X90|  -------------------     |X90|  ---  |RO|
                                     sweep phase

    timing of the flux pulse relative to the center of the first X90 pulse

    Args:
        thetas: numpy array of phase shifts for the second pi/2 pulse
        qb: qubit object (must have the methods get_operation_dict(),
        get_drive_pars() etc.
        X90_separation: float (separation of the two pi/2 pulses for Ramsey
        verbose: bool
        upload: bool
        return_seq: bool

    Returns:
        if return_seq:
          seq: qcodes sequence
          el_list: list of pulse elements
        else:
            seq_name: string
    '''

    qb_name = qb.name
    operation_dict = qb.get_operation_dict()
    pulse_pars = qb.get_drive_pars()
    RO_pars = qb.get_RO_pars()
    seq_name = 'Measurement_Ramsey_sequence_with_Flux_pulse'
    seq = sequence.Sequence(seq_name)
    el_list = []

    pulses = get_pulse_dict_from_pars(pulse_pars)
    flux_pulse = operation_dict["flux "+qb_name]
    # Used for checking dynamic phase compensation
    # if flux_pulse['amplitude'] != 0:
    #     flux_pulse['basis_rotation'] = {qb_name: -80.41028958782647}

    flux_pulse['ref_point'] = 'end'
    X90_2 = deepcopy(pulses['X90'])
    X90_2['pulse_delay'] = X90_separation - flux_pulse['pulse_delay'] \
                            - X90_2['nr_sigma']*X90_2['sigma']
    X90_2['ref_point'] = 'start'

    for i, theta in enumerate(thetas):
        X90_2['phase'] = theta*180/np.pi
        if cal_points and (i == (len(thetas)-4) or i == (len(thetas)-3)):
            el = multi_pulse_elt(i, station, [RO_pars])
        elif cal_points and (i == (len(thetas)-2) or i == (len(thetas)-1)):
            flux_pulse['amplitude'] = 0
            el = multi_pulse_elt(i, station,
                                 [pulses['X90'], flux_pulse, X90_2, RO_pars])
        else:
            el = multi_pulse_elt(i, station,
                                 [pulses['X90'], flux_pulse, X90_2, RO_pars])
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)

    if return_seq:
        return seq, el_list
    else:
        return seq_name


def dynamic_phase_seq(qb_name, hard_sweep_dict, operation_dict,
                      cz_pulse_name, cal_points=None, prepend_n_cz=0,
                      upload=False, prep_params=dict()):
    '''
    Performs a Ramsey with interleaved Flux pulse
    Sequence
                   |fluxpulse|
        |X90|  -------------------     |X90|  ---  |RO|
                                     sweep phase
    Optional: prepend n Flux pulses before starting ramsey
    '''

    seq_name = 'Dynamic_phase_seq'

    ge_half_start = deepcopy(operation_dict['X90 ' + qb_name])
    ge_half_start['name'] = 'pi_half_start'
    # ge_half_start['element_name'] = 'pi_half_start_el'
    ge_half_start['element_name'] = 'pi'

    flux_pulse = deepcopy(operation_dict[cz_pulse_name])
    flux_pulse['name'] = 'flux'
    flux_pulse['element_name'] = 'flux_el'

    ge_half_end = deepcopy(operation_dict['X90 ' + qb_name])
    ge_half_end['name'] = 'pi_half_end'
    # ge_half_end['element_name'] = 'pi_half_end_el'
    ge_half_end['element_name'] = 'pi'

    ro_pulse = deepcopy(operation_dict['RO ' + qb_name])

    pulse_list = [deepcopy(operation_dict[cz_pulse_name])
                  for _ in range(prepend_n_cz)]

    pulse_list += [ge_half_start, flux_pulse, ge_half_end, ro_pulse]
    hsl = len(list(hard_sweep_dict.values())[0]['values'])
    if 'amplitude' in flux_pulse:
        param_to_set = 'amplitude'
    elif 'dv_dphi' in flux_pulse:
        param_to_set = 'dv_dphi'
    else:
        raise ValueError('Unknown flux pulse amplitude control parameter. '
                         'Cannot do measurement without flux pulse.')

    params = {f'flux.{param_to_set}': np.concatenate(
        [flux_pulse[param_to_set]*np.ones(hsl//2), np.zeros(hsl//2)])}
    print((hsl // 2))

    if 'aux_channels_dict' in flux_pulse:
        params.update({'flux.aux_channels_dict': np.concatenate([
            [flux_pulse['aux_channels_dict']] * (hsl // 2),
             [{}] * (hsl // 2)])})
    params.update({f'pi_half_end.{k}': v['values']
                   for k, v in hard_sweep_dict.items()})
    swept_pulses = sweep_pulse_params(pulse_list, params)
    for k, p in enumerate(swept_pulses):
        for prepended_cz_idx in range(prepend_n_cz):
            fp = p[prepended_cz_idx]
            fp['element_name'] = 'flux_el_{}'.format(k)
        fp = p[prepend_n_cz + 1]
        fp['element_name'] = 'flux_el_{}'.format(k)
    swept_pulses_with_prep = \
        [add_preparation_pulses(p, operation_dict, [qb_name], **prep_params)
         for p in swept_pulses]
    seq = pulse_list_list_seq(swept_pulses_with_prep, seq_name, upload=False)

    if cal_points is not None:
        # add calibration segments
        seq.extend(cal_points.create_segments(operation_dict, **prep_params))

    log.debug(seq)
    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)

    return seq, np.arange(seq.n_acq_elements())


def rabi_flux_pulse_sequence(amplitudes, cz_pulse_amp, qb_name,
                             operation_dict, cz_pulse_name, n=1,
                             delay=None, cal_points=None,
                             prep_params=dict(), upload=True):
    '''
    Performs X180 pulse on top of a fluxpulse

    Timings of sequence

       |          ----------           |X180|  ------------------------ |RO|
       |          ---    | --------- fluxpulse ---------- |
    '''

    seq_name = 'Fluxpulse_amplitude_sequence'
    drag_length = operation_dict['X180 ' + qb_name]['sigma'] * operation_dict[
        'X180 ' + qb_name]['nr_sigma']

    rabi_pulses = n*['']
    for i in range(n):
        rabi_pulse = deepcopy(operation_dict['X180 ' + qb_name])
        rabi_pulse['name'] = f'Rabi_{i}'
        rabi_pulse['element_name'] = 'Rabi_el'
        rabi_pulses[i] = rabi_pulse

    flux_pulse = deepcopy(operation_dict[cz_pulse_name])
    flux_pulse['name'] = 'FPA_Flux'
    flux_pulse['amplitude'] = cz_pulse_amp
    flux_pulse['pulse_length'] = (n+1) * drag_length
    flux_pulse['ref_pulse'] = f'Rabi_{n//2}'
    flux_pulse['ref_point'] = 'middle'
    flux_pulse['ref_point_new'] = 'middle'

    # if delay is None:
    #     delay = flux_pulse['pulse_length'] / 2
    # flux_pulse['pulse_delay'] = -flux_pulse.get('buffer_length_start',
    #                                             0) - delay

    ro_pulse = deepcopy(operation_dict['RO ' + qb_name])
    ro_pulse['name'] = 'FPA_Ro'
    # ro_pulse['ref_pulse'] = 'Rabi_0'
    # ro_pulse['ref_point'] = 'middle'
    # ro_pulse['pulse_delay'] = flux_pulse['pulse_length'] - delay + \
    #                           flux_pulse.get('buffer_length_end', 0)
    ro_pulse['ref_pulse'] = 'FPA_Flux'

    pulses = rabi_pulses + [flux_pulse, ro_pulse]
    swept_pulses = sweep_pulse_params(pulses,
                                      {f'Rabi_{i}.amplitude': amplitudes
                                       for i in range(n)})

    if cal_points is not None:
        # add calibration segments
        I = deepcopy(operation_dict['I ' + qb_name])
        I['name'] = 'Ical'
        X180 = deepcopy(operation_dict['X180 ' + qb_name])
        X180['name'] = 'X180cal'
        for i, cal_pulse in enumerate([I, X180]):
            cal_pulse['element_name'] = f'cal{i}'
            flux_pulse = deepcopy(operation_dict[cz_pulse_name])
            flux_pulse['name'] = 'cal_flux'
            flux_pulse['pulse_length'] = 2 * drag_length
            flux_pulse['amplitude'] = cz_pulse_amp
            flux_pulse['ref_pulse'] = cal_pulse['name']
            flux_pulse['ref_point'] = 'middle'
            flux_pulse['ref_point_new'] = 'middle'
            # delay = flux_pulse['pulse_length'] / 2
            # flux_pulse['pulse_delay'] = -flux_pulse.get(
            #     'buffer_length_start', 0) - delay
            ro_pulse = deepcopy(operation_dict['RO ' + qb_name])
            ro_pulse['name'] = 'cal_ro'
            ro_pulse['ref_pulse'] = 'cal_flux'
            # ro_pulse['ref_pulse'] = cal_pulse['name']
            # ro_pulse['ref_point'] = 'middle'
            # ro_pulse['pulse_delay'] = flux_pulse['pulse_length'] - delay + \
            #                           flux_pulse.get('buffer_length_end', 0)
            swept_pulses += [[cal_pulse, ro_pulse, flux_pulse]]
            swept_pulses += [[cal_pulse, ro_pulse, flux_pulse]]

    swept_pulses_with_prep = \
        [add_preparation_pulses(p, operation_dict, [qb_name], **prep_params)
         for p in swept_pulses]

    seq = pulse_list_list_seq(swept_pulses_with_prep, seq_name, upload=False)

    log.debug(seq)
    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)

    return seq, np.arange(seq.n_acq_elements())


def fluxpulse_amplitude_sequence(amplitudes,
                                 freqs,
                                 qb_name,
                                 operation_dict,
                                 flux_pulse_name,
                                 delay,
                                 cal_points=None,
                                 prep_params=dict(),
                                 upload=True):
    '''
    Performs X180 pulse on top of a fluxpulse

    Timings of sequence

       |          ----------           |X180|  ------------------------ |RO|
       |          ---    | --------- fluxpulse ---------- |
    '''

    seq_name = 'Fluxpulse_amplitude_sequence'
    ge_pulse = deepcopy(operation_dict['X180 ' + qb_name])
    ge_pulse['name'] = 'FPA_Pi'
    ge_pulse['element_name'] = 'FPA_Pi_el'

    flux_pulse = deepcopy(operation_dict[flux_pulse_name])
    flux_pulse['name'] = 'FPA_Flux'
    flux_pulse['ref_pulse'] = 'FPA_Pi'
    flux_pulse['ref_point'] = 'middle'

    if delay is None:
        delay = flux_pulse['pulse_length'] / 2

    flux_pulse['pulse_delay'] = -flux_pulse.get('buffer_length_start',
                                                0) - delay

    ro_pulse = deepcopy(operation_dict['RO ' + qb_name])
    ro_pulse['name'] = 'FPA_Ro'
    ro_pulse['ref_pulse'] = 'FPA_Pi'
    ro_pulse['ref_point'] = 'middle'


    ro_pulse['pulse_delay'] = flux_pulse['pulse_length'] - delay + \
                              flux_pulse.get('buffer_length_end', 0)

    pulses = [ge_pulse, flux_pulse, ro_pulse]
    swept_pulses = sweep_pulse_params(pulses,
                                      {'FPA_Flux.amplitude': amplitudes})

    swept_pulses_with_prep = \
        [add_preparation_pulses(p, operation_dict, [qb_name], **prep_params)
         for p in swept_pulses]

    seq = pulse_list_list_seq(swept_pulses_with_prep, seq_name, upload=False)

    if cal_points is not None:
        # add calibration segments
        seq.extend(cal_points.create_segments(operation_dict, **prep_params))

    log.debug(seq)
    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)

    return seq, np.arange(seq.n_acq_elements()), freqs


def ramsey_flux_pulse_seq(qb_name, times, operation_dict,
                          cz_pulse_name, cz_pulse_amp,
                          artificial_detunings=0,
                          cal_points=None,
                          upload=False, prep_params=dict()):
    '''
    Performs a Ramsey with interleaved Flux pulse
    Sequence
      | ----------  fluxpulse  ---------------  |
        |X90|  -------------------     |X90|  ---  |RO|
                                     sweep time
    '''

    seq_name = 'Ramsey_flux_seq'

    drag_length = operation_dict['X180 ' + qb_name]['sigma'] * operation_dict[
        'X180 ' + qb_name]['nr_sigma']

    flux_pulse = deepcopy(operation_dict[cz_pulse_name])
    flux_pulse['name'] = 'flux'
    flux_pulse['element_name'] = 'flux_el'
    flux_pulse['pulse_delay'] = - drag_length -flux_pulse.get(
        'buffer_length_start', 0)
    flux_pulse['amplitude'] = cz_pulse_amp
    flux_pulse['ref_point'] = 'start'
    flux_pulse['ref_pulse'] = 'Ramsey_x1'

    ramsey_ops = ["X90"] * 2
    ramsey_ops += ["RO"]
    ramsey_ops = add_suffix(ramsey_ops, " " + qb_name)

    # pulses
    ramsey_pulses = [deepcopy(operation_dict[op]) for op in ramsey_ops]
    ramsey_pulses[-1]['ref_pulse'] = 'flux'
    ramsey_pulses += [flux_pulse]

    # name and reference swept pulse
    ramsey_pulses[0]["name"] = f"Ramsey_x1"
    ramsey_pulses[1]["name"] = f"Ramsey_x2"
    ramsey_pulses[1]['ref_point'] = 'start'


    # compute dphase
    a_d = artificial_detunings if np.ndim(artificial_detunings) == 1 \
        else [artificial_detunings]
    dphase = [((t - times[0]) * a_d[i % len(a_d)] * 360) % 360
              for i, t in enumerate(times)]
    # sweep pulses
    params = {f'Ramsey_x2.pulse_delay': times}
    params.update({f'Ramsey_x2.phase': dphase})
    params.update({f'flux.pulse_length': times+3*drag_length})
    swept_pulses = sweep_pulse_params(ramsey_pulses, params)

    if cal_points is not None:
        # add cal points
        I = deepcopy(operation_dict['I ' + qb_name])
        I['name'] = 'Ical'
        X180 = deepcopy(operation_dict['X180 ' + qb_name])
        X180['name'] = 'X180cal'
        for i, cal_pulse in enumerate([I, X180]):
            cal_pulse['element_name'] = f'cal{i}'
            flux_pulse = deepcopy(operation_dict[cz_pulse_name])
            flux_pulse['name'] = 'cal_flux'
            flux_pulse['amplitude'] = cz_pulse_amp
            flux_pulse['pulse_length'] = 3*drag_length
            flux_pulse['ref_pulse'] = cal_pulse['name']
            flux_pulse['ref_point'] = 'middle'
            delay = flux_pulse['pulse_length'] / 2
            flux_pulse['pulse_delay'] = -flux_pulse.get('buffer_length_start',
                                                        0) - delay
            RO = deepcopy(operation_dict['RO ' + qb_name])
            RO['ref_pulse'] = 'cal_flux'
            swept_pulses += [[cal_pulse, RO, flux_pulse]]
            swept_pulses += [[cal_pulse, RO, flux_pulse]]

    swept_pulses_with_prep = \
        [add_preparation_pulses(p, operation_dict, [qb_name], **prep_params)
         for p in swept_pulses]

    seq = pulse_list_list_seq(swept_pulses_with_prep, seq_name, upload=False)

    # if cal_points is not None:
    #     # add calibration segments
    #     seq.extend(cal_points.create_segments(operation_dict, **prep_params))

    log.debug(seq)
    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)

    return seq, np.arange(seq.n_acq_elements())


def chevron_seqs(qbc_name, qbt_name, qbr_name, hard_sweep_dict, soft_sweep_dict,
                 operation_dict, cz_pulse_name, prep_params=dict(),
                 cal_points=None, upload=True):

    '''
    chevron sequence (sweep of the flux pulse length)

    Timings of sequence
                                  <-- length -->
    qb_control:    |X180|  ---   |  fluxpulse   |

    qb_target:     |X180|  --------------------------------------  |RO|

   '''

    seq_name = 'Chevron_sequence'

    ge_pulse_qbc = deepcopy(operation_dict['X180 ' + qbc_name])
    ge_pulse_qbc['name'] = 'chevron_pi_qbc'
    ge_pulse_qbt = deepcopy(operation_dict['X180s ' + qbt_name])
    ge_pulse_qbt['name'] = 'chevron_pi_qbt'
    for ge_pulse in [ge_pulse_qbc, ge_pulse_qbt]:
        ge_pulse['element_name'] = 'chevron_pi_el'

    flux_pulse = deepcopy(operation_dict[cz_pulse_name])
    flux_pulse['name'] = 'chevron_flux'
    flux_pulse['element_name'] = 'chevron_flux_el'

    ro_pulses = generate_mux_ro_pulse_list([qbc_name, qbt_name],
                                           operation_dict)
    if 'pulse_length' in hard_sweep_dict:
        max_flux_length = max(hard_sweep_dict['pulse_length']['values'])
        ro_pulses[0]['ref_pulse'] = 'chevron_pi_qbc'
        ro_pulses[0]['pulse_delay'] = \
            max_flux_length + flux_pulse.get('buffer_length_start', 0) + \
            flux_pulse.get('buffer_length_end', 0)

    ssl = len(list(soft_sweep_dict.values())[0]['values'])
    sequences = []
    for i in range(ssl):
        flux_p = deepcopy(flux_pulse)
        flux_p.update({k: v['values'][i] for k, v in soft_sweep_dict.items()})
        pulses = [ge_pulse_qbc, ge_pulse_qbt, flux_p] + ro_pulses
        swept_pulses = sweep_pulse_params(
            pulses, {f'chevron_flux.{k}': v['values']
                        for k, v in hard_sweep_dict.items()})
        swept_pulses_with_prep = \
            [add_preparation_pulses(p, operation_dict, [qbc_name, qbt_name],
                                    **prep_params)
             for p in swept_pulses]
        seq = pulse_list_list_seq(swept_pulses_with_prep,
                                  seq_name+f'_{i}', upload=False)
        if cal_points is not None:
            seq.extend(cal_points.create_segments(operation_dict,
                                                  **prep_params))
        sequences.append(seq)

    if upload:
        ps.Pulsar.get_instance().program_awgs(sequences[0])

    return sequences, np.arange(sequences[0].n_acq_elements()), np.arange(ssl)


def fluxpulse_scope_sequence(
              delays, freqs, qb_name, operation_dict, cz_pulse_name,
              cal_points=None, prep_params=dict(), upload=True):
    '''
    Performs X180 pulse on top of a fluxpulse

    Timings of sequence

       |          ----------           |X180|  ----------------------------  |RO|
       |        ---      | --------- fluxpulse ---------- |
                         <-  delay  ->
    '''
    seq_name = 'Fluxpulse_scope_sequence'
    ge_pulse = deepcopy(operation_dict['X180 ' + qb_name])
    ge_pulse['name'] = 'FPS_Pi'
    ge_pulse['element_name'] = 'FPS_Pi_el'

    flux_pulse = deepcopy(operation_dict[cz_pulse_name])
    flux_pulse['name'] = 'FPS_Flux'
    flux_pulse['ref_pulse'] = 'FPS_Pi'
    flux_pulse['ref_point'] = 'middle'
    flux_pulse['pulse_delay'] = -flux_pulse.get('buffer_length_start', 0)

    ro_pulse = deepcopy(operation_dict['RO ' + qb_name])
    ro_pulse['name'] = 'FPS_Ro'
    ro_pulse['ref_pulse'] = 'FPS_Pi'
    ro_pulse['ref_point'] = 'middle'
    ro_pulse['pulse_delay'] = flux_pulse['pulse_length'] - np.min(delays) + \
                              flux_pulse.get('buffer_length_end', 0)

    pulses = [ge_pulse, flux_pulse, ro_pulse]
    swept_pulses = sweep_pulse_params(pulses,
                                      {'FPS_Flux.pulse_delay': -delays})

    swept_pulses_with_prep = \
        [add_preparation_pulses(p, operation_dict, [qb_name], **prep_params)
         for p in swept_pulses]

    seq = pulse_list_list_seq(swept_pulses_with_prep, seq_name, upload=False)

    if cal_points is not None:
        # add calibration segments
        seq.extend(cal_points.create_segments(operation_dict, **prep_params))

    log.debug(seq)
    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)

    return seq, np.arange(seq.n_acq_elements()), freqs


def cz_bleed_through_phase_seq(phases, qb_name, CZ_pulse_name, CZ_separation,
                               operation_dict, oneCZ_msmt=False, nr_cz_gates=1,
                               verbose=False, upload=True, return_seq=False,
                               upload_all=True, cal_points=True):
    '''
    Performs a Ramsey-like with interleaved Flux pulse

    Timings of sequence
                           CZ_separation
    |CZ|-|CZ|- ... -|CZ| <---------------> |X90|-|CZ|-|X90|-|RO|

    Args:
        end_times: numpy array of delays after second CZ pulse
        qb: qubit object (must have the methods get_operation_dict(),
            get_drive_pars() etc.
        CZ_pulse_name: str of the form
            'CZ ' + qb_target.name + ' ' + qb_control.name
        X90_separation: float (separation of the two pi/2 pulses for Ramsey
        verbose: bool
        upload: bool
        return_seq: bool

    Returns:
        if return_seq:
          seq: qcodes sequence
          el_list: list of pulse elements
        else:
            seq_name: string
    '''
    # if maximum_CZ_separation is None:
    #     maximum_CZ_separation = CZ_separation

    seq_name = 'CZ Bleed Through phase sweep'
    seq = sequence.Sequence(seq_name)
    el_list = []

    X90_1 = deepcopy(operation_dict['X90 ' + qb_name])
    X90_2 = deepcopy(operation_dict['X90 ' + qb_name])
    RO_pars = deepcopy(operation_dict['RO ' + qb_name])
    CZ_pulse1 = deepcopy(operation_dict[CZ_pulse_name])
    CZ_pulse_len = CZ_pulse1['pulse_length']
    drag_pulse_len = deepcopy(X90_1['sigma']*X90_1['nr_sigma'])
    spacerpulse = {'pulse_type': 'SquarePulse',
                   'channel': X90_1['I_channel'],
                   'amplitude': 0.0,
                   'length': CZ_separation - drag_pulse_len,
                   'ref_point': 'end',
                   'pulse_delay': 0}
    if oneCZ_msmt:
        spacerpulse_X90 = {'pulse_type': 'SquarePulse',
                           'channel': X90_1['I_channel'],
                           'amplitude': 0.0,
                           'length': CZ_pulse1['buffer_length_start'] +
                                     CZ_pulse1['buffer_length_end'] +
                                     CZ_pulse_len,
                           'ref_point': 'end',
                           'pulse_delay': 0}
        main_pulse_list = [CZ_pulse1, spacerpulse,
                           X90_1, spacerpulse_X90]
    else:
        CZ_pulse2 = deepcopy(operation_dict[CZ_pulse_name])
        main_pulse_list = int(nr_cz_gates)*[CZ_pulse1]
        main_pulse_list += [spacerpulse, X90_1, CZ_pulse2]
    el_main = multi_pulse_elt(0, station,  main_pulse_list,
                              trigger=True, name='el_main')
    el_list.append(el_main)

    if upload_all:
        upload_AWGs = 'all'
        upload_channels = 'all'
        # else:
        # upload_AWGs = [station.pulsar.get(CZ_pulse1['channel'] + '_AWG')] + \
        #               [station.pulsar.get(ch + '_AWG') for ch in
        #                CZ_pulse1['aux_channels_dict']]
        # upload_channels = [CZ_pulse1['channel']] + \
        #                   list(CZ_pulse1['aux_channels_dict'])

    for i, theta in enumerate(phases):
        if cal_points and (theta == phases[-4] or theta == phases[-3]):
            el = multi_pulse_elt(i, station,
                                 [operation_dict['I ' + qb_name], RO_pars],
                                 name='el_{}'.format(i+1), trigger=True)
            el_list.append(el)
            seq.append('e_{}'.format(3*i), 'el_{}'.format(i+1),
                       trigger_wait=True)
        elif cal_points and (theta == phases[-2] or theta == phases[-1]):
            el = multi_pulse_elt(i, station,
                                 [operation_dict['X180 ' + qb_name], RO_pars],
                                 name='el_{}'.format(i+1), trigger=True)
            el_list.append(el)
            seq.append('e_{}'.format(3*i), 'el_{}'.format(i+1),
                       trigger_wait=True)
        else:
            X90_2['phase'] = theta*180/np.pi
            el = multi_pulse_elt(i+1, station, [X90_2, RO_pars], trigger=False,
                                 name='el_{}'.format(i+1),
                                 previous_element=el_main)
            el_list.append(el)

            seq.append('m_{}'.format(i), 'el_main',  trigger_wait=True)
            seq.append('e_{}'.format(3*i), 'el_{}'.format(i+1),
                       trigger_wait=False)

    if upload:
        station.pulsar.program_awgs(seq, *el_list,
                                    AWGs=upload_AWGs,
                                    channels=upload_channels,
                                    verbose=verbose)
    if return_seq:
        return seq, el_list
    else:
        return seq_name


def cphase_seqs(qbc_name, qbt_name, hard_sweep_dict, soft_sweep_dict,
                operation_dict, cz_pulse_name, num_cz_gates=1,
                max_flux_length=None, cal_points=None, upload=True,
                qbs_operations=None, prep_params=dict()):

    assert num_cz_gates % 2 != 0

    seq_name = 'cphase_sequence'

    if qbs_operations is None:
        qbs_operations = []
    # initial_rotations = [deepcopy(operation_dict['X180 ' + qbc_name]),
    #                     deepcopy(operation_dict['X90s ' + qbt_name])]
    initial_rotations = [deepcopy(operation_dict[op]) for
                         op in (['X180 ' + qbc_name, 'X90s ' + qbt_name] +
                                qbs_operations)]

    initial_rotations[0]['name'] = 'cphase_init_pi_qbc'
    initial_rotations[1]['name'] = 'cphase_init_pihalf_qbt'
    for rot_pulses in initial_rotations:
        rot_pulses['element_name'] = 'cphase_initial_rots_el'

    flux_pulse = deepcopy(operation_dict[cz_pulse_name])
    flux_pulse['name'] = 'cphase_flux'
    flux_pulse['element_name'] = 'cphase_flux_el'

    final_rotations = [deepcopy(operation_dict['X180 ' + qbc_name]),
                       deepcopy(operation_dict['X90s ' + qbt_name])]
    final_rotations[0]['name'] = 'cphase_final_pi_qbc'
    final_rotations[1]['name'] = 'cphase_final_pihalf_qbt'
    # final_rotations = [deepcopy(operation_dict['X90 ' + qbt_name])]
    # final_rotations[0]['name'] = 'cphase_final_pihalf_qbt'

    for rot_pulses in final_rotations:
        rot_pulses['element_name'] = 'cphase_final_rots_el'

    # set pulse delay of final_rotations[0] to max_flux_length
    if max_flux_length is None:
        if 'pulse_length' in soft_sweep_dict:
            max_flux_length = max(soft_sweep_dict['pulse_length']['values'])
            print(f'max_pulse_length = {max_flux_length*1e9:.2f} ns, '
                  f'from sweep points.')
        else:
            max_flux_length = flux_pulse['pulse_length']
            print(f'max_pulse_length = {max_flux_length*1e9:.2f} ns, '
                  f'from pulse dict.')
    # add buffers to this delay
    delay = (max_flux_length + flux_pulse.get('buffer_length_start', 0) +
        flux_pulse.get('buffer_length_end', 0))*num_cz_gates
    # # ensure the delay is commensurate with 16/2.4e9
    # comm_const = (16/2.4e9)
    # if delay % comm_const > 1e-15:
    #     delay = comm_const * (delay // comm_const + 1)
    #     print(f'delay adjusted to {delay*1e9:.2f} ns '
    #           f'to fulfill commensurability conditions with 16/2.4e9.')
    final_rotations[0]['ref_pulse'] = 'cphase_init_pi_qbc'
    final_rotations[0]['pulse_delay'] = delay

    ro_pulses = generate_mux_ro_pulse_list([qbc_name, qbt_name],
                                            operation_dict)

    # the phases in the hard_sweep_dict must be tiled by 2!
    hsl = len(list(hard_sweep_dict.values())[0]['values'])
    params = {'cphase_init_pi_qbc.amplitude': np.concatenate(
        [initial_rotations[0]['amplitude']*np.ones(hsl//2), np.zeros(hsl//2)]),
              'cphase_final_pi_qbc.amplitude': np.concatenate(
        [final_rotations[0]['amplitude']*np.ones(hsl//2), np.zeros(hsl//2)])}
    params.update({f'cphase_final_pihalf_qbt.{k}': v['values']
                   for k, v in hard_sweep_dict.items()})

    ssl = len(list(soft_sweep_dict.values())[0]['values'])
    sequences = []
    for i in range(ssl):
        fp_list = []
        flux_p = deepcopy(flux_pulse)
        flux_p.update({k: v['values'][i] for k, v in soft_sweep_dict.items()})
        for j in range(num_cz_gates):
            fp = deepcopy(flux_p)
            fp['name'] = f'cphase_flux_{j}'
            fp_list += [fp]
        pulses = initial_rotations + fp_list + final_rotations + ro_pulses
        swept_pulses = sweep_pulse_params(pulses, params)
        swept_pulses_with_prep = \
            [add_preparation_pulses(p, operation_dict, [qbc_name, qbt_name],
                                    **prep_params)
             for p in swept_pulses]
        seq = pulse_list_list_seq(swept_pulses_with_prep,
                                  seq_name+f'_{i}', upload=False)
        if cal_points is not None:
            seq.extend(cal_points.create_segments(operation_dict,
                                                  **prep_params))
        sequences.append(seq)

    # reuse sequencer memory by repeating readout pattern
    for s in sequences:
        s.repeat_ro(f"RO {qbc_name}", operation_dict)
        s.repeat_ro(f"RO {qbt_name}", operation_dict)

    if upload:
        ps.Pulsar.get_instance().program_awgs(sequences[0])

    return sequences, np.arange(sequences[0].n_acq_elements()), np.arange(ssl)


def cphase_fluxed_spectators_seqs(qbc_name, qbt_name, qbs_name,
                                  hard_sweep_dict, soft_sweep_dict,
                                  operation_dict, cz_pulse_name,
                                  num_cz_gates=1, qbs_operations=None,
                                  max_flux_length=None,
                                  cal_points=None, upload=True,
                                  prep_params=dict()):

    assert num_cz_gates % 2 != 0

    seq_name = 'cphase_spectator_seq'

    if qbs_operations is None:
        qbs_operations = []

    initial_rotations = [deepcopy(operation_dict[op]) for
                         op in (['X180 ' + qbc_name, 'X90s ' + qbt_name] +
                                qbs_operations)]

    initial_rotations[0]['name'] = 'cphase_init_pi_qbc'
    initial_rotations[1]['name'] = 'cphase_init_pihalf_qbt'
    for rot_pulses in initial_rotations:
        rot_pulses['element_name'] = 'cphase_initial_rots_el'

    cz_gate = deepcopy(operation_dict[cz_pulse_name])
    cz_gate['name'] = 'cphase_flux'
    cz_gate['element_name'] = 'cphase_flux_el'
    cz_gate['ref_pulse'] = 'cphase_init_pi_qbc'

    final_rotations = [deepcopy(operation_dict['X180 ' + qbc_name]),
                       deepcopy(operation_dict['X90s ' + qbt_name])]
    final_rotations[0]['name'] = 'cphase_final_pi_qbc'
    final_rotations[1]['name'] = 'cphase_final_pihalf_qbt'

    for rot_pulses in final_rotations:
        rot_pulses['element_name'] = 'cphase_final_rots_el'

    # set pulse delay of final_rotations[0] to max_flux_length
    if max_flux_length is None:
        max_flux_length = cz_gate['pulse_length']
        print(f'max_pulse_length = {max_flux_length*1e9:.2f} ns, '
              f'from pulse dict.')
    # add buffers to this delay
    delay = (max_flux_length + cz_gate.get('buffer_length_start', 0) +
        cz_gate.get('buffer_length_end', 0))*num_cz_gates
    final_rotations[0]['ref_pulse'] = 'cphase_init_pi_qbc'
    final_rotations[0]['pulse_delay'] = delay

    # drag_length = operation_dict['X180 ' + qbc_name]['sigma'] * operation_dict[
    #     'X180 ' + qbc_name]['nr_sigma']
    spectator_flux_pulse = deepcopy(operation_dict['FP ' + qbs_name])
    spectator_flux_pulse['name'] = 'spec_flux_pulse'
    spectator_flux_pulse['element_name'] = 'cphase_flux_el'
    delta_buffers = cz_gate.get('buffer_length_start', 0) + \
            cz_gate.get('buffer_length_end', 0) - (
            spectator_flux_pulse.get('buffer_length_start', 0) +
            spectator_flux_pulse.get('buffer_length_end', 0))
    spectator_flux_pulse['pulse_length'] = max_flux_length + delta_buffers
                                           # + 2*drag_length + \
                                           # 2*drag_length
    # assumption below: the flux pulse is referenced to the start of the
    # control qubit, not the spectator qubit -> assummed to be simultaneous
    # spectator_flux_pulse['ref_pulse'] = 'cphase_flux'
    # spectator_flux_pulse['ref_point'] = 'middle'
    # spectator_flux_pulse['ref_point_new'] = 'middle'
    # spectator_flux_pulse['pulse_delay'] = - drag_length

    ro_pulses = generate_mux_ro_pulse_list([qbc_name, qbt_name],
                                            operation_dict)
    # ro_pulses[0]['ref_pulse'] = 'spec_flux_pulse'

    # the phases in the hard_sweep_dict must be tiled by 2!
    hsl = len(list(hard_sweep_dict.values())[0]['values'])
    params = {'cphase_init_pi_qbc.amplitude': np.concatenate(
        [initial_rotations[0]['amplitude']*np.ones(hsl//2), np.zeros(hsl//2)]),
              'cphase_final_pi_qbc.amplitude': np.concatenate(
        [final_rotations[0]['amplitude']*np.ones(hsl//2), np.zeros(hsl//2)])}
    params.update({f'cphase_final_pihalf_qbt.{k}': v['values']
                   for k, v in hard_sweep_dict.items()})
    from pprint import pprint
    pprint(params)
    ssl = len(list(soft_sweep_dict.values())[0]['values'])
    sequences = []
    for i in range(ssl):
        flux_p = deepcopy(spectator_flux_pulse)
        flux_p.update({k: v['values'][i] for k, v in soft_sweep_dict.items()})
        cz_list = []
        for j in range(num_cz_gates):
            fp = deepcopy(cz_gate)
            fp['name'] = f'cphase_flux_{j}'
            cz_list += [fp]
        pulses = initial_rotations + [flux_p] + cz_list + final_rotations + \
                 ro_pulses
        swept_pulses = sweep_pulse_params(pulses, params)
        swept_pulses_with_prep = \
            [add_preparation_pulses(p, operation_dict, [qbc_name, qbt_name],
                                    **prep_params)
             for p in swept_pulses]
        seq = pulse_list_list_seq(swept_pulses_with_prep,
                                  seq_name+f'_{i}', upload=False)
        if cal_points is not None:
            seq.extend(cal_points.create_segments(operation_dict,
                                                  **prep_params))
        sequences.append(seq)

    # reuse sequencer memory by repeating readout pattern
    for s in sequences:
        s.repeat_ro(f"RO {qbc_name}", operation_dict)
        s.repeat_ro(f"RO {qbt_name}", operation_dict)

    if upload:
        ps.Pulsar.get_instance().program_awgs(sequences[0])

    return sequences, np.arange(sequences[0].n_acq_elements()), np.arange(ssl)


def cphase_interleaved_fluxed_spectators_seqs(qbc_name, qbt_name, qbs_names,
                                  hard_sweep_dict, soft_sweep_dict_list,
                                  operation_dict, cz_pulse_name,
                                  num_cz_gates=1, qbs_operations=None,
                                  max_flux_length=None,
                                  cal_points=None, upload=True,
                                  prep_params=dict()):

    # ATTENTION! The soft_sweep_dicts in soft_sweep_dict_list is assumed to
    # correspond to the qubits in the order given in qbs_names.

    assert num_cz_gates % 2 != 0

    seq_name = 'cphase_multi_spectators_seq'

    if qbs_operations is None:
        qbs_operations = []

    initial_rotations = [deepcopy(operation_dict[op]) for
                         op in (['X180 ' + qbc_name, 'X90s ' + qbt_name] +
                                qbs_operations)]

    initial_rotations[0]['name'] = 'cphase_init_pi_qbc'
    initial_rotations[1]['name'] = 'cphase_init_pihalf_qbt'
    for i in range(len(qbs_names)):
        initial_rotations[2+i]['name'] = f'spec_pulse_{qbs_names[i]}'
    for rot_pulses in initial_rotations:
        rot_pulses['element_name'] = 'cphase_initial_rots_el'

    cz_gate = deepcopy(operation_dict[cz_pulse_name])
    cz_gate['name'] = 'cphase_flux'
    cz_gate['element_name'] = 'cphase_flux_el'
    cz_gate['ref_pulse'] = 'cphase_init_pi_qbc'

    final_rotations = [deepcopy(operation_dict['X180 ' + qbc_name]),
                       deepcopy(operation_dict['X90s ' + qbt_name])]
    final_rotations[0]['name'] = 'cphase_final_pi_qbc'
    final_rotations[1]['name'] = 'cphase_final_pihalf_qbt'

    for rot_pulses in final_rotations:
        rot_pulses['element_name'] = 'cphase_final_rots_el'

    # set pulse delay of final_rotations[0] to max_flux_length
    if max_flux_length is None:
        max_flux_length = cz_gate['pulse_length']
        print(f'max_pulse_length = {max_flux_length*1e9:.2f} ns, '
              f'from pulse dict.')
    # add buffers to this delay
    delay = (max_flux_length + cz_gate.get('buffer_length_start', 0) +
        cz_gate.get('buffer_length_end', 0))*num_cz_gates
    final_rotations[0]['ref_pulse'] = 'cphase_init_pi_qbc'
    final_rotations[0]['pulse_delay'] = delay

    ro_pulses = generate_mux_ro_pulse_list([qbc_name, qbt_name],
                                            operation_dict)

    # the phases in the hard_sweep_dict must be tiled by 4!
    hsl = len(list(hard_sweep_dict.values())[0]['values'])
    print(hsl)
    params = {'cphase_init_pi_qbc.amplitude': np.tile(np.concatenate(
        [initial_rotations[0]['amplitude']*np.ones(hsl//4), np.zeros(hsl//4)]), 2),
              'cphase_final_pi_qbc.amplitude': np.tile(np.concatenate(
        [final_rotations[0]['amplitude']*np.ones(hsl//4), np.zeros(hsl//4)]), 2)}
    params.update({f'spec_pulse_{qbs_names[i]}.amplitude': np.repeat(np.concatenate(
        [initial_rotations[i+2]['amplitude']*np.ones(hsl//4), np.zeros(
            hsl//4)]), 2) for i in range(len(qbs_names))})
    params.update({f'cphase_final_pihalf_qbt.{k}': v['values']
                   for k, v in hard_sweep_dict.items()})
    # params = {'cphase_init_pi_qbc.amplitude': np.concatenate(
    #     [initial_rotations[0]['amplitude']*np.ones(hsl//2), np.zeros(hsl//2)]),
    #           'cphase_final_pi_qbc.amplitude': np.concatenate(
    #     [final_rotations[0]['amplitude']*np.ones(hsl//2), np.zeros(hsl//2)])}
    # params.update({f'cphase_final_pihalf_qbt.{k}': v['values']
    #                for k, v in hard_sweep_dict.items()})

    ssl = len(list(soft_sweep_dict_list[0].values())[0]['values'])
    print(ssl)
    from pprint import pprint
    pprint(params)
    sequences = []
    for i in range(ssl):
        # add the spectator flux pulses
        fp_lst = []
        cnt = 0
        for p, qbsn in enumerate(qbs_names):
            spec_fp = deepcopy(operation_dict['FP ' + qbsn])
            if len(spec_fp['channel']) > 0:
                spec_fp['name'] = f'spec_flux_pulse_{p}'
                spec_fp['element_name'] = 'cphase_flux_el'
                delta_buffers = cz_gate.get('buffer_length_start', 0) + \
                                cz_gate.get('buffer_length_end', 0) - (
                                        spec_fp.get('buffer_length_start', 0) +
                                        spec_fp.get('buffer_length_end', 0))
                spec_fp['pulse_length'] = max_flux_length + delta_buffers
                if cnt != 0:
                    spec_fp['ref_point'] = 'start'
                spec_fp.update({k: v['values'][i] for k, v in
                                soft_sweep_dict_list[p].items()})
                fp_lst += [spec_fp]
                cnt += 1
            else:
                print(f'{qbsn} has no flux pulse channel. Not adding the flux'
                      f'pulse for this qubit.')
                print()

        # add the num_cz_gates CZ gates
        cz_list = []
        for j in range(num_cz_gates):
            cz = deepcopy(cz_gate)
            cz['name'] = f'cphase_flux_{j}'
            cz_list += [cz]
        # put the pulses in each segment
        pulses = initial_rotations + fp_lst + cz_list + final_rotations + \
                 ro_pulses
        swept_pulses = sweep_pulse_params(pulses, params)
        swept_pulses_with_prep = \
            [add_preparation_pulses(p, operation_dict, [qbc_name, qbt_name],
                                    **prep_params)
             for p in swept_pulses]

        seq = pulse_list_list_seq(swept_pulses_with_prep,
                                  seq_name+f'_{i}', upload=False)
        if cal_points is not None:
            seq.extend(cal_points.create_segments(operation_dict,
                                                  **prep_params))
        sequences.append(seq)

    # reuse sequencer memory by repeating readout pattern
    for s in sequences:
        s.repeat_ro(f"RO {qbc_name}", operation_dict)
        s.repeat_ro(f"RO {qbt_name}", operation_dict)

    if upload:
        ps.Pulsar.get_instance().program_awgs(sequences[0])

    return sequences, np.arange(sequences[0].n_acq_elements()), np.arange(ssl)


def add_suffix(operation_list, suffix):
    return [op + suffix for op in operation_list]