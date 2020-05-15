from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.analysis_v2 import timedomain_analysis as tda
import matplotlib.pyplot as plt
import numpy as np
import lmfit
import sys
from copy import deepcopy

#############################################################################
#                   C-ARB related
#############################################################################

################# Chevron functions #####################
def effective_time(J_11_20, delta_20_11, n=1, time=None):
    if time is None:
        time = n / (2 * J_11_20)
    return 2 * np.pi * 2 * J_11_20 * time / np.sqrt(
        4 * (2 * np.pi * J_11_20) ** 2 + (2 * np.pi * delta_20_11) ** 2)


def calc_delta_20_11(ge_freq_qbt, ge_freq_qbc, anharmonicity_qbt):
    return ge_freq_qbt - anharmonicity_qbt - ge_freq_qbc


def calc_junction_asym(alpha=1 / 8):
    return (1 - alpha) / (1 + alpha)


def calc_ge_freq(voltage, dphi_dv, junction_asym, Ec, Ej_max, ej_correction_factor,
                 flux_upwards=False):
    # correction factor is to correct for approximation in formula
    Ej = calc_Ej_from_voltage(voltage, dphi_dv, junction_asym, Ej_max,
                              flux_upwards=flux_upwards)
    return ej_correction_factor * np.sqrt(8 * Ec * Ej) - Ec


def calc_Ej_from_voltage(voltage, dphi_dv, junction_asym, Ej_max,
                         flux_upwards=False):
    offset = 0
    if flux_upwards:
        offset = np.pi / 2
    return Ej_max * np.sqrt(
        np.cos(dphi_dv * voltage + offset) ** 2 + junction_asym ** 2 * np.sin(
            dphi_dv * voltage + offset) ** 2)


def calc_Ej_from_ge_freq(ge_freq, Ec):
    return ((ge_freq + Ec) ** 2) / (8 * Ec)


def calc_tau_effective(voltages, qbc_freq_sweetspot, qbc_anharmonicity,
                       qbt_ge_freq, qbt_anharmonicity, dphi_dv, J_00_10,
                       junction_asym=None,
                       n=1, time=None, ej_correction_factor=1, flux_upwards=False):
    if junction_asym is None:
        junction_asym = calc_junction_asym()
    Ej_max = calc_Ej_from_ge_freq(qbc_freq_sweetspot, -qbc_anharmonicity)
    ge_freq_qbc = calc_ge_freq(voltage=voltages, dphi_dv=dphi_dv,
                               junction_asym=junction_asym, Ec=-qbc_anharmonicity,
                               Ej_max=Ej_max,
                               ej_correction_factor=ej_correction_factor,
                               flux_upwards=flux_upwards)
    if flux_upwards:
        delta = calc_delta_20_11(qbt_ge_freq, ge_freq_qbc, -qbt_anharmonicity)
    else:
        delta = calc_delta_20_11(ge_freq_qbc, qbt_ge_freq, -qbc_anharmonicity)
    return effective_time(np.sqrt(2) * J_00_10, delta, n=n, time=time)

cos = lambda t, a, b, c, d: a * np.cos(b * t + c) + d

def cos_fit(t, pe, init_guess_cos=(0.5, 20e7, 0, 0.5)):
    cos = lambda t, a, b, c, d: a * np.cos(b * t + c) + d
    model = lmfit.Model(cos)
    model.set_param_hint('a', value=init_guess_cos[0], min=1e-2, max=1)
    model.set_param_hint('b', value=init_guess_cos[1], min=5e6, max=1e9)
    model.set_param_hint('c', value=init_guess_cos[2], min=-2 * np.pi,
                         max=2 * np.pi)
    model.set_param_hint('d', value=init_guess_cos[3], min=0, max=1)

    results = model.fit(data=pe, t=t)
    # print(results.best_values.values())
    return results.best_values.values()


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def find_t_max_voltage(t, pe, init_guess_max=None, n=0, tol=0.1,
                       init_guess_cos=None,
                       filter=None, fargs=(), at_timestep=None):
    """
    finds time for which population in state e (pe) is maximized.
    Returns closest to guess value if specified else the nth maximum of the cosine fit.
    """
    a, b, c, d = cos_fit(t, pe, init_guess_cos=init_guess_cos)
    if at_timestep is not None:
        return (at_timestep - c) / b
    c = c % (2 * np.pi)
    print(a, b, c, d)

    maxs = [(np.pi * n - c) / b for n in range(100)]
    # filter if maxima outside of window
    if filter is None:
        maxs = np.array([m for m in maxs if t[0] <= m <= t[-1]])
    else:
        maxs = filter(maxs, *fargs)
    dt = 1e-11  # t[1] - t[0]
    deriv = np.diff([cos(t[0] - dt, a, b, c, d), cos(t[0], a, b, c, d)])
    print("Derivative at t0", deriv)
    print("Detected Extremum ", maxs)
    if deriv > 0 and len(maxs) > 1:
        maxs = maxs[::2]
    elif len(maxs) > 1:
        maxs = maxs[1::2]
    #     maxs = [m for m in maxs if np.abs(np.max(pe)-cos(m, a,b,c,d)) <= tol*(np.max(pe)-np.min(pe))]
    if init_guess_max is None:
        return maxs[n]  # np.min(t[np.argsort(cos(t, a, b,c,d))[-3:]]) # maxs[n]
    else:
        return find_nearest(np.array(maxs), init_guess_max)


def moving_average(x, w=2):
    return np.convolve(x, np.ones(w), 'valid') / w


## dynamic phase functions ##
def compute_dyn_phase(f_park, f, tend, tstart=0, fargs=()):
    # f is the function giving the frequency of qbc at time t
    from scipy.integrate import quad as integrate
    return -(integrate(lambda t: f(*fargs) - f_park, 0, tend)[
                 0] * 2 * np.pi * 180 / np.pi) % 360


def calc_qubit_freq_during_fluxpulse(qbc_freq_sweetspot, qbc_anharmonicity,
                                     amplitudes, dphi_dv,
                                     ej_correction_factor=1.00,
                                     junction_asym=0.485, flux_upwards=False):
    Ej_max = calc_Ej_from_ge_freq(qbc_freq_sweetspot, -qbc_anharmonicity)
    ge_freq_qbc = calc_ge_freq(voltage=amplitudes, dphi_dv=dphi_dv,
                               junction_asym=junction_asym, Ec=-qbc_anharmonicity,
                               Ej_max=Ej_max,
                               ej_correction_factor=ej_correction_factor,
                               flux_upwards=flux_upwards)
    return ge_freq_qbc


def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def find_ampl_spacing_for_exp(dyn_model, dyn_model_amplitudes, amin=0, amax=0.35,
                              phase_sep=100,
                              max_spacing=0.0005, search_window_size=0.0001,
                              plot=False):
    """
    Generates the array of amplitudes at which the dynamic phase should be evaluated in the experiment,
    while ensuring the spacing between two phases will be equal to phase_sep (based on model dynamic phase and ampl)
    Args:
        amin (float): amplitude at which experiment should begin
        amax (float): amplitude at which experiment should end
        dyn_model (array): unwraped dynamic phase estimated from model (radian)
        dyn_model_amplitudes (array): array corresponding to the amplitudes at which dyn_model is evaluated
        phase_sep (float, array): max phase separation between two points (deg), can also be array shaped like dyn_model_amplitudes
        max_spacing (float): maximal spacing between two values (trucates
        gradient) or list of tuples [(low_ampl, high_ampl, value of max spacing)]
        search_window_size (float):  will search the max spacing for each amplitude in window: (a-search_window_size, a + search_window_size)
            to avoid undersampling around a if there is a narrow peak (due to noise) around a.
    """
    deriv_dphase_dV = np.abs(
        np.gradient(dyn_model * 180 / np.pi, np.diff(dyn_model_amplitudes)[0]))
    max_ampl_spacing = phase_sep / deriv_dphase_dV
    if np.ndim(max_spacing) == 0:
        max_ampl_spacing = np.minimum(phase_sep / deriv_dphase_dV,
                                      np.ones_like(deriv_dphase_dV) * max_spacing)
    else:
        for (low, high, val) in max_spacing:
            idx_low = find_nearest_idx(dyn_model_amplitudes, low)
            idx_high = find_nearest_idx(dyn_model_amplitudes, high)
            print(idx_low, idx_high)
            max_ampl_spacing[idx_low: idx_high] = np.minimum(max_ampl_spacing[
                                                             idx_low: idx_high],
                                                             np.ones(idx_high -
                                                                     idx_low)* val)

    if plot:
        plt.plot(dyn_model_amplitudes, max_ampl_spacing)
        plt.yscale('log')
    ampl = [amin]
    while (ampl[-1] < amax):
        search_window_min = find_nearest_idx(dyn_model_amplitudes, ampl[-1])
        search_window_max = find_nearest_idx(dyn_model_amplitudes,
                                             ampl[-1] + max_ampl_spacing[
                                                 search_window_min])
                # search_window_min =  find_nearest(dyn_model_amplitudes, ampl[-1] - search_window_size)
                # search_window_max = find_nearest(dyn_model_amplitudes, ampl[-1] + search_window_size)
        try:
            min_val = np.min(max_ampl_spacing[search_window_min:search_window_max])
            if np.ndim(max_spacing) != 0:
                for (low, high, val) in max_spacing:
                    if low < ampl[-1] < high:
                        min_val = min(val, min_val)
                search_window_min = find_nearest_idx(dyn_model_amplitudes,
                                                     ampl[-1] - search_window_size)
                search_window_max = find_nearest_idx(dyn_model_amplitudes, ampl[-1]
                                                     + search_window_size)
                # print(search_window_min, search_window_max)
                min_val = min(min_val, np.min(
                    max_ampl_spacing[search_window_min:search_window_max]))
        except Exception as e:
            min_val = 0.001
            if np.ndim(max_spacing) == 0:
                min_val = max_spacing
            else:
                min_val = 1
                for (low, high, val) in max_spacing:
                    if low < ampl[-1] < high:
                        min_val = min(val, min_val)
                search_window_min =  find_nearest_idx(dyn_model_amplitudes,
                                                   ampl[-1] - search_window_size)
                search_window_max = find_nearest_idx(dyn_model_amplitudes, ampl[-1]
                                                  + search_window_size)
                # print(search_window_min, search_window_max)
                min_val = min(min_val, np.min(
                    max_ampl_spacing[search_window_min:search_window_max]))
        ampl.append(ampl[-1] + min_val)
    return np.asarray(ampl)

# get amplitudes to measure
def get_amplitudes_to_measure(qbc, qbt, amplitude_range, cz_pulse_name,
                              chevron_params_dict, **kw):
    ampl_theory = np.linspace(amplitude_range[0], amplitude_range[-1], 5000)  # take amplitude range of interest: same as cphi
    chevron_params = deepcopy(chevron_params_dict['default'])
    gate_dict = qbc.get_operation_dict()[cz_pulse_name]
    pulse_class = getattr(sys.modules['pycqed.measurement.waveform_control.pulse_library'],
                          gate_dict['pulse_type'])  # get correct pulse class from type

    if f'{qbt.name}{qbc.name}' in chevron_params_dict:
        chevron_params.update(chevron_params_dict[f'{qbt.name}{qbc.name}'])
    dyn_phases_theory = []
    for amp in ampl_theory:
        pulse_length = pulse_class.calc_physical_length(amp, gate_dict['cphase_calib_dict'])
        dph = compute_dyn_phase(qbc.ge_freq(), calc_qubit_freq_during_fluxpulse,
                                   tend=pulse_length,
                                   fargs=(qbc.ge_freq(), qbc.anharmonicity(), amp,
                                          chevron_params['dphi_dv'], chevron_params['ej_correction_factor'],
                                          chevron_params['d']))
        dyn_phases_theory.append(dph)
    dyn_phases_theory = np.array(dyn_phases_theory)
    ampl_to_measure = find_ampl_spacing_for_exp(np.unwrap(dyn_phases_theory * np.pi / 180),
                                                   ampl_theory, ampl_theory[0], ampl_theory[-1],
                                                   phase_sep= kw.get('phase_sep', 80),
                                                   search_window_size=kw.get('search_window_size', 0.0005),
                                                   max_spacing=kw.get('max_ampl_spacing', 0.0008))
    return ampl_to_measure


#############################################################################
#                   CZ related
#############################################################################

def update_cz_amplitude(qbc, qbt, phases, amplitudes, target_phase=np.pi,
                        update=True):
    print(f"old amplitude: {qbc.get('upCZ_{}_amplitude'.format(qbt.name))}")
    print(f"amplitudes: {amplitudes}")
    phases %= 2*np.pi
    print(f"phases: {phases}")
    fit_res = lmfit.Model(lambda x, m, b: m*np.tan(x/2-np.pi/2) + b).fit(
        x=phases, data=amplitudes, m=1, b=np.mean(amplitudes))
    new_ampl = fit_res.model.func(target_phase, **fit_res.best_values)
    print('BEST {} '.format('amplitude'), new_ampl)
    if update:
        qbc.set('upCZ_{}_amplitude'.format(qbt.name), new_ampl)


def get_optimal_amp(qbc, qbt, soft_sweep_points, timestamp=None,
                    classified_ro=False, tangent_fit=False):

    if classified_ro:
        channel_map = {qb.name: [vn + ' ' +
                                 qb.instr_uhf() for vn in
                                 qb.int_avg_classif_det.value_names]
                       for qb in [qbc, qbt]}
    else:
        channel_map = {qb.name: [vn + ' ' +
                                 qb.instr_uhf() for vn in
                                 qb.int_avg_det.value_names]
                       for qb in [qbc, qbt]}
    tdma = tda.CPhaseLeakageAnalysis(
        t_start=timestamp,
        qb_names=[qbc.name, qbt.name],
        options_dict={'TwoD': True, 'plot_all_traces': False,
                      'plot_all_probs': False,
                      'channel_map': channel_map})
    cphases = tdma.proc_data_dict[
        'analysis_params_dict']['cphase']['val']

    sweep_pts = list(soft_sweep_points.values())[0]['values']
    if tangent_fit:
        fit_res = lmfit.Model(lambda x, m, b: m*np.tan(x/2-np.pi/2) + b).fit(
            x=cphases, data=sweep_pts,
            m=(max(sweep_pts)-min(sweep_pts))/((max(cphases)-min(cphases))),
            b=np.min(sweep_pts))
    else:
        fit_res = lmfit.Model(lambda x, m, b: m*x + b).fit(
            x=cphases, data=sweep_pts,
            m=(max(sweep_pts)-min(sweep_pts))/((max(cphases)-min(cphases))),
            b=np.min(sweep_pts))
    plot_and_save_cz_amp_sweep(cphases=cphases, timestamp=timestamp,
                               soft_sweep_params_dict=soft_sweep_points,
                               fit_res=fit_res, save_fig=True, plot_guess=False,
                               qbc_name=qbc.name, qbt_name=qbt.name)
    return fit_res


def plot_and_save_cz_amp_sweep(cphases, soft_sweep_params_dict, fit_res,
                               qbc_name, qbt_name, save_fig=True, show=True,
                               plot_guess=False, timestamp=None):

    sweep_param_name = list(soft_sweep_params_dict)[0]
    sweep_points = soft_sweep_params_dict[sweep_param_name]['values']
    unit = soft_sweep_params_dict[sweep_param_name]['unit']
    best_val = fit_res.model.func(np.pi, **fit_res.best_values)
    fit_points_init = fit_res.model.func(cphases, **fit_res.init_values)
    fit_points = fit_res.model.func(cphases, **fit_res.best_values)

    fig, ax = plt.subplots()
    ax.plot(cphases*180/np.pi, sweep_points, 'o-')
    ax.plot(cphases*180/np.pi, fit_points, '-r')
    if plot_guess:
        ax.plot(cphases*180/np.pi, fit_points_init, '--k')
    ax.hlines(best_val, cphases[0]*180/np.pi, cphases[-1]*180/np.pi)
    ax.vlines(180, sweep_points.min(), sweep_points.max())
    ax.set_ylabel('Flux pulse {} ({})'.format(sweep_param_name, unit))
    ax.set_xlabel('Conditional phase (rad)')
    ax.set_title('CZ {}-{}'.format(qbc_name, qbt_name))

    ax.text(0.5, 0.95, 'Best {} = {:.6f} ({})'.format(
        sweep_param_name, best_val*1e9 if unit=='s' else best_val, unit),
            horizontalalignment='center', verticalalignment='top',
            transform=ax.transAxes)
    if save_fig:
        import datetime
        import os
        fig_title = 'CPhase_amp_sweep_{}_{}'.format(qbc_name, qbt_name)
        fig_title = '{}--{:%Y%m%d_%H%M%S}'.format(
            fig_title, datetime.datetime.now())
        if timestamp is None:
            save_folder = a_tools.latest_data()
        else:
            save_folder = a_tools.get_folder(timestamp)
        filename = os.path.abspath(os.path.join(save_folder, fig_title+'.png'))
        fig.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()
