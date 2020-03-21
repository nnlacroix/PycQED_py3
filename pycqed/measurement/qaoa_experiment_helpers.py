import numpy as np
from numpy import array
import lmfit
import matplotlib.pyplot as plt
import pycqed.analysis.measurement_analysis as ma
from pycqed.analysis import analysis_toolbox as a_tools
from scipy.interpolate import interp1d

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


## cphase functions ##
def ascending_phase_order(cphases_calib):
    cphases_calib[cphases_calib < 0] += 2 * np.pi
    cphases_calib[cphases_calib > 2 * np.pi] -= 2 * np.pi
    pi_phase_idx = np.argmin(np.abs(cphases_calib - np.pi))
    for ind in range(len(cphases_calib)):
        if ind < pi_phase_idx and cphases_calib[ind] > cphases_calib[pi_phase_idx]:
            cphases_calib[ind] -= 2 * np.pi
        elif ind > pi_phase_idx and cphases_calib[ind] < cphases_calib[
            pi_phase_idx]:
            cphases_calib[ind] += 2 * np.pi
    return cphases_calib


## dynamic phase functions ##
def compute_dyn_phase(f_park, f, tend, tstart=0, fargs=()):
    # f is the function giving the frequency of qbc at time t
    from scipy.integrate import quad as integrate

    return -(integrate(lambda t: f(*fargs) - f_park, 0, tend)[
                 0] * 2 * np.pi * 180 / np.pi) % 360


def calc_qubit_freq_during_fluxpulse(qbc_freq_sweetspot, qbc_anharmonicity,
                                     amplitudes, dphi_dv,
                                     ej_correction_factor=1.00,
                                     junction_asym=0.485):
    Ej_max = calc_Ej_from_ge_freq(qbc_freq_sweetspot, -qbc_anharmonicity)
    ge_freq_qbc = calc_ge_freq(voltage=amplitudes, dphi_dv=dphi_dv,
                               junction_asym=junction_asym, Ec=-qbc_anharmonicity,
                               Ej_max=Ej_max,
                               ej_correction_factor=ej_correction_factor,
                               flux_upwards=False)
    return ge_freq_qbc


def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def find_ampl_spacing_for_exp(dyn_model, dyn_model_amplitudes, amin=0, amax=0.35,
                              phase_sep=160,
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


#################### reloading parameter functions and build arb phase dict###
def get_calib_dict(name=None):
    import os
    l = []
    for file in os.listdir("."):
        if file.startswith("carb_calib_ts_dict"):
            print(os.path.join(file))
            l.append(file)
    if name is None and len(l)> 0:
        name = l[-1]
    print(f'Loading {name}')
    return np.load(name)[0]
def save_calib_dict(calib_dict, name=None):
    import datetime
    if name is None:
        name = "carb_calib_ts_dict_{:%Y%m%d_%H%M%S}.npy".format(datetime.datetime.now())
    np.save(name, [calib_dict])

def load_dyn_phase(timestamp, qb_names, plot=True):
    f = a_tools.get_folder(timestamp=timestamp)
    amplitudes = np.load(f + "\\amplitudes.npy")
    dph_all = {}
    if isinstance(qb_names, str):
        qb_names = [qb_names]
    for qbn in qb_names:
        dph = np.load(f + f"\\dynamic_phases_{qbn}.npy")
        if plot:
            plt.scatter(amplitudes, np.unwrap(dph / 180 * np.pi) * 180 / np.pi,
                     label=f"{qbn} measured")
            plt.xlabel("Amplitude")
            plt.ylabel("Dynamic phase (deg.)")
        dyn_phase_func = lambda ampl_test: np.interp(ampl_test, amplitudes,
                                                     np.unwrap(
                                                         dph / 180 * np.pi) * 180 / np.pi,
                                                     period=360)
        #         dyn_phase_func_str = f"lambda ampl_test: np.interp(ampl_test, {repr(amplitudes)}, {repr( np.unwrap(dph/180*np.pi)*180/np.pi)}, period=360)"
        dyn_phase_func_str = f"lambda ampl_test: interp1d({repr(amplitudes)}, " \
                            f"{repr(np.unwrap(dph / 180 * np.pi) * 180 /np.pi)}, kind='cubic', fill_value='extrapolate' )(ampl_test)"
        dph_all[qbn] = dyn_phase_func_str
    return dph_all


def load_cphase(timestamp, offset=None, remove=None):
    a = ma.MeasurementAnalysis(timestamp=timestamp, auto=False)
    a.get_naming_and_values()
    ampl_calib = a.exp_metadata['soft_sweep_params']['amplitude']['values']
    cphases_calib = np.array(
        a.data_file['Analysis']["Processed data"]['analysis_params_dict'][
            "cphase"]['val'])

    # remove bad points
    mask = np.ones_like(ampl_calib, dtype=bool)
    if remove is not None:
        mask[remove] = [False] * len(remove)
        ampl_calib = ampl_calib[mask]
        cphases_calib = cphases_calib[mask]

    # clean discontinuities
    cphases_calib[cphases_calib < 0] = cphases_calib[cphases_calib < 0] + 2 * np.pi
    cphases_calib[cphases_calib > 2 * np.pi] = cphases_calib[
                                                   cphases_calib > 2 * np.pi] - 2 * np.pi

    # take first point as lowest point
    cphases_calib[cphases_calib < cphases_calib[0]] = \
        cphases_calib[cphases_calib  < cphases_calib[0]] + 2 * np.pi
    pi_phase_idx = np.argmin(np.abs(cphases_calib - np.pi))
    for ind in range(len(cphases_calib)):
        if ind < pi_phase_idx and cphases_calib[ind] > cphases_calib[pi_phase_idx]:
            cphases_calib[ind] -= 2 * np.pi
        elif ind > pi_phase_idx and cphases_calib[ind] < cphases_calib[
            pi_phase_idx]:
            cphases_calib[ind] += 2 * np.pi
    if offset is None:
        offset = cphases_calib[0]
    return ampl_calib, cphases_calib, offset

def create_ampl_to_phase_func(ampl_calib, cphases_calib, offset):
    arb_phase_amp_func = lambda target_phase: np.interp((target_phase - offset) % (2 * np.pi), cphases_calib - offset, ampl_calib)
    arb_phase_amp_func_str = f"lambda target_phase: np.interp((target_phase - {offset}) % (2*np.pi), {repr( cphases_calib - offset)}, {repr(ampl_calib)}, period=2*np.pi)"
    return arb_phase_amp_func_str

def load_amplitude_func(timestamp, offset=None, remove=None):
    a, ph, off = load_cphase(timestamp, offset=offset, remove=remove)
    return create_ampl_to_phase_func(a, ph, off)

def create_arb_phase_func(amplitude_function, dyn_phase_functions):
    arb_phase_func_str = f"lambda target_phase: (({amplitude_function})(target_phase)," + "{"
    for qbn, func in dyn_phase_functions.items():
        arb_phase_func_str += f"'{qbn}': ({func})(({amplitude_function})(target_phase)),"
    arb_phase_func_str += "})"
    return arb_phase_func_str

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


###############################################################################
#           qaoa related
###############################################################################

def initial_simplex(nreps, depth=1, dev=0.2, gs=None, bs=None, range_g=(0,np.pi), range_b=(0,2*np.pi), seed=2):
    simplexes = []
    np.random.seed(seed)
    if gs is None:
        gs = np.random.uniform(range_g[0], range_g[1], (nreps, depth))
    if bs is None:
        bs = np.random.uniform(range_b[0], range_b[1], (nreps, depth))
    for i in range(nreps):
        s = []
        base = np.concatenate([gs[i], bs[i]])
        s.append(base)
        for j in range(0, 2*depth):
            row = [b + dev if j == k else b for k, b in enumerate(base)]
            s.append(row)
        simplexes.append(np.asarray(s))
    return gs, bs, simplexes