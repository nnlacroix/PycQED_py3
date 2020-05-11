import numpy as np
from numpy import array
import lmfit
import matplotlib.pyplot as plt
import pycqed.analysis.measurement_analysis as ma
from pycqed.analysis import analysis_toolbox as a_tools
from scipy.interpolate import interp1d
from qutip import QubitCircuit
import sys
import datetime as dt
import time
from pycqed.measurement import hdf5_data as h5d
from pycqed.measurement import qaoa as qaoa
from pycqed.utilities import general as gen
from pycqed.analysis_v2 import timedomain_analysis as tda
from copy import deepcopy
import pycqed.measurement.waveform_control.pulse_library as pl

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
        print(ge_freq_qbc)
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
    return np.load(name, allow_pickle=True)[0]
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
                     label=f"{qbn} measured", marker=".")
            plt.xlabel("Amplitude")
            plt.ylabel("Dynamic phase (deg.)")
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



def qaoa_landscape_rowwise(qbs_qaoa, gates_info, nb=45, ng=45,
                           cphase_implementation="software", init_state='+',
                           analyze=True, analysis_kwargs=None, start_from=0,
                           max_b=np.pi / 2, max_g=np.pi, shots=15000):
    reload(mqm)
    reload(qaoa)
    b = np.linspace(0, max_b, nb)
    g = np.linspace(0, max_g, ng)
    print(b)
    print(g)
    gg, bb = np.meshgrid(g, b, indexing='ij')
    err = []

    for j in range(nb):
        if j < start_from:
            continue
        label = f"QAOA_landscape_{j}"
        print(dt.datetime.now(), label)
        try:
            mqm.measure_qaoa(qbs_qaoa,
                             gates_info,
                             betas=np.expand_dims(bb[:, j], -1),
                             gammas=np.expand_dims(gg[:, j], -1),
                             init_state=init_state, label=label,
                             cphase_implementation=cphase_implementation,
                             shots=shots,
                             analyze=False)

        except Exception as e:
            print(e)
            raise e
            err.append((i, j))

    # analysis
    # get timestamps
    tps, mmnt = ba.BaseDataAnalysis.get_latest_data(ng)
    # filter by measurement name in case the measurement crashed
    # and not all rows were performed and reorder to have
    # first measurement first in list
    tps = list(reversed([t for t, m in zip(tps, mmnt) if 'QAOA_landscape' in m]))
    if analyze:
        if analysis_kwargs is None:
            analysis_kwargs = {}
        return analyze_qaoa_landscape_rowwise(tps, **analysis_kwargs)
    return tps


###############################################################################
#           QAOA analysis and plotting
###############################################################################

def load_landscape_from_npy(timestamp):
    a = ma.MeasurementAnalysis(timestamp=timestamp, auto=False)
    gg = np.load(a.folder + "\\gammas.npy")
    bb = np.load(a.folder + "\\betas.npy")
    energy = np.load(a.folder + "\\energy.npy")
    a.data_file.close()
    # TODO: add leakage
    return gg, bb, energy, None, a


def load_landscape_from_analyzed_files(tstart, tend, save=False, plot=True):
    tps = a_tools.get_timestamps_in_range(tstart, tend)
    energy = []
    leakages = {}
    gammas = []
    betas = []
    an = []
    for j, t in enumerate(tps):
        print(j)
        a = ma.MeasurementAnalysis(timestamp=t, auto=False)
        gammas.append(np.array(
            a.data_file["Experimental Data"]["Experimental Metadata"]['gammas']))
        betas.append(np.array(
            a.data_file["Experimental Data"]["Experimental Metadata"]['betas']))
        e = list(a.data_file["Analysis"]["Processed data"]["analysis_params_dict"][
                     "energy"])
        for qb, leak in \
        a.data_file["Analysis"]["Processed data"]["analysis_params_dict"][
            "leakage"].items():
            if qb not in leakages:
                leakages[qb] = []
            leakages[qb].append(list(
                a.data_file["Analysis"]["Processed data"]["analysis_params_dict"][
                    "leakage"][qb]))
        energy.append(e)
        an.append(a)
    clear_output()

    gg, bb, energy = np.asarray(gammas), np.asarray(betas), np.asarray(energy)
    print(gg.shape)
    print(bb.shape)
    print(energy.shape)
    if save:
        f = a.folder
        np.save(f + "\\gammas.npy", gg)
        np.save(f + "\\betas.npy", bb)
        np.save(f + "\\energy.npy", energy)
        for qb in leakages:
            np.save(f + f"\\leakage_{qb}.npy", leakages[qb])
    if plot:
        # plot landscape
        fig, ax = plt.subplots()
        cmap = "seismic"
        im = ax.pcolormesh(np.pad(gg, (1, 0), "linear_ramp"),
                           np.pad(bb, (1, 0), "linear_ramp"),
                           energy, cmap=cmap, vmin=-1.05, vmax=1.05)
        ax.set_xlabel("gamma (rad.)")
        ax.set_ylabel("beta (rad.)")
        cb = fig.colorbar(im, label=r"Energy (arb. u.)")
        if save:
            fig.savefig(f + "\\energy_landscape.png")
        # plot leakage
        for qb in leakages:
            fig, ax = plt.subplots()
            im = ax.pcolormesh(np.pad(gg, (1, 0), "linear_ramp"),
                               np.pad(bb, (1, 0), "linear_ramp"), leakages[qb])
            ax.set_xlabel("gamma (rad.)")
            ax.set_ylabel("beta (rad.)")
            cb = fig.colorbar(im, label=f"leakage {qb}")
            if save:
                fig.savefig(f + f"\\leakage_{qb}.png")

    return np.asarray(gammas), np.asarray(betas), energy, leakages, an


def analyze_qaoa_landscape_rowwise(qubits, tps, save=True, plot=True, plot_range=None,
                                   overwrite_gates_info=None):
    start_time = time.time()
    if len(tps) == 2:
        tps = a_tools.get_timestamps_in_range(tps[0], tps[1])

    a = ma.MeasurementAnalysis(auto=False, timestamp=tps[0])
    metadata = h5d.read_dict_from_hdf5({}, a.data_file['Experimental Data'][
        'Experimental Metadata'])
    qb_names = eval(metadata['qb_names'])
    qbs_qaoa = [{qb.name:qb for qb in qubits}[qbn] for qbn in qb_names]

    gates_info = eval(metadata['gates_info'])
    if overwrite_gates_info is not None:
        gates_info = overwrite_gates_info
    ng = len(metadata['gammas'])
    nb = len(tps)
    energy = np.zeros((ng, nb))
    c_infos, _ = qaoa.QAOAHelper.get_corr_and_coupl_info(gates_info)
    energy_individual = {str(c_info): np.zeros((ng, nb)) for c_info in c_infos}
    avg_z = {qb.name: np.zeros((ng, nb)) for qb in qbs_qaoa}
    leakage = {qb.name: np.zeros((ng, nb)) for qb in qbs_qaoa}
    gg = np.zeros((ng, nb))
    bb = np.zeros((ng, nb))
    a.data_file.close()

    for qb in qbs_qaoa:
        gen.load_settings(qb, timestamp=tps[0])
        qb.update_detector_functions()

    an = []
    for j, t in enumerate(tps):
        print("--- %s seconds ---" % (time.time() - start_time))
        print(f"{j}: {t}")
        channel_map = {qb.name: qb.int_avg_classif_det.value_names for qb in
                       qbs_qaoa}
        options = dict(
            channel_map=channel_map,
            plot_proj_data=False,
            plot_raw_data=False)
        a = tda.MultiQubit_TimeDomain_Analysis(
            qb_names=qb_names, t_start=t,
            options_dict=options,
            auto=False)
        a.extract_data()
        a.process_data()

        gg[:, j] = np.transpose(a.metadata['gammas'])
        bb[:, j] = np.transpose(a.metadata['betas'])

        print("--- %s seconds ---" % (time.time() - start_time))
        # additional processing
        a.proc_data_dict['analysis_params_dict'] = {}
        a.proc_data_dict['analysis_params_dict']['energy'] = []
        a.proc_data_dict['analysis_params_dict']['energy_individual'] = {}
        a.proc_data_dict['analysis_params_dict']['avg_z'] = {}
        a.proc_data_dict['analysis_params_dict']['leakage'] = {}
        for qb in qb_names:
            a.proc_data_dict['analysis_params_dict']['avg_z'][qb] = []
            a.proc_data_dict['analysis_params_dict']['leakage'][qb] = []
        for c_info in c_infos:
            a.proc_data_dict['analysis_params_dict']['energy_individual'][
                str(c_info)] = []
        for i in range(ng):
            qubit_states = []
            exp_metadata = a.raw_data_dict['exp_metadata']
            filter = np.ones_like(exp_metadata['shots'], dtype=bool)
            for qb in qb_names:
                qb_probs = \
                    a.proc_data_dict['meas_results_per_qb'][qb].values()
                qb_probs = np.asarray(list(qb_probs)).T[
                           i::ng]  # (n_shots, (pg,pe,pf))
                states = np.argmax(qb_probs, axis=1)
                qubit_states.append(states)
                leaked_shots = states == 2
                a.proc_data_dict['analysis_params_dict']["leakage"][qb].append(
                    np.sum(leaked_shots) / exp_metadata['shots'])
                filter = np.logical_and(filter, states != 2)
            qb_states_filtered = np.array(qubit_states).T[filter]
            e = qaoa.ProblemHamiltonians.nbody_zterms_individual(qb_states_filtered,
                                                                 gates_info)
            a.proc_data_dict['analysis_params_dict']['energy'].append(
                np.sum(list(e.values())))

            for c_info in c_infos:
                a.proc_data_dict['analysis_params_dict']['energy_individual'][
                    str(c_info)].append(e[c_info])
            avg_z_array = qaoa.average_sigmaz(qb_states_filtered)
            for qb_ind, qb in enumerate(qb_names):
                a.proc_data_dict['analysis_params_dict']["avg_z"][qb].append(
                    avg_z_array[qb_ind])

        energy[:, j] = a.proc_data_dict['analysis_params_dict']['energy']
        for c_info in c_infos:
            energy_individual[str(c_info)][:, j] = \
                a.proc_data_dict['analysis_params_dict']['energy_individual'][
                    str(c_info)]
        for qb in qb_names:
            leakage[qb][:, j] = a.proc_data_dict['analysis_params_dict']["leakage"][
                qb]
            avg_z[qb][:, j] = a.proc_data_dict['analysis_params_dict']["avg_z"][qb]

        # save
        a.save_processed_data()
        a.data_file.close()
        an.append(a)
    print("--- %s seconds ---" % (time.time() - start_time))

    if save:
        f = a.raw_data_dict['folder']
        np.save(f + "\\gammas.npy", gg)
        np.save(f + "\\betas.npy", bb)
        np.save(f + "\\energy.npy", energy)
        for qb in qbs_qaoa:
            np.save(f + f"\\leakage_{qb.name}.npy", leakage[qb.name])

    if plot:
        fig, ax = plt.subplots()
        im = plot_colormesh(gg, bb, energy, colorbar=False, labels=False, ax=ax,
                            vmin=None if plot_range is None else plot_range[0],
                            vmax=None if plot_range is None else plot_range[1]
                            )
        # im = ax.pcolormesh(gg,bb, energy, cmap= "seismic", )
        ax.set_xlabel("gamma (rad.)")
        ax.set_ylabel("beta (rad.)")
        cb = fig.colorbar(im, label="energy", )
        # ax.plot(gammas[iteration_mask][::1] % np.pi,betas[iteration_mask][::1], "-", linewidth=1, color="c" )
        if save:
            fig.savefig(f + "\\energy_landscape.png")
        for c_info in c_infos:
            fig, ax = plt.subplots()
            im = plot_colormesh(gg, bb, energy_individual[str(c_info)],
                                colorbar=False, labels=False, ax=ax,
                                vmin=None if plot_range is None else plot_range[0],
                                vmax=None if plot_range is None else plot_range[1]
                                )
            # im = ax.pcolormesh(gg,bb, energy, cmap= "seismic", )
            ax.set_xlabel("gamma (rad.)")
            ax.set_ylabel("beta (rad.)")
            cb = fig.colorbar(im, label="energy", )
            # ax.plot(gammas[iteration_mask][::1] % np.pi,betas[iteration_mask][::1], "-", linewidth=1, color="c" )
            if save:
                fig.savefig(f + f"\\energy_landscape_{str(c_info)}.png")
        for qb in qb_names:
            fig, ax = plt.subplots()
            im = plot_colormesh(gg, bb, avg_z[qb], colorbar=False, labels=False,
                                ax=ax,
                                vmin=-1, vmax=1
                                )
            # im = ax.pcolormesh(gg,bb, energy, cmap= "seismic", )
            ax.set_xlabel("gamma (rad.)")
            ax.set_ylabel("beta (rad.)")
            cb = fig.colorbar(im, label="energy", )
            # ax.plot(gammas[iteration_mask][::1] % np.pi,betas[iteration_mask][::1], "-", linewidth=1, color="c" )
            if save:
                fig.savefig(f + f"\\avg_z_landscape_{qb}.png")

        for qbn, l in leakage.items():
            fig, ax = plt.subplots()
            im = ax.pcolormesh(gg, bb, l)
            ax.set_xlabel("gamma (rad.)")
            ax.set_ylabel("beta (rad.)")
            cb = fig.colorbar(im, label=f"leakage {qbn}")
            if save:
                fig.savefig(f + f"\\leakage_{qbn}_landscape.png")
    return (an)


def plot_colormesh(xx, yy, zz, ax=None, labels=True, colorbar=True, **plot_kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    plot_kwargs = deepcopy(plot_kwargs)
    vmin = plot_kwargs.pop("vmin", None)
    vmax = plot_kwargs.pop("vmax", None)
    cmap = plot_kwargs.pop("cmap", "seismic")
    im = ax.pcolormesh(np.pad(xx, (1, 0), "linear_ramp"),
                       np.pad(yy, (1, 0), "linear_ramp"), zz,
                       cmap=cmap, vmin=vmin, vmax=vmax,
                       **plot_kwargs)
    if labels:
        ax.set_xlabel("gamma (rad.)")
        ax.set_ylabel("beta (rad.)")
    if colorbar:
        cb = fig.colorbar(im, label=r"$\sigma_z \sigma_z$ (arb. u.)")
    return im



###############################################################################
#           sequence debugging, simulation, plotting
###############################################################################

def printseg(self):
    string_repr = f"---- {self.name} ----\n"
    for i, p in enumerate(self.unresolved_pulses):
        string_repr += f"{i} {p.pulse_obj.name}: " + repr(p.__dict__) + "\n"
    print(string_repr)

def seq2qutip(seq, qb_names, q=None):
    if q is None:
        q = QubitCircuit(len(qb_names), reverse_states=False)
    for seg in seq.segments.values():
        seg.resolve_segment()
        for p in seg.unresolved_pulses:
            if p.op_code != '':
                # l = p.pulse_obj.length
                # t = p.pulse_obj._t0 + l / 2
                # print(f"{t*1e6: .3f}us {p.op_code}")
                qb = qb_names.index(p.op_code[-3:])
                op_code = p.op_code[:-4]
                qbt = 0
                if op_code[-3:-1] == 'qb':
                    qbt = qb_names.index(op_code[-3:])
                    op_code = op_code[:-4]
                if op_code[-1:] == 's':
                    op_code = op_code[:-1]
                if op_code[:4] == 'upCZ':
                    val = -float(op_code[4:])/180*np.pi if len(op_code) > 4 else np.pi
                    q.add_gate("CPHASE", controls=qb, targets=qbt, arg_value=val)
                    # print("CPHASE", qb, qbt, val)
                elif op_code == 'RO':
                    continue
                elif op_code[0] == 'I':
                    continue
                else:
                    if op_code[0] == 'm':
                        factor = -1
                        op_code = op_code[1:]
                    else:
                        factor = 1
                    gate_type = 'R'+op_code[:1]
                    val = float(op_code[1:])
#                     print(gate_type, qb, factor*val/180*np.pi)
                    q.add_gate(gate_type, targets=qb, arg_value=factor*val/180*np.pi)
        return q

def rotate_state(psi):
    """
        rotate such that first nonzero element is real positive
        to make states comparable despite different global phase
    """
    for i in range(len(np.asarray(psi))):
        if abs(psi[i]) > 0:
            return psi*np.exp(-1j*np.angle(psi[i]))
    raise('State is zero')

def seq2tikz(seq, qb_names, tscale=1e-6):
    last_z = [(-np.inf, 0)] * len(qb_names)

    output = ''
    z_output = ''
    start_output = '\\documentclass{standalone}\n\\usepackage{tikz}\n\\begin{document}\n\\scalebox{2}{'
    start_output += '\\begin{tikzpicture}[x=10cm,y=2cm]\n'
    start_output += '\\tikzstyle{CZdot} = [shape=circle, thick,draw,inner sep=0,minimum size=.5mm, fill=black]\n'
    start_output += '\\tikzstyle{gate} = [draw,fill=white,minimum width=1cm, rotate=90]\n'
    start_output += '\\tikzstyle{zgate} = [rotate=0]\n'
    tmin = np.inf
    tmax = -np.inf
    num_single_qb = 0
    num_two_qb = 0
    num_virtual = 0
    for seg in seq.segments.values():
        seg.resolve_segment()
        for p in seg.unresolved_pulses:
            if p.op_code != '' and p.op_code[:2] != 'RO':
                l = p.pulse_obj.length
                t = p.pulse_obj._t0 + l / 2
                tmin = min(tmin, p.pulse_obj._t0)
                tmax = max(tmax, p.pulse_obj._t0 + p.pulse_obj.length)
                #                 print(t, tmin, tmax)
                #                 print(p.op_code)
                qb = qb_names.index(p.op_code[-3:])
                op_code = p.op_code[:-4]
                qbt = 0
                if op_code[-3:-1] == 'qb':
                    qbt = qb_names.index(op_code[-3:])
                    op_code = op_code[:-4]
                if op_code[-1:] == 's':
                    op_code = op_code[:-1]
                if op_code[:4] == 'upCZ':
                    num_two_qb += 1
                    if len(op_code) > 4:
                        val = -float(op_code[4:])
                        gate_formatted = f'{gate_type}{(factor * val):.1f}'.replace(
                            '.0', '')
                        output += f'\\draw({t / tscale:.4f},-{qb})  node[CZdot] {{}} -- ({t / tscale:.4f},-{qbt}) node[gate, minimum height={l / tscale * 100:.4f}mm] {{\\tiny {gate_formatted}}};\n'
                    else:
                        output += f'\\draw({t / tscale:.4f},-{qb})  node[CZdot] {{}} -- ({t / tscale:.4f},-{qbt}) node[CZdot] {{}};\n'
                elif op_code[0] == 'I':
                    continue
                else:
                    if op_code[0] == 'm':
                        factor = -1
                        op_code = op_code[1:]
                    else:
                        factor = 1
                    gate_type = 'R' + op_code[:1]
                    val = float(op_code[1:])
                    if val == 180:
                        gate_formatted = op_code[:1]
                    else:
                        gate_formatted = f'{gate_type}{(factor * val):.1f}'.replace(
                            '.0', '')
                    #                     print(gate_type, qb, factor*val/180*np.pi)
                    #                     q.add_gate(gate_type, targets=qb, arg_value=factor*val/180*np.pi)
                    if l == 0:
                        if t - last_z[qb][0] > 1e-9:
                            z_height = 0 if (
                                        t - last_z[qb][0] > 100e-9 or last_z[qb][
                                    1] >= 3) else last_z[qb][1] + 1
                            z_output += f'\\draw[dashed,thick,shift={{(0,.03)}}] ({t / tscale:.4f},-{qb})--++(0,{0.3 + z_height * 0.1});\n'
                        else:
                            z_height = last_z[qb][1] + 1
                        z_output += f'\\draw({t / tscale:.4f},-{qb})  node[zgate,shift={{({(0, .35 + z_height * .1)})}}] {{\\tiny {gate_formatted}}};\n'
                        last_z[qb] = (t, z_height)
                        num_virtual += 1
                    else:
                        output += f'\\draw({t / tscale:.4f},-{qb})  node[gate, minimum height={l / tscale * 100:.4f}mm] {{\\tiny {gate_formatted}}};\n'
                        num_single_qb += 1
        qb_output = ''
        for qb, qb_name in enumerate(qb_names):
            qb_output += f'\draw ({tmin / tscale:.4f},-{qb}) node[left] {{{qb_name}}} -- ({tmax / tscale:.4f},-{qb});\n'
        output = start_output + qb_output + output + z_output
        axis_ycoord = -len(qb_names) + .4
        output += f'\\foreach\\x in {{{tmin / tscale},{tmin / tscale + .2},...,{tmax / tscale}}} \\pgfmathprintnumberto[fixed]{{\\x}}{{\\tmp}} \draw (\\x,{axis_ycoord})--++(0,-.1) node[below] {{\\tmp}} ;\n'
        output += f'\\draw[->] ({tmin / tscale},{axis_ycoord}) -- ({tmax / tscale},{axis_ycoord}) node[right] {{$t/\\mathrm{{\\mu s}}$}};\n'
        output += '\\end{tikzpicture}}\end{document}'
        output += f'\n# {num_single_qb} single-qubit gates, {num_two_qb} two-qubit gates, {num_virtual} virtual gates'
        return output