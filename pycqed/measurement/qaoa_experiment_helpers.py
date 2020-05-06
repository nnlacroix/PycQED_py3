import numpy as np
from numpy import array
import matplotlib
import matplotlib.pyplot as plt
import pycqed.analysis.measurement_analysis as ma
from pycqed.analysis import analysis_toolbox as a_tools
from scipy.interpolate import interp1d
from qutip import QubitCircuit
import datetime as dt
import time
from pycqed.measurement import hdf5_data as h5d
from pycqed.measurement import qaoa as qaoa
from pycqed.utilities import general as gen
from pycqed.analysis_v2 import timedomain_analysis as tda
from copy import deepcopy


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

    [a.data_file.close() for a in an]
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
    x_off = (xx[1, 0] - xx[0, 0]) / 2
    y_off = (yy[0, 1] - yy[0, 0]) / 2
    current_cmap = matplotlib.cm.get_cmap(cmap)
    current_cmap.set_bad(color='yellow')
    im = ax.pcolormesh(np.pad(xx + x_off, (1, 0), "linear_ramp"),
                       np.pad(yy + y_off, (1, 0), "linear_ramp"), zz,
                       cmap=current_cmap, vmin=vmin, vmax=vmax,
                       **plot_kwargs)
    ax.set_xlim([np.min(xx),np.max(xx)])
    ax.set_ylim([np.min(yy),np.max(yy)])
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
        tmp = deepcopy(p.__dict__)
        tmp.pop('pulse_obj', None)
        string_repr += f"{i} {p.pulse_obj.name}: " + repr(tmp) + ', ' + repr(p.pulse_obj.__dict__)  + "\n"
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

def rotate_operator(U):
    """
        rotate such that first nonzero element is real positive
        to make operator comparable despite different global phase
    """
    Utmp = np.asarray(U)
    for i in range(np.shape(Utmp)[0]):
        for j in range(np.shape(Utmp)[0]):
            if abs(Utmp[i,j]) > 0:
                Utmp = Utmp*np.exp(-1j*np.angle(Utmp[i,j]))
    raise('Operator is zero')


def seq_gate_count(seq, qb_names):
    num_single_qb = 0
    num_two_qb = 0
    num_virtual = 0
    for seg in seq.segments.values():
        for p in seg.unresolved_pulses:
            if p.op_code != '' and p.op_code[:2] != 'RO':
                l = p.pulse_obj.length
                op_code = p.op_code[:-4]
                if op_code[-3:-1] == 'qb':
                    qbt = qb_names.index(op_code[-3:])
                    num_two_qb += 1
                elif op_code[0] == 'I':
                    continue
                elif l == 0:
                    num_virtual += 1
                else:
                    num_single_qb += 1
        return (num_two_qb, num_single_qb, num_virtual)