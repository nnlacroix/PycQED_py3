import logging
log = logging.getLogger(__name__)

import sys
import lmfit
import numpy as np
from scipy import optimize
from collections import OrderedDict
from pycqed.analysis import fitting_models as fit_mods
from pycqed.analysis_v3 import fitting as fit_mod
from pycqed.analysis_v3 import plotting as plot_mod
from pycqed.analysis_v3 import helper_functions as hlp_mod
from pycqed.analysis_v3 import processing_pipeline as pp_mod
from pycqed.analysis_v3 import pipeline_analysis as pla
from pycqed.measurement import sweep_points as sp_mod
from pycqed.measurement.calibration import calibration_points as cp_mod
from copy import deepcopy

pla.search_modules.add(sys.modules[__name__])

# Create pipelines


def pipeline_single_qubit_rb_ssro(meas_obj_names, mospm, sweep_points,
                                     n_shots, dim_hilbert, cal_points=None,
                                     ro_thresholds=None, nreps=1,
                                     plot_all_shots=False, slow_cliffords=False):

    """
    Wrapper to create the standard processing pipeline for an single qubit RB
        measurement, measured in SSRO.
    WARNING: if you use plot_all_shots=True, disable data saving. It will try
        to save a huge string of the large numpy array this node will generate.
    :param meas_obj_names: list of measured object names
    :param mospm: meas_obj_sweep_points_map
    :param sweep_points: SweepPoints object (of one file if the measurement
        was split into several files)
    :param n_shots: number of shots
    :param dim_hilbert: dimension of Hilebert space. 4 for 2QB RB, 2 for 1QB RB
    :param cal_points: CalibrationPoints object
    :param ro_thresholds: optional (the threshold_data node can also extract
        them from the data_dict. See docstring there).
        Dict with meas_obj_names as keys and their readout thresholds as values.
    :param nreps: int specifying the number of files to combine into one
        measurement. IMPORTANT! This feature only works if the measurement was
        split by seeds, not by cliffords. Meaning that each measurement file
        contains data for all the Cliffords in sweep_points, but for a subset
        of the total seeds.
    :param plot_all_shots: bool specifying whether to produce a raw plot of
        of all the shots vs cliffords. SEE WARNING ABOVE.
    :return: the unresolved ProcessingPipeline
    """

    sweep_points = sp_mod.SweepPoints.cast_init(sweep_points)
    seeds_dim = sweep_points.find_parameter(f'{meas_obj_names[0]}_seeds')
    clf_dim = sweep_points.find_parameter(f'{meas_obj_names[0]}_cliffords')
    if cal_points is None:
        num_cal_states = 0
    else:
        if isinstance(cal_points, str):
            cal_points = cp_mod.CalibrationPoints.from_string(cal_points)
        num_cal_states = len(cal_points.states)
    if slow_cliffords :
        # n_segments = nr_seeds + nr_cal_segments
        n_segments_subexp = nreps*(sweep_points.length(seeds_dim) + num_cal_states)
        n_segments_all = n_segments_subexp
        # n_sequences = nr_cliffords
        n_sequences = sweep_points.length(clf_dim)
    else:
        # n_segments = nr_cliffords + nr_cal_segments
        n_segments_subexp = nreps*(sweep_points.length(clf_dim) + num_cal_states)
        n_segments_all = n_segments_subexp
        # n_sequences = nr_seeds
        n_sequences = sweep_points.length(seeds_dim)

    processing_pipeline = pp_mod.ProcessingPipeline()
    if nreps > 1:
        processing_pipeline.add_node('combine_datafiles_by_clifford',
                                     keys_in='raw',
                                     n_shots=n_shots,
                                     meas_obj_names=meas_obj_names)
    keys_in = 'previous combine_datafiles_by_clifford' if nreps > 1 else 'raw'
    processing_pipeline.add_node('threshold_data',
                                 keys_in=keys_in,
                                 ro_thresholds=ro_thresholds,
                                 meas_obj_names=meas_obj_names)
    processing_pipeline.add_node('average_data',
                                 # shape=(n_shots, n_segments_all*n_sequences),
                                 # averaging_axis=0,
                                 shape=(n_sequences, n_shots, n_segments_all),
                                 averaging_axis=1,
                                 keys_in='previous threshold_data',
                                 meas_obj_names=meas_obj_names)
    for label in ['rb']:
        pp = pp_mod.ProcessingPipeline(global_keys_out_container=label)
        pp.add_node('average_data',
                    shape=(n_sequences, n_segments_subexp),
                    averaging_axis=-1 if slow_cliffords else 0,
                    keys_in='previous average_data',
                    meas_obj_names=meas_obj_names)
        pp.add_node('get_std_deviation',
                    shape=(n_sequences, n_segments_subexp) ,
                    averaging_axis=-1 if slow_cliffords else 0,
                    keys_in='previous average_data',
                    meas_obj_names=meas_obj_names)
        pp.add_node('rb_analysis',
                    d=dim_hilbert,
                    keys_in=f'previous {label}.average_data',
                    keys_in_std=f'previous {label}.get_std_deviation',
                    keys_out=None,
                    meas_obj_names=meas_obj_names)
        for mobjn in meas_obj_names:
            cliffords = sweep_points.get_sweep_params_property(
                'values', clf_dim, mospm[mobjn][clf_dim])
            if plot_all_shots:
                pp.add_node('prepare_1d_raw_data_plot_dicts',
                            sp_name=mospm[mobjn][clf_dim],
                            xvals=np.repeat(cliffords, n_segments_all*n_shots
                                if slow_cliffords else n_sequences*n_shots),
                            do_plotting=True,
                            figname_suffix=f'shots_{label}',
                            title_suffix=' - All shots',
                            plot_params={'linestyle': 'none'},
                            keys_in=keys_in,
                            keys_out=None,
                            meas_obj_names=mobjn)
            if slow_cliffords:
                xvals = np.repeat(cliffords, n_segments_subexp)
            else:
                xvals = np.tile(cliffords, n_sequences)
            pp.add_node('prepare_1d_raw_data_plot_dicts',
                        sp_name=mospm[mobjn][clf_dim],
                        xvals=xvals,
                        do_plotting=True,
                        figname_suffix=f'{label}',
                        title_suffix=' - All seeds',
                        ylabel='Probability, $P(|e\\rangle)$',
                        yunit='',
                        plot_params={'linestyle': 'none'},
                        keys_in='previous average_data',
                        keys_out=None,
                        meas_obj_names=mobjn)
        processing_pipeline += pp
    return processing_pipeline


def pipeline_interleaved_rb_irb_classif(meas_obj_names, mospm, sweep_points,
                                        dim_hilbert, cal_points=None, nreps=1):
    """
    Wrapper to create the standard processing pipeline for an interleaved RB/RIB
        measurement, measured with a the classifier detector with qutrit readout
    :param meas_obj_names: list of measured object names
    :param mospm: meas_obj_sweep_points_map
    :param sweep_points: SweepPoints object (of one file if the measurement
        was split into several files)
    :param dim_hilbert: dimension of Hilebert space. 4 for 2QB RB, 2 for 1QB RB
    :param cal_points: CalibrationPoints object
    :param nreps: int specifying the number of files to combine into one
        measurement. IMPORTANT! This feature only works if the measurement was
        split by seeds, not by cliffords. Meaning that each measurement file
        contains data for all the Cliffords in sweep_points, but for a subset
        of the total seeds.
    :return: the unresolved ProcessingPipeline
    """
    sweep_points = sp_mod.SweepPoints.cast_init(sweep_points)
    if cal_points is None:
        num_cal_states = 0
    else:
        if isinstance(cal_points, str):
            cal_points = cp_mod.CalibrationPoints.from_string(cal_points)
        num_cal_states = len(cal_points.states)
    # n_segments = nr_seeds + nr_cal_segments
    n_segments = nreps*(sweep_points.length(0) + num_cal_states)
    # n_sequences = nr_cliffords
    n_sequences = sweep_points.length(1)
    processing_pipeline = pp_mod.ProcessingPipeline()
    if nreps > 1:
        processing_pipeline.add_node('combine_datafiles_by_clifford',
                                     keys_in='raw',
                                     interleaved_irb=True,
                                     meas_obj_names=meas_obj_names)
    for label in ['rb', 'irb']:
        pp = pp_mod.ProcessingPipeline(global_keys_out_container=label)
        keys_in = 'previous combine_datafiles_by_clifford' if \
            nreps > 1 else 'raw'
        pp.add_node(f'{label}_data_from_interleaved_msmt',
                    keys_in=keys_in, meas_obj_names=meas_obj_names)
        pp.add_node('average_data',
                    shape=(n_sequences, n_segments),
                    keys_in=f'previous {label}.{label}_data_'
                            f'from_interleaved_msmt',
                    meas_obj_names=meas_obj_names)
        pp.add_node('get_std_deviation',
                    shape=(n_sequences, n_segments),
                    keys_in=f'previous {label}.{label}_data_'
                            f'from_interleaved_msmt',
                    meas_obj_names=meas_obj_names)
        pp.add_node('rb_analysis',
                    d=dim_hilbert,
                    keys_in=f'previous {label}.average_data',
                    keys_in_std=f'previous {label}.get_std_deviation',
                    keys_out=None,
                    meas_obj_names=meas_obj_names)
        for mobjn in meas_obj_names:
            cliffords = sweep_points.get_sweep_params_property(
                'values', 1, mospm[mobjn][-1])
            pp.add_node('prepare_1d_raw_data_plot_dicts',
                        sp_name=mospm[mobjn][-1],
                        xvals=np.repeat(cliffords, n_segments),
                        do_plotting=True,
                        figname_suffix=f'{label}',
                        title_suffix=' - All seeds',
                        ylabel='Probability, $P(|ee\\rangle)$' if
                            mobjn=='correlation_object' else None,
                        yunit='',
                        keys_in=f'previous {label}.{label}_data_'
                                f'from_interleaved_msmt',
                        keys_out=None,
                        meas_obj_names=mobjn)
        processing_pipeline += pp

    # calculate interleaved gate error
    processing_pipeline.add_node('irb_gate_error',
                                 meas_obj_names='correlation_object',
                                 d=dim_hilbert)
    return processing_pipeline


def pipeline_interleaved_rb_irb_ssro(meas_obj_names, mospm, sweep_points,
                                     n_shots, dim_hilbert, cal_points=None,
                                     ro_thresholds=None, nreps=1,
                                     plot_all_shots=False):

    """
    Wrapper to create the standard processing pipeline for an interleaved RB/RIB
        measurement, measured in SSRO.
    WARNING: if you use plot_all_shots=True, disable data saving. It will try
        to save a huge string of the large numpy array this node will generate.
    :param meas_obj_names: list of measured object names
    :param mospm: meas_obj_sweep_points_map
    :param sweep_points: SweepPoints object (of one file if the measurement
        was split into several files)
    :param n_shots: number of shots
    :param dim_hilbert: dimension of Hilebert space. 4 for 2QB RB, 2 for 1QB RB
    :param cal_points: CalibrationPoints object
    :param ro_thresholds: optional (the threshold_data node can also extract
        them from the data_dict. See docstring there).
        Dict with meas_obj_names as keys and their readout thresholds as values.
    :param nreps: int specifying the number of files to combine into one
        measurement. IMPORTANT! This feature only works if the measurement was
        split by seeds, not by cliffords. Meaning that each measurement file
        contains data for all the Cliffords in sweep_points, but for a subset
        of the total seeds.
    :param plot_all_shots: bool specifying whether to produce a raw plot of
        of all the shots vs cliffords. SEE WARNING ABOVE.
    :return: the unresolved ProcessingPipeline
    """

    sweep_points = sp_mod.SweepPoints.cast_init(sweep_points)
    if cal_points is None:
        num_cal_states = 0
    else:
        if isinstance(cal_points, str):
            cal_points = cp_mod.CalibrationPoints.from_string(cal_points)
        num_cal_states = len(cal_points.states)
    # n_segments = nr_seeds + nr_cal_segments
    n_segments_subexp = nreps*(sweep_points.length(0) + num_cal_states)
    n_segments_all = 2*n_segments_subexp
    # n_sequences = nr_cliffords
    n_sequences = sweep_points.length(1)
    processing_pipeline = pp_mod.ProcessingPipeline()
    if nreps > 1:
        processing_pipeline.add_node('combine_datafiles_by_clifford',
                                     keys_in='raw',
                                     n_shots=n_shots,
                                     interleaved_irb=True,
                                     meas_obj_names=meas_obj_names)
    keys_in = 'previous combine_datafiles_by_clifford' if nreps > 1 \
        else 'raw'
    processing_pipeline.add_node('threshold_data',
                                 keys_in=keys_in,
                                 ro_thresholds=ro_thresholds,
                                 meas_obj_names=meas_obj_names)
    processing_pipeline.add_node('correlate_qubits',
                                 keys_in='previous threshold_data',
                                 meas_obj_names=meas_obj_names,
                                 joint_processing=True, num_keys_out=1,
                                 keys_out_container='correlation_object',
                                 add_mobjn_container=False)
    processing_pipeline.add_node('average_data',
                                 shape=(n_segments_all*n_sequences, n_shots),
                                 keys_in='previous threshold_data',
                                 meas_obj_names=meas_obj_names)
    processing_pipeline.add_node('average_data',
                                 shape=(n_segments_all*n_sequences, n_shots),
                                 keys_in='previous correlate_qubits',
                                 meas_obj_names=['correlation_object'])
    meas_obj_names = deepcopy(meas_obj_names)
    meas_obj_names += ['correlation_object']
    for label in ['rb', 'irb']:
        pp = pp_mod.ProcessingPipeline(global_keys_out_container=label)
        pp.add_node(f'{label}_data_from_interleaved_msmt',
                    keys_in='previous average_data',
                    meas_obj_names=meas_obj_names)
        pp.add_node('average_data',
                    shape=(n_sequences, n_segments_subexp),
                    keys_in=f'previous {label}.{label}_data_'
                            f'from_interleaved_msmt',
                    meas_obj_names=meas_obj_names)
        pp.add_node('get_std_deviation',
                    shape=(n_sequences, n_segments_subexp),
                    keys_in=f'previous {label}.{label}_data_'
                            f'from_interleaved_msmt',
                    meas_obj_names=meas_obj_names)
        pp.add_node('rb_analysis',
                    d=dim_hilbert,
                    keys_in=f'previous {label}.average_data',
                    keys_in_std=f'previous {label}.get_std_deviation',
                    keys_out=None,
                    meas_obj_names=meas_obj_names)
        for mobjn in meas_obj_names:
            cliffords = sweep_points.get_sweep_params_property(
                'values', 1, mospm[mobjn][-1])
            if plot_all_shots and mobjn in meas_obj_names[:-1]:
                keys_in = 'previous combine_datafiles_by_clifford' \
                    if nreps > 1 else 'raw'
                pp.add_node('prepare_1d_raw_data_plot_dicts',
                            sp_name=mospm[mobjn][-1],
                            xvals=np.repeat(cliffords,
                                            n_segments_all*n_shots),
                            do_plotting=True,
                            figname_suffix=f'shots_{label}',
                            title_suffix=' - All shots',
                            keys_in=keys_in,
                            keys_out=None,
                            meas_obj_names=mobjn)
            pp.add_node('prepare_1d_raw_data_plot_dicts',
                        sp_name=mospm[mobjn][-1],
                        xvals=np.repeat(cliffords, n_segments_subexp),
                        do_plotting=True,
                        figname_suffix=f'{label}',
                        title_suffix=' - All seeds',
                        ylabel='Probability, ' + ('$P(|ee\\rangle)$' if
                            mobjn=='correlation_object' else '$P(|e\\rangle)$'),
                        yunit='',
                        keys_in=f'previous {label}.{label}_data_'
                                f'from_interleaved_msmt',
                        keys_out=None,
                        meas_obj_names=mobjn)
        processing_pipeline += pp

    # calculate interleaved gate error
    processing_pipeline.add_node('irb_gate_error',
                                 meas_obj_names='correlation_object',
                                 d=dim_hilbert)
    return processing_pipeline


# nodes related to extracting data
def combine_datafiles_by_clifford(data_dict, keys_in, keys_out,
                                  slow_cliffords=False, interleaved_irb=False,
                                  **params):
    """
    NOT FULLY IMPLEMENTED FOR slow_cliffords == True!!!
    Combines the data from an interleaved RB/IRB measurement that was saved in
    multiple files into one data set that would look as if it had all been
    taken in one measurements (one file).
    :param data_dict: OrderedDict containing data to be processed and where
                    processed data is to be stored
    :param keys_in: list of key names or dictionary keys paths in
                    data_dict for the data to be processed
    :param keys_out: list of key names or dictionary keys paths in
                    data_dict for the processed data to be saved into
    :param params: keyword arguments:
        Should contain 'exp_metadata_list', 'n_shots', 'mospm', 'rev_movnm',
        'cp' if they are not in data_dict
        ToDo: put n_shots info in the metadata (27.07.2020)
    :return:

    Assumptions:
        - ASSUMES MEASUREMENT WAS SPLIT BY SEEDS NOT BY CLIFFORDS. Meaning that
        each measurement file contains data for all the Cliffords in
        sweep_points, but for a subset of the total seeds.

    """
    assert len(keys_in) == len(keys_out)
    n_shots = hlp_mod.get_param('n_shots', data_dict, default_value=1, **params)
    mospm, rev_movnm, cp, mobjn = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['mospm', 'rev_movnm', 'cp', 'mobjn'],
        **params)
    metadata_list = hlp_mod.get_param('exp_metadata_list', data_dict,
                                      raise_error=True, **params)
    sp_list = [hlp_mod.get_param('sweep_points', mdl, raise_error=True)
               for mdl in metadata_list]
    sp0 = sp_mod.SweepPoints.cast_init(sp_list[0])
    seeds_dim = sp0.find_parameter(f'{mobjn}_seeds')
    clf_dim = sp0.find_parameter(f'{mobjn}_cliffords')
    if slow_cliffords:
        segment_chunk = sp0.length(seeds_dim) + len(cp.states)
    else:
        segment_chunk = sp0.length(clf_dim) + len(cp.states)
    nr_cliffords = sp0.length(clf_dim)

    data_to_proc_dict = hlp_mod.get_data_to_process(data_dict, keys_in)
    for keyi, keyo in zip(keys_in, keys_out):
        data = data_to_proc_dict[keyi]
        if np.ndim(data) != 2:
            raise ValueError(f'Data corresponding to {keyi} is not 2D.')
        # take the segment_chunk * n_shots for each clifford from each array
        # in data list and concatenate them. Put all the nr_cliffords
        # concatenations in the list data_combined
        data_combined = [np.concatenate(
            [d[j*segment_chunk*n_shots:(j+1)*segment_chunk*n_shots]
             for d in data]) for j in np.arange(
            (interleaved_irb+1)*nr_cliffords)]
        # concatenate all the lists in data_combined to get one complete
        # array of data
        data_combined = np.concatenate(data_combined)
        hlp_mod.add_param(keyo, data_combined, data_dict, **params)

    # update the sweep_points if they were a list
    nr_sp0 = sp0.length(0)
    nr_exp = len(sp_list)
    sp_all_vals_list = [np.zeros(nr_exp*nr_sp0) for _
                        in range(len(sp0.get_sweep_dimension(0)))]

    for i, sp in enumerate(sp_list):
        sp = sp_mod.SweepPoints.cast_init(sp)
        sp_vals_list = sp.get_sweep_params_property('values', 0, 'all')
        for j, sp_vals in enumerate(sp_vals_list):
            sp_all_vals_list[j][i::nr_exp] = sp_vals

    sweep_points = sp_mod.SweepPoints()
    for i, sp_name in enumerate(sp0.get_sweep_dimension(0)):
        sweep_points.add_sweep_parameter(
            sp_name, sp_all_vals_list[i],
            sp0.get_sweep_params_property('unit', 0, sp_name),
            sp0.get_sweep_params_property('label', 0, sp_name))
    sweep_points += [sp0.get_sweep_dimension(1)]
    hlp_mod.add_param('exp_metadata.sweep_points', sweep_points,
                      data_dict, replace_value=True)


def rb_data_from_interleaved_msmt(data_dict, keys_in, keys_out=None,
                                  start_index=0, **params):
    data_to_proc_dict = hlp_mod.get_data_to_process(data_dict, keys_in)
    sp, cp = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['sp', 'cp'], **params)
    segment_chunk = sp.length(0) + len(cp.states)
    nr_cliffords = sp.length(1)

    if keys_out is None:
        keys_out = [f'rb_data_from_interleaved_msmt.{s}' for s in keys_in]
    for keyi, keyo in zip(keys_in, keys_out):
        data = data_to_proc_dict[keyi]
        selected_data = np.concatenate([
            data[j*segment_chunk:(j+1)*segment_chunk]
            for j in np.arange(2*nr_cliffords)[start_index::2]])
        hlp_mod.add_param(
            keyo, selected_data, data_dict, **params)


def irb_data_from_interleaved_msmt(data_dict, keys_in, keys_out=None,
                                   start_index=1, **params):
    data_to_proc_dict = hlp_mod.get_data_to_process(data_dict, keys_in)
    sp, cp = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['sp', 'cp'], **params)
    segment_chunk = sp.length(0) + len(cp.states)
    nr_cliffords = sp.length(1)

    if keys_out is None:
        keys_out = [f'irb_data_from_interleaved_msmt.{s}' for s in keys_in]
    for keyi, keyo in zip(keys_in, keys_out):
        data = data_to_proc_dict[keyi]
        selected_data = np.concatenate([
            data[j*segment_chunk:(j+1)*segment_chunk]
            for j in np.arange(2*nr_cliffords)[start_index::2]])
        hlp_mod.add_param(
            keyo, selected_data, data_dict, **params)


# run ramsey analysis
def rb_analysis(data_dict, keys_in, **params):
    """
    Does single qubit RB analysis. Prepares fits and plots, and extracts
    errors per clifford.
    :param data_dict: OrderedDict containing data to be processed and where
                processed data is to be stored
    :param keys_in: list of key names or dictionary keys paths in
                data_dict for the data to be processed

    Assumptions:
        - cal_points, sweep_points, qb_sweep_points_map, qb_name exist in
        metadata or params
        - expects a 2d sweep with nr_seeds on innermost sweep and cliffords
        on outermost
        - if active reset was used, 'filter' must be in the key names of the
        filtered data if you want the filtered raw data to be plotted
    """
    data_to_proc_dict = hlp_mod.get_data_to_process(data_dict, keys_in)
    keys_in = list(data_to_proc_dict)

    prep_fit_dicts = hlp_mod.pop_param('prep_fit_dicts', data_dict,
                                       default_value=True, node_params=params)
    do_fitting = hlp_mod.pop_param('do_fitting', data_dict,
                                   default_value=True, node_params=params)
    prepare_plotting = hlp_mod.pop_param('prepare_plotting', data_dict,
                                         default_value=True, node_params=params)
    do_plotting = hlp_mod.pop_param('do_plotting', data_dict,
                                    default_value=True, node_params=params)

    keys_in_std = hlp_mod.get_param('keys_in_std', data_dict,
                                    raise_error=False, **params)
    if keys_in_std is None:
        keys_in_std = [''] * len(keys_in)
    if len(keys_in_std) != len(keys_in):
        raise ValueError('keys_in_std and keys_in do not have '
                         'the same length.')

    sp, mospm, mobjn = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['sp', 'mospm', 'mobjn'], **params)
    seeds_dim = sp.find_parameter(f'{mobjn}_seeds')
    clf_dim = sp.find_parameter(f'{mobjn}_cliffords')
    nr_seeds = sp.length(seeds_dim)
    if len(data_dict['timestamps']) > 1:
        nr_seeds *= len(data_dict['timestamps'])
    cliffords = sp.get_sweep_params_property('values', clf_dim,
                                             mospm[mobjn][clf_dim])

    # prepare fitting
    if prep_fit_dicts:
        prepare_rb_fitting(data_dict, data_to_proc_dict, cliffords, nr_seeds,
                        **params)

    if do_fitting:
        getattr(fit_mod, 'run_fitting')(data_dict, keys_in=list(
                data_dict['fit_dicts']),**params)
        # extract EPC, leakage, and seepage from fits and save to
        # data_dict[meas_obj_name]
        analyze_rb_fit_results(data_dict, keys_in, **params)

    # prepare plots
    if prepare_plotting:
        prepare_rb_plots(data_dict, keys_in, cliffords, **params)
    if do_plotting:
        getattr(plot_mod, 'plot')(data_dict, keys_in=list(
            data_dict['plot_dicts']), **params)


def prepare_rb_fitting(data_dict, data_to_proc_dict, cliffords, nr_seeds,
                       **params):
    cp, mobjn = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['cp', 'mobjn'], **params)
    conf_level = hlp_mod.get_param('conf_level', data_dict,
                                   default_value=0.68, **params)
    do_simple_fit = hlp_mod.get_param(
        'do_simple_fit', data_dict, default_value=True, **params)
    d = hlp_mod.get_param('d', data_dict, raise_error=True, **params)
    print('d: ', d)

    fit_dicts = OrderedDict()
    rb_mod = lmfit.Model(fit_mods.RandomizedBenchmarkingDecay)
    rb_mod.set_param_hint('Amplitude', value=0.5)
    rb_mod.set_param_hint('p', value=.9)
    rb_mod.set_param_hint('offset', value=.5)
    rb_mod.set_param_hint('fidelity_per_Clifford',
                          expr=f'1-(({d}-1)*(1-p)/{d})')
    rb_mod.set_param_hint('error_per_Clifford',
                          expr='1-fidelity_per_Clifford')
    gate_decomp = hlp_mod.get_param('gate_decomp', data_dict,
                                    default_value='HZ', **params)
    if gate_decomp == 'XY':
        rb_mod.set_param_hint('fidelity_per_gate',
                              expr='fidelity_per_Clifford**(1./1.875)')
    elif gate_decomp == 'HZ':
        rb_mod.set_param_hint('fidelity_per_gate',
                              expr='fidelity_per_Clifford**(1./1.125)')
    else:
        raise ValueError('Gate decomposition not recognized.')
    rb_mod.set_param_hint('error_per_gate', expr='1-fidelity_per_gate')
    guess_pars = rb_mod.make_params()

    keys_in_std = hlp_mod.get_param('keys_in_std', data_dict, raise_error=False,
                                    **params)
    for keyi, keys in zip(data_to_proc_dict, keys_in_std):
        if 'pf' in keyi:
            # if this is the |f> state population data, then do an additional
            # fit based on the Google style
            fit_mod.prepare_rbleakage_fit_dict(
                data_dict, [keyi], indep_var_array=cliffords,
                fit_name='rbleak_fit', **params)

        # do standard fit to A*p**m + B
        key = 'rb_fit' + keyi
        data_fit = hlp_mod.get_msmt_data(data_to_proc_dict[keyi], cp, mobjn)

        model = deepcopy(rb_mod)
        fit_dicts[key] = {
            'fit_fn': fit_mods.RandomizedBenchmarkingDecay,
            'fit_xvals': {'numCliff': cliffords},
            'fit_yvals': {'data': data_fit},
            'guess_pars': guess_pars}

        if do_simple_fit:
            fit_kwargs = {}
        elif keys is not None:
            fit_kwargs = {'scale_covar': False,
                          'weights': 1/hlp_mod.get_param(
                              keys, data_dict)}
        else:
            # Run once to get an estimate for the error per Clifford
            fit_res = model.fit(data_fit, numCliff=cliffords,
                                params=guess_pars)
            # Use the found error per Clifford to standard errors for
            # the data points fro Helsen et al. (2017)
            epsilon_guess = hlp_mod.get_param('epsilon_guess', data_dict,
                                              default_value=0.01, **params)
            epsilon = calculate_rb_confidence_intervals(
                nr_seeds=nr_seeds,
                nr_cliffords=cliffords,
                depolariz_param=fit_res.best_values['p'],
                conf_level=conf_level,
                epsilon_guess=epsilon_guess, d=2)

            hlp_mod.add_param(
                keys, epsilon, data_dict,
                replace_value=params.get('replace_value', False))
            # Run fit again with scale_covar=False, and
            # weights = 1/epsilon if an entry in epsilon_sqrd is 0,
            # replace it with half the minimum value in the epsilon_sqrd
            # array
            idxs = np.where(epsilon == 0)[0]
            epsilon[idxs] = min([eps for eps in epsilon if eps != 0])/2
            fit_kwargs = {'scale_covar': False, 'weights': 1/epsilon}
        fit_dicts[key]['fit_kwargs'] = fit_kwargs

    hlp_mod.add_param('fit_dicts', fit_dicts, data_dict, update_value=True)


def analyze_rb_fit_results(data_dict, keys_in, **params):
    mobjn = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['mobjn'], **params)
    keys_out_container = hlp_mod.get_param('keys_out_container', data_dict,
                                           default_value=mobjn, **params)
    fit_dicts = hlp_mod.get_param('fit_dicts', data_dict, raise_error=True)
    for keyi in keys_in:
        fit_res = fit_dicts['rb_fit' + keyi]['fit_res']
        hlp_mod.add_param(f'{keys_out_container}.EPC value',
                          fit_res.params['error_per_Clifford'].value,
                          data_dict, replace_value=True)
        hlp_mod.add_param(f'{keys_out_container}.EPC stderr',
                          fit_res.params['fidelity_per_Clifford'].stderr,
                          data_dict, replace_value=True)
        hlp_mod.add_param(f'{keys_out_container}.depolarization parameter value',
                          fit_res.params['p'].value,
                          data_dict, replace_value=True)
        hlp_mod.add_param(f'{keys_out_container}.depolarization parameter stderr',
                          fit_res.params['p'].stderr,
                          data_dict, replace_value=True)
        if 'pf' in keyi:
            A = fit_res.best_values['Amplitude']
            Aerr = fit_res.params['Amplitude'].stderr
            p = fit_res.best_values['p']
            perr = fit_res.params['p'].stderr
            # IBM-style leakage and seepage:
            # https://journals.aps.org/pra/abstract/10.1103/PhysRevA.97.032306
            hlp_mod.add_param(f'{keys_out_container}.IBM-style leakage value',
                              A*(1-p),
                              data_dict,
                              replace_value=True)
            hlp_mod.add_param(f'{keys_out_container}.IBM-style leakage stderr',
                              np.sqrt((A*perr)**2 + (Aerr*(p-1))**2),
                              data_dict,
                              replace_value=True)
            hlp_mod.add_param(f'{keys_out_container}.IBM-style seepage value',
                              (1-A)*(1-p),
                              data_dict,
                              replace_value=True)
            hlp_mod.add_param(f'{keys_out_container}.IBM-style seepage stderr',
                              np.sqrt((Aerr*(p-1))**2 + ((A-1)*perr)**2),
                              data_dict,
                              replace_value=True)

            # Google-style leakage and seepage:
            # https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.116.020501
            fit_res = fit_dicts['rbleak_fit' + keyi]['fit_res']
            hlp_mod.add_param(f'{keys_out_container}.Google-style leakage value',
                              fit_res.best_values['pu'],
                              data_dict,
                              replace_value=True)
            hlp_mod.add_param(f'{keys_out_container}.Google-style leakage stderr',
                              fit_res.params['pu'].stderr,
                              data_dict,
                              replace_value=True)
            hlp_mod.add_param(f'{keys_out_container}.Google-style seepage value',
                              fit_res.best_values['pd'],
                              data_dict,
                              replace_value=True)
            hlp_mod.add_param(f'{keys_out_container}.Google-style seepage stderr',
                              fit_res.params['pd'].stderr,
                              data_dict,
                              replace_value=True)

    if hlp_mod.get_param('plot_T1_lim', data_dict, default_value=False,
                         **params):
        # get T1, T2, gate length from HDF file
        get_meas_obj_coh_times(data_dict, **params)
        F_T1, p_T1 = calc_rb_coherence_limited_fidelity(
            hlp_mod.get_param(f'{mobjn}.T1', data_dict),
            hlp_mod.get_param(f'{mobjn}.T2', data_dict),
            hlp_mod.get_param(f'{mobjn}.ge_sigma', data_dict) *
            hlp_mod.get_param(f'{mobjn}.ge_nr_sigma', data_dict),
            hlp_mod.get_param('gate_decomp', data_dict,
                              default_value='HZ', **params))
        hlp_mod.add_param(f'{keys_out_container}.EPC coh_lim', 1-F_T1,
                          data_dict, replace_value=True)
        hlp_mod.add_param(
            f'{keys_out_container}.depolarization parameter coh_lim', p_T1,
            data_dict, replace_value=True)


def prepare_rb_plots(data_dict, keys_in, cliffords, **params):
    sp, cp, mospm, mobjn, movnm = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['sp', 'cp', 'mospm', 'mobjn', 'movnm'],
        **params)
    clf_dim = sp.find_parameter(f'{mobjn}_cliffords')

    plot_dicts = OrderedDict()
    keys_in_std = hlp_mod.get_param('keys_in_std', data_dict, raise_error=False,
                                    **params)
    classified_msmt = any([v == 3 for v in [len(chs) for chs in movnm.values()]])
    for keyi, keys in zip(keys_in, keys_in_std):
        figure_name = 'RB_' + keyi
        sp_name = mospm[mobjn][clf_dim]

        # plot data
        ylabel = 'Probability, $P(|ee\\rangle)$' if 'corr' in mobjn else None
        if ylabel is None and not classified_msmt:
            ylabel = 'Probability, $P(|e\\rangle)$'
        plot_mod.prepare_1d_plot_dicts(data_dict=data_dict, keys_in=[keyi],
                                          figure_name=figure_name,
                                          ylabel=ylabel,
                                          sp_name=sp_name, yerr_key=keys,
                                          do_plotting=False, **params)

        if len(cp.states) != 0:
            # plot cal states
            plot_mod.prepare_cal_states_plot_dicts(data_dict=data_dict,
                                                      keys_in=[keyi],
                                                      figure_name=figure_name,
                                                      sp_name=sp_name,
                                                      do_plotting=False,
                                                      **params)

        if 'fit_dicts' in data_dict:
            # plot fits
            fit_dicts = data_dict['fit_dicts']
            textstr = ''
            if 'pf' in keyi:
                # plot Google-style leakage fit + textbox
                plot_mod.prepare_fit_plot_dicts(
                    data_dict=data_dict,
                    figure_name=figure_name,
                    fit_names=['rbleak_fit' + keyi],
                    plot_params={'color': 'C1',
                                 'setlabel': 'fit - Google',
                                 'legend_ncol': 3},
                    do_plotting=False, **params)
                textstr += get_rb_textbox_properties(
                    data_dict, fit_dicts['rbleak_fit' + keyi]['fit_res'],
                    textstr_style=['leakage_google'],
                    **params)[0]

            # plot fit trace
            plot_mod.prepare_fit_plot_dicts(
                data_dict=data_dict,
                figure_name=figure_name,
                fit_names=['rb_fit' + keyi],
                plot_params={'color': 'C0',
                             'setlabel': 'fit - IBM' if 'pf' in keyi else 'fit',
                             'legend_ncol': 3},
                do_plotting=False, **params)

            # plot coherence-limit
            fit_res = fit_dicts['rb_fit' + keyi]['fit_res']
            if hlp_mod.get_param('plot_T1_lim', data_dict,
                    default_value=False, **params) and 'pf' not in keyi:
                keys_out_container = hlp_mod.get_param('keys_out_container',
                                                       data_dict,
                                                       default_value=mobjn,
                                                       **params)
                epc_T1 = hlp_mod.get_param(f'{keys_out_container}.EPC coh_lim',
                                         data_dict,  **params)
                p_T1 = hlp_mod.get_param(
                    f'{keys_out_container}.depolarization parameter coh_lim',
                    data_dict,  **params)
                clfs_fine = np.linspace(cliffords[0], cliffords[-1], 1000)
                T1_limited_curve = fit_res.model.func(
                    clfs_fine, fit_res.best_values['Amplitude'], p_T1,
                    fit_res.best_values['offset'])
                plot_dicts['t1Lim_' + keyi] = {
                    'fig_id': figure_name,
                    'plotfn': 'plot_line',
                    'xvals': clfs_fine,
                    'yvals': T1_limited_curve,
                    'setlabel': 'coh-lim',
                    'do_legend': True,
                    'legend_ncol': 3,
                    'legend_bbox_to_anchor': (1, -0.15),
                    'legend_pos': 'upper right',
                    'linestyle': '--',
                    'marker': ''}
            else:
                epc_T1 = None

            # add texbox
            textstr, ha, hp, va, vp = get_rb_textbox_properties(
                data_dict, fit_res, epc_T1=None if 'pf' in keyi else epc_T1,
                va='top' if 'pg' in keyi else 'bottom',
                textstr_style='leakage_ibm' if 'pf' in keyi else 'regular',
                textstr=textstr if 'pf' in keyi else '', **params)
            plot_dicts['text_msg_' + keyi] = {
                'fig_id': figure_name,
                'plotfn': 'plot_text',
                'ypos': vp,
                'xpos': hp,
                'horizontalalignment': ha,
                'verticalalignment': va,
                'box_props': None,
                'text_string': textstr}

    hlp_mod.add_param('plot_dicts', plot_dicts, data_dict, update_value=True)


def prepare_cz_irb_plot(data_dict_rb, data_dict_irb, keys_in, **params):
    plot_dicts = OrderedDict()
    do_plotting = params.pop('do_plotting', False)
    textstr = ''
    for keyi in keys_in:
        for i, data_dict in enumerate([data_dict_rb, data_dict_irb]):
            cp, sp, mospm, mobjn = hlp_mod.get_measurement_properties(
                data_dict, props_to_extract=['cp', 'sp', 'mospm', 'mobjn'],
                **params)
            clf_dim = sp.find_parameter(f'{mobjn}_cliffords')
            keys_in_std = hlp_mod.get_param('keys_in_std', data_dict,
                                            raise_error=False, **params)
            figure_name = 'IRB' + mobjn
            key_suffix = 'RB' if i == 0 else 'IRB'
            sp_name = mospm[mobjn][clf_dim]
            cliffords = sp.get_sweep_params_property('values', 1, sp_name)

            # plot data
            plot_dicts.update(
                plot_mod.prepare_1d_plot_dicts(
                    data_dict=data_dict, keys_in=[keyi],
                    figure_name=figure_name, key_suffix=key_suffix,
                    sp_name=sp_name, #yerr_key=keys,
                    ylabel=r'$\langle \sigma_z\sigma_z \rangle$ correlator',
                    plot_params={'color': 'C0' if i == 0 else 'C1',
                                 'setlabel': key_suffix},
                    do_plotting=False, **params))

            if len(cp.states) != 0:
                # plot cal states
                plot_dicts.update(
                    plot_mod.prepare_cal_states_plot_dicts(
                        data_dict=data_dict, keys_in=[keyi],
                        figure_name=figure_name, sp_name=sp_name,
                        key_suffix=key_suffix,
                        plot_params={'color': 'C0' if i == 0 else 'C1',
                                     'setlabel': key_suffix},
                        do_plotting=False, **params))

            if 'fit_dicts' in data_dict:
                # plot fits
                fit_dicts = data_dict['fit_dicts']
                # plot fit trace
                plot_dicts.update(
                    plot_mod.prepare_fit_plot_dicts(
                        data_dict=data_dict, figure_name=figure_name,
                        key_suffix=key_suffix,
                        fit_names=['rb_fit' + keyi],
                        plot_params={
                            'color': 'C0' if i == 0 else 'C1',
                            'setlabel': f'{key_suffix} - fit'},
                        do_plotting=False, **params))

                # plot coherence-limit
                fit_res = fit_dicts['rb_fit' + keyi]['fit_res']
                if hlp_mod.get_param('plot_T1_lim', data_dict,
                                     default_value=False, **params):
                    keys_out_container = hlp_mod.get_param('keys_out_container',
                                                           data_dict,
                                                           default_value=mobjn,
                                                           **params)
                    epc_T1 = hlp_mod.get_param(
                        f'{keys_out_container}.EPC coh_lim',
                        data_dict,  **params)
                    p_T1 = hlp_mod.get_param(
                        f'{keys_out_container}.depolarization parameter coh_lim',
                        data_dict,  **params)
                    clfs_fine = np.linspace(cliffords[0], cliffords[-1], 1000)
                    T1_limited_curve = fit_res.model.func(
                        clfs_fine, fit_res.best_values['Amplitude'], p_T1,
                        fit_res.best_values['offset'])
                    plot_dicts['t1Lim_' + keyi + key_suffix] = {
                        'fig_id': figure_name,
                        'plotfn': 'plot_line',
                        'xvals': clfs_fine,
                        'yvals': T1_limited_curve,
                        'setlabel': f'{suffix} coh-lim',
                        'do_legend': True,
                        'legend_ncol': 3,
                        'legend_bbox_to_anchor': (1, -0.15),
                        'legend_pos': 'upper right',
                        'linestyle': '--',
                        'marker': ''}
                else:
                    epc_T1 = None

                # add texbox
                textstr, ha, hp, va, vp = get_rb_textbox_properties(
                    data_dict, fit_res, epc_T1=epc_T1,
                    va=params.pop('va', 'bottom'),
                    textstr_style='irb', suffix=key_suffix,
                    textstr=textstr, **params)
        if len(textstr) != 0:
            plot_dicts['text_msg_' + keyi + key_suffix] = {
                'fig_id': figure_name,
                'plotfn': 'plot_text',
                'ypos': vp,
                'xpos': hp,
                'horizontalalignment': ha,
                'verticalalignment': va,
                'box_props': None,
                'text_string': textstr}

    hlp_mod.add_param('plot_dicts', plot_dicts, data_dict_irb,
                      update_value=True)
    if do_plotting:
        getattr(plot_mod, 'plot')(data_dict_irb, keys_in=list(plot_dicts),
                                     **params)


def get_rb_leakage_ibm_textstr(data_dict, fit_res=None, **params):
    mobjn = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['mobjn'], **params)
    keys_out_container = hlp_mod.get_param('keys_out_container', data_dict,
                                           default_value=mobjn, **params)
    textstr = 'IBM style:'
    p_value = hlp_mod.get_param(
        f'{keys_out_container}.depolarization parameter value', data_dict,
        raise_error=True)
    p_stderr = hlp_mod.get_param(
        f'{keys_out_container}.depolarization parameter stderr', data_dict)
    textstr += f'\np = {100*p_value:.4f}%'
    if p_stderr is not None:
        textstr += f'$\pm$ {100*p_stderr:.3f}%'

    L_value = hlp_mod.get_param(
        f'{keys_out_container}.IBM-style leakage value', data_dict,
        raise_error=True)
    textstr += f'\nL = {100*L_value:.4f}%'
    L_stderr = hlp_mod.get_param(
        f'{keys_out_container}.IBM-style leakage stderr', data_dict)
    if L_stderr is not None:
        textstr += f'$\pm$ {100*L_stderr:.3f}%'

    S_value = hlp_mod.get_param(
        f'{keys_out_container}.IBM-style seepage value', data_dict,
        raise_error=True)
    textstr += f'\nS = {100*S_value:.4f}%'
    S_stderr = hlp_mod.get_param(
        f'{keys_out_container}.IBM-style seepage stderr', data_dict)
    if S_stderr is not None:
        textstr += f'$\pm$ {100*S_stderr:.3f}%'
    return textstr


def get_rb_leakage_google_textstr(fit_res, **params):
    textstr = 'Google style:'
    textstr += ('\n$p_{\\uparrow}$' +
                ' = {:.4f}% $\pm$ {:.3f}%'.format(
                    fit_res.params['pu'].value*100,
                    fit_res.params['pu'].stderr*100) +
                '\n$p_{\\downarrow}$' +
                ' = {:.4f}% $\pm$ {:.3f}%'.format(
                    fit_res.params['pd'].value*100,
                    fit_res.params['pd'].stderr*100) +
                '\n$p_0$' + ' = {:.2f}% $\pm$ {:.2f}%\n'.format(
                fit_res.params['p0'].value,
                fit_res.params['p0'].stderr))
    return textstr


def get_rb_regular_textstr(fit_res, epc_T1=None, **params):
    textstr = ('$r_{\mathrm{Cl}}$' + ' = {:.4f}% $\pm$ {:.3f}%'.format(
        (1-fit_res.params['fidelity_per_Clifford'].value)*100,
        fit_res.params['fidelity_per_Clifford'].stderr*100))
    if epc_T1 is not None:
        textstr += ('\n$r_{\mathrm{coh-lim}}$  = ' +
                    '{:.3f}%'.format(epc_T1*100))
    textstr += ('\n' + 'p = {:.4f}% $\pm$ {:.3f}%'.format(
        fit_res.params['p'].value*100, fit_res.params['p'].stderr*100))
    textstr += ('\n' + r'$\langle \sigma_z \rangle _{m=0}$ = ' +
                '{:.2f} $\pm$ {:.2f}'.format(
                    fit_res.params['Amplitude'].value +
                    fit_res.params['offset'].value,
                    np.sqrt(fit_res.params['offset'].stderr**2 +
                            fit_res.params['Amplitude'].stderr**2)))
    return textstr


def get_cz_irb_textstr(fit_res,  epc_T1=None, **params):
    suffix = params.get('suffix', 'RB')
    textstr = (f'$r_{{\mathrm{{Cl}}, {{{suffix}}}}}$' +
               ' = {:.4f}% $\pm$ {:.3f}%'.format(
        (1-fit_res.params['fidelity_per_Clifford'].value)*100,
        fit_res.params['fidelity_per_Clifford'].stderr*100))
    if epc_T1 is not None:
        textstr += ('\n$r_{\mathrm{coh-lim}}$  = ' +
                    '{:.3f}%'.format(epc_T1*100))
    textstr += (f'\n$p_{{\\uparrow, {suffix}}}$' +
                ' = {:.4f}% $\pm$ {:.3f}%'.format(
                    fit_res.params['pu'].value*100,
                    fit_res.params['pu'].stderr*100) +
                f'\n$p_{{\\downarrow, {suffix}}}$' +
                ' = {:.4f}% $\pm$ {:.3f}%'.format(
                    fit_res.params['pd'].value*100,
                    fit_res.params['pd'].stderr*100))
    return textstr


def get_rb_textbox_properties(data_dict, fit_res, epc_T1=None,
                              textstr_style=(), textstr='', **params):
    if len(textstr_style) != 0:
        textstr += '\n'
        if 'regular' in textstr_style:
            textstr += get_rb_regular_textstr(fit_res, epc_T1, **params)
        if 'leakage_google' in textstr_style:
            textstr += get_rb_leakage_google_textstr(fit_res, **params)
        if 'leakage_ibm' in textstr_style:
            textstr += get_rb_leakage_ibm_textstr(data_dict, **params)
        if 'irb' in textstr_style:
            textstr += get_cz_irb_textstr(fit_res, **params)
        if len(textstr) == 0:
            raise NotImplementedError(f'The textstring style {textstr_style} '
                                      f'has not been implemented yet.')
    ha = params.pop('ha', 'right')
    hp = 0.975
    if ha == 'left':
        hp = 0.025
    va = params.pop('va', 'top')
    vp = 0.95
    if va == 'bottom':
        vp = 0.035

    return textstr, ha, hp, va, vp


def irb_gate_error(data_dict, **params):
    """
    Calculates the average gate error from a set of RB-IRB measurements and
    saves it in data_dict.
    :param data_dict: OrderedDict containing the results of running rb_analysis
        node.
    :param params: keyword arguments:
        meas_obj_names (str or list of str): name of the measurement object(s)
            for which to calculate average gate error.
            Should be correlation_object for a two-qubit RB.
        d (int): dimension of the Hilbert space
        interleaved_gate (str or int): the interleaved gate for which to
            calculate average gate error.

    Assumptions:
        - meas_obj_names, d, interleaved_gate must exist wither in data_dict,
        metadata, or params
    """
    meas_obj_names = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['mobjn'], enforce_one_meas_obj=False,
        **params)
    d = hlp_mod.get_param('d', data_dict, raise_error=True, **params)
    interleaved_gate = hlp_mod.get_param(
        'interleaved_gate', data_dict, raise_error=True, **params)
    if interleaved_gate == 4368:
        interleaved_gate = 'CZ'

    for mobjn in meas_obj_names:
        prb = hlp_mod.get_param(
            f'{mobjn}.rb.depolarization parameter value', data_dict,
            raise_error=True, **params)
        prb_err = hlp_mod.get_param(
            f'{mobjn}.rb.depolarization parameter stderr', data_dict,
            raise_error=True, **params)
        pirb = hlp_mod.get_param(
            f'{mobjn}.irb.depolarization parameter value', data_dict,
            raise_error=True, **params)
        pirb_err = hlp_mod.get_param(
            f'{mobjn}.irb.depolarization parameter stderr', data_dict,
            raise_error=True, **params)
        hlp_mod.add_param(f'{mobjn}.average_gate_error_{interleaved_gate} value',
                          ((d-1)/d)*(1 - pirb/prb),
                          data_dict, **params)
        hlp_mod.add_param(f'{mobjn}.average_gate_error_{interleaved_gate} stderr',
                          ((d-1)/d)*np.sqrt((pirb_err*prb)**2 +
                                            (prb_err*pirb)**2)/(prb**2),
                          data_dict, **params)


def calc_rb_coherence_limited_fidelity(T1, T2, pulse_length, gate_decomp='HZ'):
    """
    Formula from Asaad et al.
    pulse separation is time between start of pulses

    Returns:
        F_cl (float): decoherence limited fildelity
        p (float): decoherence limited depolarization parameter
    """
    # Np = 1.875  # Avg. number of gates per Clifford for XY decomposition
    # Np = 0.9583  # Avg. number of gates per Clifford for HZ decomposition
    if gate_decomp == 'HZ':
        Np = 1.125
    elif gate_decomp == 'XY':
        Np = 1.875
    else:
        raise ValueError('Gate decomposition not recognized.')

    F_cl = (1/6*(3 + 2*np.exp(-1*pulse_length/(T2)) +
                 np.exp(-pulse_length/T1)))**Np
    p = 2*F_cl - 1

    return F_cl, p


def get_meas_obj_coh_times(data_dict, extract_T2s=True, **params):
    mobjn = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['mobjn'], **params)
    # Get from the hdf5 file any parameters specified in
    # params_dict and numeric_params.
    params_dict = {}
    s = 'Instrument settings.' + mobjn
    for trans_name in ['', '_ef']:
        params_dict[f'{mobjn}.T1{trans_name}'] = s + f'.T1{trans_name}'
        if extract_T2s:
            params_dict[f'{mobjn}.T2{trans_name}'] = s + f'.T2_star{trans_name}'
        else:
            params_dict[f'{mobjn}.T2{trans_name}'] = s + f'.T2{trans_name}'
    for trans_name in ['ge', 'ef']:
        params_dict[f'{mobjn}.{trans_name}_sigma'] = \
            s + f'.{trans_name}_sigma'
        params_dict[f'{mobjn}.{trans_name}_nr_sigma'] = \
            s + f'.{trans_name}_nr_sigma'
    hlp_mod.get_params_from_hdf_file(data_dict, params_dict=params_dict,
                                     numeric_params=list(params_dict), **params)


def calculate_rb_confidence_intervals(
        nr_seeds, nr_cliffords, conf_level=0.68, depolariz_param=1,
        epsilon_guess=0.01, d=2):

    # From Helsen et al. (2017)
    # For each number of cliffords in nr_cliffords (array), finds epsilon
    # such that with probability greater than conf_level, the true value of
    # the survival probability, p_N_m, for a given N=nr_seeds and
    # m=nr_cliffords, is in the interval
    # [p_N_m_measured-epsilon, p_N_m_measured+epsilon]
    # See Helsen et al. (2017) for more details.

    # eta is the SPAM-dependent prefactor defined in Helsen et al. (2017)
    epsilon = []
    delta = 1-conf_level
    infidelity = (d-1)*(1-depolariz_param)/d

    for n_cl in nr_cliffords:
        if n_cl == 0:
            epsilon.append(0)
        else:
            if d == 2:
                V_short_n_cl = (13*n_cl*infidelity**2)/2
                V_long_n_cl = 7*infidelity/2
                V = min(V_short_n_cl, V_long_n_cl)
            else:
                V_short_n_cl = \
                    (0.25*(-2+d**2)/((d-1)**2)) * (infidelity**2) + \
                    (0.5*n_cl*(n_cl-1)*(d**2)/((d-1)**2)) * (infidelity**2)
                V1 = 0.25*((-2+d**2)/((d-1)**2))*n_cl*(infidelity**2) * \
                     depolariz_param**(n_cl-1) + ((d/(d-1))**2) * \
                     (infidelity**2)*(
                             (1+(n_cl-1)*(depolariz_param**(2*n_cl)) -
                              n_cl*(depolariz_param**(2*n_cl-2))) /
                             (1-depolariz_param**2)**2 )
                V = min(V1, V_short_n_cl)
            H = lambda eps: (1/(1-eps))**((1-eps)/(V+1)) * \
                            (V/(V+eps))**((V+eps)/(V+1)) - \
                            (delta/2)**(1/nr_seeds)
            epsilon.append(optimize.fsolve(H, epsilon_guess)[0])
    return np.asarray(epsilon)


