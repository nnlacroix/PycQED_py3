import logging
log = logging.getLogger(__name__)

import sys
import lmfit
import numpy as np
from scipy import optimize
from collections import OrderedDict
from pycqed.analysis import fitting_models as fit_mods
from pycqed.analysis_v3 import fitting as fit_module
from pycqed.analysis_v3 import plotting as plot_module
from pycqed.analysis_v3 import helper_functions as hlp_mod
from pycqed.analysis_v3 import processing_pipeline as ppmod
from pycqed.analysis_v3 import pipeline_analysis as pla
from copy import deepcopy

pla.search_modules.add(sys.modules[__name__])

# Create pipelines

# run ramsey analysis
def rb_analysis(data_dict, keys_in, **params):
    """
    Does single qubit RB analysis. Prepares fits and plot, and extracts
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
    prep_plot_dicts = hlp_mod.pop_param('prep_plot_dicts', data_dict,
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
    nr_seeds = len(sp[0][mospm[mobjn][0]][0])
    if len(data_dict['timestamps']) > 1:
        nr_seeds *= len(data_dict['timestamps'])
    cliffords = sp[1][mospm[mobjn][1]][0]

    # prepare fitting
    if prep_fit_dicts:
        prepare_fitting(data_dict, data_to_proc_dict, cliffords, nr_seeds,
                        **params)

    if do_fitting:
        getattr(fit_module, 'run_fitting')(data_dict, keys_in=list(
                data_dict['fit_dicts']),**params)
        # extract EPC, leakage, and seepage from fits and save to
        # data_dict[meas_obj_name]
        analyze_fit_results(data_dict, keys_in, **params)

    # prepare plots
    if prep_plot_dicts:
        prepare_plots(data_dict, keys_in, cliffords, nr_seeds, **params)
    if do_plotting:
        getattr(plot_module, 'plot')(data_dict, keys_in=list(
            data_dict['plot_dicts']), **params)


def prepare_fitting(data_dict, data_to_proc_dict, cliffords, nr_seeds,
                    **params):
    cp, mobjn = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['cp', 'mobjn'], **params)
    conf_level = hlp_mod.get_param('conf_level', data_dict,
                                   default_value=0.68, **params)
    do_simple_fit = hlp_mod.get_param(
        'do_simple_fit', data_dict, default_value=True, **params)
    d = hlp_mod.get_param('d', data_dict, default_value=2, **params)
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
            # fit based on the IBM style
            fit_module.prepare_rbleakage_fit_dict(
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
            epsilon = calculate_confidence_intervals(
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


def analyze_fit_results(data_dict, keys_in, **params):
    mobjn = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['mobjn'], **params)
    fit_dicts = hlp_mod.get_param('fit_dicts', data_dict, raise_error=True)
    for keyi in keys_in:
        fit_res = fit_dicts['rb_fit' + keyi]['fit_res']
        hlp_mod.add_param(f'{mobjn}.EPC value',
                          fit_res.params['error_per_Clifford'].value,
                          data_dict, replace_value=True)
        hlp_mod.add_param(f'{mobjn}.EPC stderr',
                          fit_res.params['fidelity_per_Clifford'].stderr,
                          data_dict, replace_value=True)
        hlp_mod.add_param(f'{mobjn}.depolarization parameter value',
                          fit_res.params['p'].value,
                          data_dict, replace_value=True)
        hlp_mod.add_param(f'{mobjn}.depolarization parameter stderr',
                          fit_res.params['p'].stderr,
                          data_dict, replace_value=True)
        if 'pf' in keyi:
            A = fit_res.best_values['Amplitude']
            Aerr = fit_res.params['Amplitude'].stderr
            p = fit_res.best_values['p']
            perr = fit_res.params['p'].stderr
            # Google-style leakage and seepage:
            # https://journals.aps.org/pra/abstract/10.1103/PhysRevA.97.032306
            hlp_mod.add_param(f'{mobjn}.Google-style leakage value',
                              A*(1-p),
                              data_dict,
                              replace_value=True)
            hlp_mod.add_param(f'{mobjn}.Google-style leakage stderr',
                              np.sqrt((A*perr)**2 + (Aerr*(p-1))**2),
                              data_dict,
                              replace_value=True)
            hlp_mod.add_param(f'{mobjn}.Google-style seepage value',
                              (1-A)*(1-p),
                              data_dict,
                              replace_value=True)
            hlp_mod.add_param(f'{mobjn}.Google-style seepage stderr',
                              np.sqrt((Aerr*(p-1))**2 + ((A-1)*perr)**2),
                              data_dict,
                              replace_value=True)

            # IBM-style leakage and seepage:
            # https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.116.020501
            fit_res = fit_dicts['rbleak_fit' + keyi]['fit_res']
            hlp_mod.add_param(f'{mobjn}.IBM-style leakage value',
                              fit_res.best_values['pu'],
                              data_dict,
                              replace_value=True)
            hlp_mod.add_param(f'{mobjn}.IBM-style leakage stderr',
                              fit_res.params['pu'].stderr,
                              data_dict,
                              replace_value=True)
            hlp_mod.add_param(f'{mobjn}.IBM-style seepage value',
                              fit_res.best_values['pd'],
                              data_dict,
                              replace_value=True)
            hlp_mod.add_param(f'{mobjn}.IBM-style seepage stderr',
                              fit_res.params['pd'].stderr,
                              data_dict,
                              replace_value=True)


def prepare_plots(data_dict, keys_in, cliffords, nr_seeds, **params):
    cp, mospm, mobjn = hlp_mod.get_measurement_properties(
            data_dict, props_to_extract=['cp', 'mospm', 'mobjn'], **params)

    # prepare raw data plot
    if hlp_mod.get_param('prepare_raw_plot', params,
                         default_value=len(data_dict['timestamps']) == 1):
        # check whether active reset was used
        reset_reps = hlp_mod.get_reset_reps_from_data_dict(data_dict)
        if reset_reps != 0:
            swpts = deepcopy(np.repeat(cliffords, nr_seeds))
            swpts = np.concatenate([
                swpts, hlp_mod.get_cal_sweep_points(
                    swpts, cp, mobjn)])
            swpts_with_rst = np.repeat(swpts, reset_reps+1)
            swpts_with_rst = np.arange(len(swpts_with_rst))
            plot_module.prepare_1d_raw_data_plot_dicts(
                data_dict, xvals=swpts_with_rst, sp_name=mospm[mobjn][1],
                **params)

            filtered_raw_keys = [k for k in data_dict.keys() if 'filter' in k]
            if len(filtered_raw_keys) > 0:
                plot_module.prepare_1d_raw_data_plot_dicts(
                    data_dict=data_dict,
                    keys_in=filtered_raw_keys,
                    figure_name='raw_data_filtered',
                    xvals=swpts, sp_name=mospm[mobjn][1],
                    **params)
        else:
            plot_module.prepare_1d_raw_data_plot_dicts(
                data_dict, sp_name=mospm[mobjn][1],
                xvals=np.repeat(cliffords, nr_seeds))

    plot_dicts = OrderedDict()
    keys_in_std = hlp_mod.get_param('keys_in_std', data_dict, raise_error=False,
                                 **params)
    for keyi, keys in zip(keys_in, keys_in_std):
        figure_name = 'RB_' + keyi
        sp_name = mospm[mobjn][1]

        # plot data
        plot_module.prepare_1d_plot_dicts(
            data_dict=data_dict,
            keys_in=[keyi],
            figure_name=figure_name,
            sp_name=sp_name,
            yerr=keys,
            do_plotting=False, **params)

        if len(cp.states) != 0:
            # plot cal states
            plot_module.prepare_cal_states_plot_dicts(
                data_dict=data_dict,
                keys_in=[keyi],
                figure_name=figure_name,
                sp_name=sp_name,
                do_plotting=False, **params)

        if 'fit_dicts' in data_dict:
            # plot fits
            fit_dicts = data_dict['fit_dicts']
            textstr = ''
            if 'pf' in keyi:
                # plot IBM-style leakage fit + textbox
                plot_module.prepare_fit_plot_dicts(
                    data_dict=data_dict,
                    figure_name=figure_name,
                    fit_names=['rbleak_fit' + keyi],
                    plot_params={'color': 'C1',
                                 'setlabel': 'fit - Google',
                                 'legend_ncol': 3},
                    do_plotting=False, **params)
                textstr += get_rb_textbox_properties(
                    data_dict, fit_dicts['rbleak_fit' + keyi]['fit_res'],
                    textstr_style=['leakage_ibm'],
                    **params)[0]

            # plot fit trace
            plot_module.prepare_fit_plot_dicts(
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
                # get T1, T2, gate length from HDF file
                get_meas_obj_coh_times(data_dict, **params)
                F_T1, p_T1 = calc_coherence_limited_fidelity(
                    hlp_mod.get_param(f'{mobjn}.T1', data_dict),
                    hlp_mod.get_param(f'{mobjn}.T2', data_dict),
                    hlp_mod.get_param(f'{mobjn}.ge_sigma', data_dict)*
                    hlp_mod.get_param(f'{mobjn}.ge_nr_sigma', data_dict),
                    hlp_mod.get_param('gate_decomp', data_dict,
                                      default_value='HZ', **params))
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
                F_T1 = None

            # add texbox
            textstr, ha, hp, va, vp = get_rb_textbox_properties(
                data_dict, fit_res, F_T1=None if 'pf' in keyi else F_T1,
                va='top' if 'pg' in keyi else 'bottom',
                textstr_style='leakage_google' if 'pf' in keyi else 'regular',
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


def get_leakage_google_textstr(data_dict, fit_res, **params):
    mobjn = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['mobjn'], **params)
    textstr = '\nGoogle style:'
    textstr += ('\n' + 'p = {:.4f}% $\pm$ {:.3f}%'.format(
        fit_res.params['p'].value*100, fit_res.params['p'].stderr*100))

    L_value = hlp_mod.get_param(f'{mobjn}.Google-style leakage value',
                                data_dict, raise_error=True)
    textstr += f'\nL = {100*L_value:.4f}%'
    L_stderr = hlp_mod.get_param(f'{mobjn}.Google-style leakage stderr',
                                 data_dict)
    if L_stderr is not None:
        textstr += f'$\pm$ {100*L_stderr:.3f}%'

    S_value = hlp_mod.get_param(f'{mobjn}.Google-style seepage value',
                                data_dict, raise_error=True)
    textstr += f'\nS = {100*S_value:.4f}%'
    S_stderr = hlp_mod.get_param(f'{mobjn}.Google-style seepage stderr',
                                 data_dict)
    if S_stderr is not None:
        textstr += f'$\pm$ {100*S_stderr:.3f}%'
    return textstr


def get_leakage_ibm_textstr(fit_res, **params):
    textstr = '\nIBM style:'
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


def get_rb_textstr(fit_res, F_T1=None, **params):
    textstr = ('$r_{\mathrm{Cl}}$' + ' = {:.4f}% $\pm$ {:.3f}%'.format(
        (1-fit_res.params['fidelity_per_Clifford'].value)*100,
        fit_res.params['fidelity_per_Clifford'].stderr*100))
    if F_T1 is not None:
        textstr += ('\n$r_{\mathrm{coh-lim}}$  = ' +
                    '{:.3f}%'.format((1-F_T1)*100))
    textstr += ('\n' + 'p = {:.4f}% $\pm$ {:.3f}%'.format(
        fit_res.params['p'].value*100, fit_res.params['p'].stderr*100))
    textstr += ('\n' + r'$\langle \sigma_z \rangle _{m=0}$ = ' +
                '{:.2f} $\pm$ {:.2f}'.format(
                    fit_res.params['Amplitude'].value +
                    fit_res.params['offset'].value,
                    np.sqrt(fit_res.params['offset'].stderr**2 +
                            fit_res.params['Amplitude'].stderr**2)))
    return textstr


def get_rb_textbox_properties(data_dict, fit_res, F_T1=None,
                              textstr_style=(), textstr='', **params):
    if len(textstr_style) != 0:
        if 'regular' in textstr_style:
            textstr += get_rb_textstr(fit_res, F_T1)
        if 'leakage_google' in textstr_style:
            textstr += get_leakage_google_textstr(data_dict, fit_res, **params)
        if 'leakage_ibm' in textstr_style:
            textstr += get_leakage_ibm_textstr(fit_res)
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
        vp = 0.025

    return textstr, ha, hp, va, vp


def calc_coherence_limited_fidelity(T1, T2, pulse_length, gate_decomp='HZ'):
    '''
    Formula from Asaad et al.
    pulse separation is time between start of pulses

    Returns:
        F_cl (float): decoherence limited fildelity
        p (float): decoherence limited depolarization parameter
    '''
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


def get_meas_obj_coh_times(data_dict, **params):
    mobjn = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['mobjn'], **params)
    # Get from the hdf5 file any parameters specified in
    # params_dict and numeric_params.
    params_dict = {}
    s = 'Instrument settings.' + mobjn
    for trans_name in ['', '_ef']:
        params_dict[f'{mobjn}.T1{trans_name}'] = s+f'.T1{trans_name}'
        params_dict[f'{mobjn}.T2{trans_name}'] = s+f'.T2{trans_name}'
    for trans_name in ['ge', 'ef']:
        params_dict[f'{mobjn}.{trans_name}_sigma'] = \
            s+f'.{trans_name}_sigma'
        params_dict[f'{mobjn}.{trans_name}_nr_sigma'] = \
            s+f'.{trans_name}_nr_sigma'
    hlp_mod.get_params_from_hdf_file(data_dict, params_dict=params_dict,
                                     numeric_params=list(params_dict), **params)


def calculate_confidence_intervals(
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


