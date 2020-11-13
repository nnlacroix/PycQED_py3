import numpy as np

from pycqed.measurement.waveform_control.sequence import Sequence
from pycqed.utilities.general import temporary_value
from pycqed.measurement.waveform_control.circuit_builder import CircuitBuilder
import pycqed.measurement.awg_sweep_functions as awg_swf
from pycqed.measurement import multi_qubit_module as mqm
import pycqed.analysis_v2.base_analysis as ba
import logging
log = logging.getLogger(__name__)


class QuantumExperiment(CircuitBuilder):
    """
    Base class for Experiments with pycqed. A QuantumExperiment consists of
    3 main parts:
    - The __init__(), which takes care of initializing the parent class
     (CircuitBuilder) and setting all the attributes of the quantum experiment
    - the run_measurement(), which is the skeleton of any measurement in pycqed.
      This function should *not* be modified by child classes
    - the run_analysis(), which defaults to calling BaseDataAnalysis. This function
      may be overwritten by child classes to start measurement-specific analysis

    """
    _metadata_params = {'cal_points', 'preparation_params', 'sweep_points',
                        'channel_map', 'meas_objs'}

    def __init__(self, dev=None, qubits=None, operation_dict=None,
                 meas_objs=None, classified=False, MC=None,
                 label=None, exp_metadata=None, upload=True, measure=True,
                 analyze=True, temporary_values=(), drive="timedomain",
                 sequences=(), sequence_function=None,
                 sequence_kwargs=None, df_kwargs=None, df_name=None,
                 mc_points=None, sweep_functions=(awg_swf.SegmentHardSweep,
                                                      awg_swf.SegmentSoftSweep),
                 compression_seg_lim=None, force_2D_sweep=True, **kw):
        """
        Initializes a QuantumExperiment.

        Args:
            dev (Device): Device object used for the experiment. Defaults to None.
            qubits (list): list of qubits used for the experiment (e.g. a subset of
                qubits on the device). Defaults to None. (see circuitBuilder for more
                details).
            operation_dict (dict): dictionary with operations. Defaults to None.
                (see circuitBuilder for more details).
            meas_objs (list): list of measure object (e.g., qubits) to be read
                out (i.e. for which the detector functions will be
                prepared). Defaults to self.qubits (attribute set by
                CircuitBuilder). Required for run_measurement() when qubits
                is None.
            classified (bool): whether
            MC (MeasurementControl): MeasurementControl object. Required for
                run_measurement() if qubits is None and device is None.
            label (str): Measurement label
            exp_metadata (dict): experimental metadata saved in hdf5 file
            upload (bool): whether or not to upload the sequences to the AWGs
            measure (bool): whether or not to measure
            analyze (bool): whether or not to analyze
            temporary_values (list): list of temporary values with the form:
                [(Qcode_param_1, value_1), (Qcode_param_2, value_2), ...]
            drive (str): qubit configuration.
            sequences (list): list of sequences for the experiment. Note that
                even in the case of a single sequence, a list is required.
                Required if sequence_function is None.
            sequence_function (callable): functions returning the sequences,
                see self._prepare_sequences() for more details. Required for
                run_measurement if sequences is None
            sequence_kwargs (dict): keyword arguments passed to the sequence_function.
                see self._prepare_sequences()
            df_kwargs (dict): detector function keyword arguments.
            df_name (str): detector function name.
            mc_points (tuple): tuple of 2 lists with first and second dimension
                measurement control points (previously also called sweep_points,
                but name has changed to avoid confusion with SweepPoints):
                [first_dim_mc_points, second_dim_mc_points]. MC points
                correspond to measurement_control sweep points i.e. sweep points
                directly related to the instruments, e.g. segment readout index.
                Not required when using sweep_functions SegmentSoftSweep and
                SegmentHardSweep as these may be inferred from the sequences objects.
                In case other sweep functions are used (e.g. for sweeping instrument
                parameters), then the sweep points must be specified. Note that the list
                must always have two entries. E.g. for a 1D sweep of LO frequencies,
                mc_points should be of the form: (freqs, [])
            sweep_functions (tuple): tuple of sweepfunctions. Similarly to mc_points,
                sweep_functions has 2 entries, one for each dimension. Defaults to
                SegmentHardSweep for the first sweep dimensions and SegmentSoftSweep
                for the second dimension.
            compression_seg_lim (int): maximal number of segments that can be in a
                single sequence. If not None and the QuantumExperiment is a 2D sweep
                with more than 1 sequence, and the sweep_functions are
                (SegmentHardSweep, SegmentSoftsweep), then the quantumExperiment
                will try to compress the sequences, see Sequence.compress_2D_sweep.
            force_2D_sweep (bool): whether or not to force a two-dimensional sweep.
                In that case, even if there is only one sequence, a second
                sweep_function dimension is added. The idea is to use this more
                and more to generalize data format passed to the analysis.
            **kw:
                further keyword arguments are passed to the CircuitBuilder __init__
        """

        if qubits is None and dev is None and operation_dict is None:
            raise NotImplementedError('Experiments without qubits are not '
                                      'implemented yet. Either dev or qubits'
                                      'or operation_dict has to be provided.')
            # planned future behavior (but has to be tested in all aspects):
            # if no qubits/devive/operation_dict are provided, use empty
            # list to skip iterations over qubit lists
            # qubits = []
        super().__init__(dev=dev, qubits=qubits, operation_dict=operation_dict,
                         **kw)

        self.exp_metadata = exp_metadata
        if self.exp_metadata is None:
            self.exp_metadata = {}

        self.create_meas_objs_list(**kw, meas_objs=meas_objs)
        self.MC = MC

        self.classified = classified
        self.label = label
        self.upload = upload
        self.measure = measure
        self.temporary_values = list(temporary_values)
        self.analyze = analyze
        self.drive = drive

        self.sequences = list(sequences)
        self.sequence_function = sequence_function
        self.sequence_kwargs = {} if sequence_kwargs is None else sequence_kwargs
        self.sweep_points = self.sequence_kwargs.get("sweep_points", None)
        self.mc_points = mc_points if mc_points is not None else [[], []]
        self.sweep_functions = sweep_functions
        self.force_2D_sweep = force_2D_sweep
        self.compression_seg_lim = compression_seg_lim
        self.channels_to_upload = []
        # The experiment_name might have been set by the user in kw or by a
        # child class as an attribute. Otherwise, the default None will
        # trigger guess_label to use the sequence name.
        self.experiment_name = kw.pop(
            'experiment_name', getattr(self, 'experiment_name', None))
        self.timestamp = None
        self.analysis = None

        # detector and sweep functions
        default_df_kwargs = {'det_get_values_kws':
                                 {'classified': self.classified,
                                  'correlated': False,
                                  'thresholded': True,
                                  'averaged': True}}
        self.df_kwargs = default_df_kwargs if df_kwargs is None else df_kwargs
        if df_name is not None:
            self.df_name = df_name
            if 'classif' in df_name:
                self.classified = True
        else:
            self.df_name = 'int_avg{}_det'.format('_classif' if self.classified else '')
        self.df = None

        # determine data type
        if "log" in self.df_name or not \
                self.df_kwargs.get('averaged', True):
            data_type = "singleshot"
        else:
            data_type = "averaged"

        self.exp_metadata.update(kw)
        self.exp_metadata.update({'classified_ro': self.classified,
                                  'cz_pulse_name': self.cz_pulse_name,
                                  'data_type': data_type})

    def create_meas_objs_list(self, meas_objs=None, **kwargs):
        """
        Creates a default list for self.meas_objs if meas_objs is not provided,
        and creates the list self.meas_obj_names.
        Args:
            meas_objs (list): a list of measurement objects (or None for
                default, which is self.qubits)
        """
        self.meas_objs = self.qubits if meas_objs is None else meas_objs
        self.meas_obj_names = [m.name for m in self.meas_objs]

    def _update_parameters(self, overwrite_dicts=True, **kwargs):
        """
        Update all attributes of the quantumExperiment class.
        Args:
            overwrite_dicts (bool): whether or not to overwrite
                attributes that are dictionaries. If False,
                then dictionaries are updated.
            **kwargs: any attribute of the QuantumExperiment class


        """
        for param_name, param_value in kwargs.items():
            if hasattr(self, param_name):
                if isinstance(param_value, dict) and not overwrite_dicts:
                    getattr(self, param_name).update(param_value)
                else:
                    setattr(self, param_name, param_value)

    def run_measurement(self, **kw):
        """
        Runs a measurement. Any keyword argument passes to this function that
        is also an attribute of the QuantumExperiment class will be updated
        before starting the experiment

        Returns:

        """
        self._update_parameters(**kw)
        assert self.meas_objs is not None, 'Cannot run measurement without ' \
                                           'measure objects.'
        if len(self.mc_points) == 1:
            self.mc_points = [self.mc_points[0], []]

        with temporary_value(*self.temporary_values):
            # Perpare all involved qubits. If not available, prepare
            # all measure objects.
            mos = self.qubits if self.qubits is not None else self.meas_objs
            for m in mos:
                m.prepare(drive=self.drive)

            # create/retrieve sequence to run
            self._prepare_sequences(self.sequences, self.sequence_function,
                                    self.sequence_kwargs)

            # configure measurement control (mc_points, detector functions)
            mode = self._configure_mc()

            self.guess_label(**kw)

            self.update_metadata()

            # run measurement
            try:
                self.MC.run(name=self.label, exp_metadata=self.exp_metadata,
                            mode=mode)
            except (Exception, KeyboardInterrupt) as e:
                self.extract_timestamp()
                raise e
        self.extract_timestamp()

    def update_metadata(self):
        # make sure that all metadata params are up to date
        for name in self._metadata_params:
            if hasattr(self, name):
                value = getattr(self, name)
                try:
                    if name in ('cal_points', 'sweep_points') and \
                            value is not None:
                        old_val = np.get_printoptions()['threshold']
                        np.set_printoptions(threshold=np.inf)
                        self.exp_metadata.update({name: repr(value)})
                        np.set_printoptions(threshold=old_val)
                    elif name in ('meas_objs', "qubits") and value is not None:
                        self.exp_metadata.update(
                            {name: [qb.name for qb in value]})
                    else:
                        self.exp_metadata.update({name: value})
                except Exception as e:
                    log.error(
                        f"Could not add {name} with value {value} to the "
                        f"metadata")
                    raise e

    def extract_timestamp(self):
        try:
            self.timestamp = self.MC.data_object._datemark + '_' \
                             + self.MC.data_object._timemark
        except Exception:
            pass  # if extraction fails, keep the old value (None from init)

    def guess_label(self, **kwargs):
        """
        Creates a default label.

        Returns:

        """
        if self.label is None:
            if self.experiment_name is None:
                self.experiment_name = self.sequences[0].name
            self.label = self.experiment_name
            _, qb_names = self.get_qubits(self.qubits)
            if self.dev is not None:
                self.label += self.dev.get_msmt_suffix(self.meas_obj_names)
            else:
                # guess_label is called from run_measurement -> we have qubits
                self.label += mqm.get_multi_qubit_msmt_suffix(self.meas_objs)

    def run_analysis(self, analysis_class=None, **kwargs):
        """
        Launches the analysis.
        Args:
            analysis_class: Class to use for the analysis
            **kwargs: keyword arguments passed to the analysis class

        Returns: analysis object

        """
        if analysis_class is None:
            analysis_class = ba.BaseDataAnalysis
        self.analysis = analysis_class(**kwargs)
        return self.analysis

    def autorun(self, **kw):
        if self.measure:
            self.run_measurement(**kw)
        if self.analyze:
            self.run_analysis(**kw)

    def serialize(self, omitted_attrs=('MC', 'device', 'qubits')):
        """
        Map a Quantum experiment to a large dict for hdf5 storage/pickle object,
        etc.
        Returns:

        """
        raise NotImplementedError()

    def _prepare_sequences(self, sequences=None, sequence_function=None,
                           sequence_kwargs=None):
        """
        Prepares/build sequences for a measurement.
        Args:
            sequences (list): list of sequences to run. Optional. If not given
                then a sequence_function from which the sequences can be created
                is required.
            sequence_function (callable): sequence function to generate sequences..
                Should return with one of the following formats:
                    - a list of sequences: valid if the first and second
                        sweepfunctions are  SegmentHardSweep and SegmentSoftsweep
                        respectively.
                    - a sequence: valid if the sweepfunction is SegmentHardsweep
                    - One of the following tuples:
                        (sequences, mc_points_tuple), where mc_points_tuple is a
                        tuple in which each entry corresponds to a dimension
                        of the sweep. This is the preferred option.
                        For backwards compatibility, the following two tuples are
                        also accepted:
                        (sequences, mc_points_first_dim, mc_points_2nd_dim)
                        (sequences, mc_points_first_dim)

            sequence_kwargs (dict): arguments to pass to the sequence function

        Returns:

        """

        if sequence_function is not None:
            # build sequence from function
            if sequence_kwargs is None:
                sequence_kwargs = {}

            seq_info = sequence_function(**sequence_kwargs)

            if isinstance(seq_info, list):
                self.sequences = seq_info
            elif isinstance(seq_info, Sequence):
                self.sequences = [seq_info]
            elif len(seq_info) == 3: # backwards compatible 2D sweep
                self.sequences, \
                    (self.mc_points[0], self.mc_points[1]) = seq_info
            elif len(seq_info) == 2:
                if np.ndim(seq_info[1]) == 1:
                    # backwards compatible 1D sweep
                    self.sequences, self.mc_points[0] = seq_info
                else:
                    self.sequences, self.mc_points = seq_info

            # ensure self.sequences is a list
            if np.ndim(self.sequences) == 0:
                self.sequences = [self.sequences]
        elif sequences is not None:
            self.sequences = sequences

        # check sequence
        assert len(self.sequences) != 0, "No sequence found."

    def _configure_mc(self, MC=None):
        """
        Configure the measurement control (self.MC) for the measurement.
        This includes setting the sweep points and the detector function.
        By default, SegmentHardSweep is the sweepfunction used for the first
        dimension and SegmentSoftSweep is the sweepfunction used for the second
        dimension. In case other sweepfunctions should be used, self.sweep_functions
        should be modified prior to the call of this function.

        Returns:
            mmnt_mode (str): "1D" or "2D"
        """
        # ensure measurement control is set
        self._set_MC(MC)

        # configure mc_points
        if len(self.mc_points[0]) == 0: # first dimension mc_points not yet set
            if self.sweep_functions[0] == awg_swf.SegmentHardSweep:
                # first dimension mc points can be retrieved as
                # ro_indices from sequence
                self.mc_points[0] = np.arange(self.sequences[0].n_acq_elements())
            else:
                raise ValueError("The first dimension of mc_points must be provided "
                                 "with sequence if the sweep function isn't "
                                 "'SegmentHardSweep'.")

        if len(self.sequences) > 1 and len(self.mc_points[1]) == 0:
            if self.sweep_functions[1] == awg_swf.SegmentSoftSweep:
                # 2nd dimension mc_points can be retrieved as sequence number
                self.mc_points[1] = np.arange(len(self.sequences))
            elif self.sweep_points is not None and len(self.sweep_points) > 1:
                # second dimension can be inferred from sweep points
                self.mc_points[1] = list(self.sweep_points[1].values())[0][0]
            else:
                raise ValueError("The second dimension of mc_points must be provided "
                                 "if the sweep function isn't 'SegmentSoftSweep' and"
                                 "no sweep_point object is given.")

        # force 2D sweep if needed (allow 1D sweep for backwards compatibility)
        if len(self.mc_points[1]) == 0 and self.force_2D_sweep:
            self.mc_points[1] = np.array([0]) # force 2d with singleton

        # set mc points
        if len(self.sequences) > 1:
            # compress 2D sweep
            if self.compression_seg_lim is not None:
                if self.sweep_functions == (awg_swf.SegmentHardSweep,
                                            awg_swf.SegmentSoftSweep):
                    self.sequences, self.mc_points[0], \
                    self.mc_points[1], cf = \
                        self.sequences[0].compress_2D_sweep(self.sequences,
                                                            self.compression_seg_lim,
                                                            True,
                                                            self.mc_points[0])
                    self.exp_metadata.update({'compression_factor': cf})
                else:
                    log.warning("Sequence compression currently does not support"
                                "sweep_functions different than (SegmentHardSweep,"
                                " SegmentSoftSweep). This could easily be implemented"
                                "by modifying Sequence.compress_2D_sweep to accept"
                                "mc_points and do the appropriate reshaping. Feel"
                                "free to make a pull request ;). Skipping compression"
                                "for now.")

        try:
            sweep_param_name = list(self.sweep_points[0])[0]
            unit = list(self.sweep_points[0].values())[0][2]
        except TypeError:
            sweep_param_name, unit = "None", ""
        sweep_func_1st_dim = self.sweep_functions[0](
            sequence=self.sequences[0], upload=self.upload,
            parameter_name=sweep_param_name,  unit=unit)

        self.MC.set_sweep_function(sweep_func_1st_dim)
        self.MC.set_sweep_points(self.mc_points[0])

        # set second dimension sweep function
        if len(self.mc_points[1]) > 0: # second dimension exists
            try:
                sweep_param_name = list(self.sweep_points[1])[0]
                unit = list(self.sweep_points[1].values())[0][2]
            except TypeError:
                sweep_param_name, unit = "None", ""
            if len(self.channels_to_upload) == 0:
                self.channels_to_upload = "all"
            if self.sweep_functions[1] == awg_swf.SegmentSoftSweep:
                self.MC.set_sweep_function_2D(self.sweep_functions[1](
                    sweep_func_1st_dim, self.sequences, sweep_param_name, unit,
                    self.channels_to_upload))
            else:
                # In case of an unknown sweep function type, it is assumed
                # that self.sweep_functions[1] has already been initialized
                # with all required parameters and can be directly passed to
                # MC.
                self.MC.set_sweep_function_2D(self.sweep_functions[1])

            self.MC.set_sweep_points_2D(self.mc_points[1])

        # check whether there is at least one measure object
        if len(self.meas_objs) == 0:
            raise ValueError('No measure objects provided. Cannot '
                             'configure detector functions')

        # Configure detector function
        # FIXME: this should be extended to meas_objs that are not qubits
        df = mqm.get_multiplexed_readout_detector_functions(
            self.meas_objs, **self.df_kwargs)[self.df_name]
        self.MC.set_detector_function(df)
        if self.dev is not None:
            meas_obj_value_names_map = self.dev.get_meas_obj_value_names_map(
                self.meas_objs, df)
        else:
            meas_obj_value_names_map = mqm.get_meas_obj_value_names_map(
                self.meas_objs, df)
        self.exp_metadata.update(
            {'meas_obj_value_names_map': meas_obj_value_names_map})

        if len(self.mc_points[1]) > 0:
            mmnt_mode = "2D"
        else:
            mmnt_mode = "1D"
        return mmnt_mode

    def _set_MC(self, MC=None):
        """
        Sets the measurement control and raises an error if no MC
        could be retrieved from device/qubits objects
        Args:
            MC (MeasurementControl):

        Returns:

        """
        if MC is not None:
            self.MC = MC
        elif self.MC is None:
            try:
                self.MC = self.dev.instr_mc.get_instr()
            except AttributeError:
                try:
                    self.MC = self.meas_objs[0].instr_mc.get_instr()
                except (AttributeError, IndexError):
                    raise ValueError("The Measurement Control (MC) could not "
                                     "be retrieved because no Device/measure "
                                     "objects were found. Pass the MC to "
                                     "run_measurement() or set the MC attribute"
                                     " of the QuantumExperiment instance.")

    # def __setattr__(self, name, value):
    #     """
    #     Observes attributes which are set to this class. If they are in the
    #     _metadata_params then they are automatically added to the experimental
    #     metadata
    #     Args:
    #         name:
    #         value:
    #
    #     Returns:
    #
    #     """
    #     if name in self._metadata_params:
    #         try:
    #             if name in 'cal_points' and value is not None:
    #                 self.exp_metadata.update({name: repr(value)})
    #             elif name in ('meas_objs', "qubits") and value is not None:
    #                 self.exp_metadata.update({name: [qb.name for qb in value]})
    #             else:
    #                 self.exp_metadata.update({name: value})
    #         except Exception as e:
    #             log.error(f"Could not add {name} with value {value} to the "
    #                       f"metadata")
    #             raise e
    #
    #     self.__dict__[name] = value

    def __repr__(self):
        return f"QuantumExperiment(dev={self.dev}, qubits={self.qubits})"