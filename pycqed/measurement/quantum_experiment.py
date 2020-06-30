import numpy as np

from pycqed.measurement.waveform_control.sequence import Sequence
from pycqed.utilities.general import temporary_value
from pycqed.measurement.waveform_control.circuit_builder import CircuitBuilder
import pycqed.measurement.awg_sweep_functions as awg_swf
from pycqed.measurement import multi_qubit_module as mqm
from pycqed.measurement.multi_qubit_module import \
    get_multiplexed_readout_detector_functions
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
                        'channel_map', 'ro_qubits'}

    def __init__(self, dev=None, qubits=None, operation_dict=None,
                 ro_qubits=None, classified=False, MC=None,
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
            ro_qubits (list): list of qubits to be read out (i.e. for which the detector
                functions will be prepared). Defaults to self.qubits (attribute set by
                circuitBuilder). Required for run_measurement() when qubits is None.
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

        # if no qubits/devices are provided, use empty list to skip iterations
        #  over qubit lists
        if qubits is None and dev is None:
            qubits = []
        super().__init__(dev=dev, qubits=qubits, operation_dict=operation_dict,
                         **kw)

        self.exp_metadata = exp_metadata
        if self.exp_metadata is None:
            self.exp_metadata = {}

        self.ro_qubits = self.qubits if ro_qubits is None else ro_qubits
        self.MC = MC

        self.classified = classified
        self.label = label
        self.upload = upload
        self.measure = measure
        self.temporary_values = temporary_values
        self.analyze = analyze
        self.drive = drive

        self.sequences = sequences
        self.sequence_function = sequence_function
        self.sequence_kwargs = {} if sequence_kwargs is None else sequence_kwargs
        self.sweep_points = self.sequence_kwargs.get("sweep_points", None)
        self.mc_points = mc_points if mc_points is not None else ([], [])
        self.sweep_functions = sweep_functions
        self.force_2D_sweep = force_2D_sweep
        self.compression_seg_lim = compression_seg_lim
        self.channels_to_upload = []

        # detector and sweep functions
        default_df_kwargs = {'classified': self.classified,
                             'correlated': False,
                             'thresholded': True,
                             'averaged': True}
        self.df_kwargs = default_df_kwargs if df_kwargs is None else df_kwargs
        if df_name is not None:
            self.df_name = df_name
        else:
            self.df_name = 'int_avg{}_det'.format('_classif' if self.classified else '')

        self.exp_metadata.update(kw)
        self.exp_metadata.update({'classified_ro': self.classified})

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

        with temporary_value(*self.temporary_values):
            # combine preparation dictionaries
            _, qb_names = self.get_qubits(self.qubits)

            # only prepare read out qubits
            for qb in self.ro_qubits:
                qb.prepare(drive=self.drive)

            # create/retrieve sequence to run
            self._prepare_sequences(self.sequences, self.sequence_function,
                                    self.sequence_kwargs)

            # configure measurement control (mc_points, detector functions)
            mode = self._configure_mc()

            if self.label is None:
                self.label = f'{self.sequences[0].name}_{",".join(qb_names)}'

            # run measurement
            self.MC.run(name=self.label, exp_metadata=self.exp_metadata,
                        mode=mode)

    def run_analysis(self, analysis_class=None, **kwargs):
        """
        Launches the analysis.
        Args:
            analysis_class: Class to use for the analysis
            **kwargs: keyword arguments passed to the analysis class

        Returns: analysis object

        """
        if self.analyze:
            if analysis_class is None:
                analysis_class = ba.BaseDataAnalysis
            self.analysis = analysis_class(**kwargs)
            return self.analysis

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
                                                            self.compression_seg_lim)
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
            if self.sweep_functions[1] != awg_swf.SegmentSoftSweep:
                raise NotImplementedError(
                    "2D sweeps with sweepfunction different than "
                    "SegmentSoftsweep are not yet supported (but "
                    "the framework should allow to implement it "
                    "quite easily: we should just distinguish which"
                    "arguments should be passed in which case ("
                    "for now, different soft sweep functions accept "
                    "different arguments...) to self.sweep_functions[1]"
                    ". Feel free to give it a go and make a pull "
                    "request ;)")
            self.MC.set_sweep_function_2D(self.sweep_functions[1](
                sweep_func_1st_dim, self.sequences, sweep_param_name, unit,
                self.channels_to_upload))

            self.MC.set_sweep_points_2D(self.mc_points[1])

        # check whether there is at least one readout qubit
        if len(self.ro_qubits) == 0:
            raise ValueError('No readout qubits provided. Cannot '
                             'configure detector functions')

        # Configure detector function
        df = get_multiplexed_readout_detector_functions(
            self.ro_qubits, det_get_values_kws=self.df_kwargs)[self.df_name]
        self.MC.set_detector_function(df)

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
                    self.MC = self.qubits[0].instr_mc.get_instr()
                except (AttributeError, IndexError):
                    raise ValueError("The Measurement Control (MC) could not "
                                     "be retrieved because no Device/qubit "
                                     "objects were found. Pass the MC to "
                                     "run_measurement() or set the MC attribute"
                                     " of the QuantumExperiment instance.")

    def __setattr__(self, name, value):
        """
        Observes attributes which are set to this class. If they are in the
        _metadata_params then they are automatically added to the experimental
        metadata
        Args:
            name:
            value:

        Returns:

        """
        if name in self._metadata_params:
            try:
                if name in ('cal_points', 'sweep_points') and value is not None:
                    self.exp_metadata.update({name: repr(value)})
                elif name in ('ro_qubits', "qubits") and value is not None:
                    self.exp_metadata.update({name: [qb.name for qb in value]})
                else:
                    self.exp_metadata.update({name: value})
            except Exception as e:
                log.error(f"Could not add {name} with value {value} to the "
                          f"metadata")
                raise e

        self.__dict__[name] = value

    def __repr__(self):
        return f"QuantumExperiment(dev={self.dev}, qubits={self.qubits})"