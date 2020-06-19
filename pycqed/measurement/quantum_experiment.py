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
    _metadata_params = {'cal_points', 'preparation_params', 'sweep_points',
                        'channel_map', 'ro_qubits'}

    def __init__(self, dev, qubits=None, ro_qubits=None, classified=False,
                 label=None, exp_metadata=None, upload=True, measure=True,
                 analyze=True, temporary_values=((), ()), drive="timedomain",
                 sequences=(), sequence_function=None,
                 sequence_kwargs=None, df_kwargs=None, df_name=None,
                 mc_points=((), ()), sweep_functions=(awg_swf.SegmentHardSweep,
                                                      awg_swf.SegmentSoftSweep),
                 compression_seg_lim=None, force_2D_sweep=True, **kw):
        super().__init__(dev=dev, qubits=qubits, **kw)

        self.exp_metadata = exp_metadata
        if self.exp_metadata is None:
            self.exp_metadata = {}

        self.ro_qubits = self.qubits if ro_qubits is None else ro_qubits
        self.MC = dev.instr_mc.get_instr()

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
        self.mc_points = mc_points
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
        if df_name is None:
            self.df_name = df_name
        if self.df_name is None:
            self.df_name = 'int_avg{}_det'.format('_classif' if self.classified else '')



        self.exp_metadata.update(kw)
        self.exp_metadata.update({'classified_ro': self.classified})

    def _update_parameters(self, overwrite_dicts=True, **kwargs):
        """
        """
        for param_name, param_value in kwargs:
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

        # combine preparation dictionaries
        qb_names = self.dev.get_qubits(self.qubits, "str")
        self.preparation_params = self.get_prep_params(qb_names)

        # only prepare read out qubits
        for qb in self.ro_qubits:
            qb.prepare(drive=self.drive)

        # create/retrieve sequence to run
        self._prepare_sequences(self.sequences, self.sequence_function,
                                self.sequence_kwargs)

        # configure measurement control (mc_points, detector functions)
        self._configure_mc()

        if self.label is None:
            self.label = f'{self.sequences[0].name}_{",".join(qb_names)}'

        # run measurement
        with temporary_value(*self.temporary_values):
            self.MC.run(name=self.label, exp_metadata=self.exp_metadata)

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
        assert len(self.sequences) == 0, "No sequence found."

    def _configure_mc(self):
        """
        Configure the measurement control (self.MC) for the measurement.
        This includes setting the sweep points and the detector function.
        By default, SegmentHardSweep is the sweepfunction used for the first
        dimension and SegmentSoftSweep is the sweepfunction used for the second
        dimension. In case other sweepfunctions should be used, self.sweep_functions
        should be modified prior to the call of this function.

        """

        # configure mc_points
        if len(self.mc_points[0]) == 0: # first dimension mc_points not yet set
            if isinstance(self.sweep_functions[0], awg_swf.SegmentHardSweep):
                # first dimension mc points can be retrieved as
                # ro_indices from sequence
                self.mc_points[0] = np.arange(self.sequences[0].n_acq_elements())
            else:
                raise ValueError("The first dimension of mc_points must be provided "
                                 "with sequence if the sweep function isn't "
                                 "'SegmentHardSweep'.")

        if len(self.sequences) > 1 and len(self.mc_points[1]) == 0:
            if isinstance(self.sweep_functions[1], awg_swf.SegmentSoftSweep):
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
                self.sequences, self.mc_points[0], \
                self.mc_points[1], cf = \
                    self.sequences[0].compress_2D_sweep(self.sequences,
                                                        self.compression_seg_lim)
                self.exp_metadata.update({'compression_factor': cf})

        # if 2D, then upload is taken care of by second sweep function
        upload_1st_dim = self.upload if len(self.mc_points[1]) > 0 else False

        try:
            sweep_param_name = list(self.sweep_points[0])[0]
            unit = list(self.sweep_points[0].values())[0][2]
        except AttributeError:
            sweep_param_name, unit = "", ""
        sweep_func_1st_dim = self.sweep_functions[0](
            sequence=self.sequences[0], upload=upload_1st_dim,
            parameter_name=sweep_param_name,  unit=unit)

        self.MC.set_sweep_function(sweep_func_1st_dim)
        self.MC.set_sweep_points(self.mc_points[0])

        # set second dimension sweep function
        if len(self.mc_points[1]) > 0: # second dimension exists
            try:
                sweep_param_name = list(self.sweep_points[1])[0]
                unit = list(self.sweep_points[1].values())[0][2]
            except AttributeError:
                sweep_param_name, unit = "", ""
            self.MC.set_sweep_function_2D(self.sweep_functions[1](
                self.upload, self.sequences, sweep_param_name, unit,
                channels_to_upload=self.channels_to_upload))
            self.MC.set_sweep_points_2D(self.mc_points[1])

        # Set detector function
        df = get_multiplexed_readout_detector_functions(
            self.ro_qubits, **self.df_kwargs)[self.df_name]
        self.MC.set_detector_function(df)

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
        return f"QuantumExperiment(dev='{self.dev.name}')"