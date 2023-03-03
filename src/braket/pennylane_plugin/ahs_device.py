from functools import partial

from braket.aws import AwsDevice
from braket.devices import LocalSimulator
from braket.ahs.atom_arrangement import AtomArrangement
from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
from braket.ahs.driving_field import DrivingField
from braket.timings.time_series import TimeSeries

from pennylane import QubitDevice
from pennylane._version import __version__
from pennylane.pulse import RydbergHamiltonian


class QuEraAquila(QubitDevice):
    """Amazon Braket AwsDevice interface for analogue hamiltonian simulation on QuEra Aquila for PennyLane."""

    name = "Braket QuEra Aquila PennyLane plugin"
    short_name = "quera.aquila"
    pennylane_requires = ">=0.29.0"
    version = __version__
    author = "Xanadu Inc."

    operations = {"ParametrizedEvolution"}

    ARN_NR = "arn:aws:braket:us-east-1::device/qpu/quera/Aquila"

    def __init__(
            self,
            wires,
            *,
            shots=100,
            simulator=True):

        self.shots = shots
        self.simulator = simulator

        super().__init__(wires=wires, shots=shots)

        if simulator:  # this should be a separate thing long-term, this is just for ease of development and testing
            self._device = LocalSimulator("braket_ahs")
        else:
            self._device = AwsDevice(self.ARN_NR)

        self.circuit = []
        self.ahs_program = None
        self.samples = None

    def reset(self):
        """Reset the device and reload configurations."""
        self.circuit = []
        self.ahs_program = None
        self.samples = None

    @property
    def hardware_capabilities(self):
        return self._device.properties.paradigm.dict()

    def apply(self, operations, **kwargs):

        self._validate_operations(operations)

        ev_op = operations[0]  # only one!

        ahs_program = self.create_ahs_program(ev_op)

        if self.simulator:
            task = self._device.run(ahs_program, shots=self.shots, steps=100)

        else:
            discretized_ahs_program = ahs_program.discretize(self._device)
            task = self._device.run(discretized_ahs_program, shots=self.shots)

        self.samples = task.result()

    def create_ahs_program(self, evolution):
        """Create AHS program for upload to hardware from a ParametrizedEvolution"""

        params = evolution.parameters

        # sets self.pulses to be the evaluated pulses (now only a function of time)
        self._evaluate_pulses(evolution, params)
        # creates AtomArrangement object from H.register
        self._create_register(evolution.H.register)

        # if multiple operators were allowed, there would be multiple time intervals here.
        # need to think about constraints in terms of how/whether these can overlap, as well as
        time_interval = evolution.t

        # no gurarentee that global drive is index 0 once we start allowing more just global drive
        drive = self._convert_pulse_to_driving_field(self.pulses[0], time_interval)

        # currently not being tested, but this will not be the correct way to handle this
        # I don't think multiple global drives should be an option in the same time_interval
        # and if additional drives aren't global drives, they will need to be implemented differently
        # (i.e. check only detuning is defined and use braket.ahs.ShiftingField instead of DrivingField)
        # if we wait to implement local control until outside PL until it's available on HW, this is not needed yet
        if len(self.pulses) > 1:
            for pulse in self.pulses[1:]:
                drive += self._convert_pulse_to_driving_field(pulse, time_interval)

        ahs_program = AnalogHamiltonianSimulation(register=self.register, hamiltonian=drive)

        self.ahs_program = ahs_program

        return ahs_program

    def generate_samples(self):
        return [self._res_to_sample_output(res) for res in self.samples.measurements]

    def _validate_operations(self, operations):

        if len(operations) > 1:
            raise NotImplementedError(
                f"Support for multiple ParametrizedEvolution operators in a single circuit is "
                f"not yet implemented. Recieved {len(operations)} operators.")

        ev_op = operations[0]  # only one!

        if not isinstance(ev_op.H, RydbergHamiltonian):
            raise RuntimeError(
                f"Expected a RydbergHamiltonian instance for interfacing with the device, but "
                f"recieved {type(ev_op.H)}.")

        if len(ev_op.H.pulses) > 1:
            raise NotImplementedError(
                f"Support for multiple pulses in a Rydberg Hamiltonian not currently supported on "
                f"hardware or in simulation")
            #  ToDo: add local drive (detuning only, uses different mechanism)
            #  Could be skipped for now as it's an upcoming feature on hardware, currently only supported in simulation)

        if ev_op.H.pulses[0].wires != self.wires:
            raise NotImplementedError(
                f"Only global drive is currently supported. Found drive defined for subset "
                f"{[ev_op.H.pulses[0].wires]} of all wires [{self.wires}]")

    # could be static method or just completely separate from the class
    def _create_register(self, coordinates):
        register = AtomArrangement()
        for [x, y] in coordinates:
            register.add([x * 1e-6, y * 1e-6])  # we ask users to specify in um, braket expects SI units

        self.register = register

    def _evaluate_pulses(self, ev_op, params):

        # ToDo: what happens if H.pulses is an empty list? I.e. if only interaction term
        pulses = ev_op.H.pulses
        coeffs = ev_op.H.coeffs_parametrized

        evaluated_coeffs = [partial(fn, param) for fn, param in zip(coeffs, params)]

        idx = 0

        for pulse in pulses:
            # is this a dangerous choice? (I think so.) Can these orders be easily mixed up?
            # yes but same problem with coeffs, params? think about this after resolving phase-as-a-callable
            if callable(pulse.rabi):
                pulse.rabi = evaluated_coeffs[idx]
                idx += 1

            if callable(pulse.detuning):
                pulse.detuning = evaluated_coeffs[idx]
                idx += 1

            if callable(pulse.phase):
                pulse.phase = evaluated_coeffs[idx]
                idx += 1

        self.pulses = pulses

    # could be static
    def _get_sample_times(self, time_interval):
        # assuming users in PL set time in us, we convert to ns
        # maybe they should just use ns? We will need to decide on a convention.
        interval_ns = np.array(time_interval) * 1e3
        timespan = interval_ns[1] - interval_ns[0]

        # number of points must ensure at least 50ns between sample points
        num_points = int(timespan // 50)

        start = interval_ns[0]
        end = interval_ns[1]

        # we want an integer number of nanoseconds
        times = np.linspace(start, end, num_points, dtype=int)  # we want an integer number of nanoseconds

        # we return time in seconds
        return times / 1e9

    # could be static?
    def _convert_to_time_series(self, coeff, time_points):

        ts = TimeSeries()

        if callable(coeff):
            # time is now in ns, but fn is defined assuming time in us? maybe we should commit to ns
            vals = [float(coeff(t * 1e6)) for t in time_points]
        else:
            vals = [coeff for t in time_points]

        for t, v in zip(time_points, vals):
            ts.put(t, v)

        return ts

    # could be static?
    def _convert_pulse_to_driving_field(self, pulse, time_interval):

        time_points = self._get_sample_times(time_interval)

        # do we need to do any unit conversions between MHz and rad/s?
        # we tell users in the docstring to specify in MHz, but where do we assume MHz mathematically in simulation?
        amplitude = self._convert_to_time_series(pulse.rabi, time_points)
        detuning = self._convert_to_time_series(pulse.detuning, time_points)
        phase = self._convert_to_time_series(pulse.phase, time_points)

        drive = DrivingField(amplitude=amplitude, detuning=detuning, phase=phase)

        return drive

    @staticmethod
    def _result_to_sample_output(res):
        """This function converts a single shot of the QuEra measurement results to 0 (ground), 1 (excited)
        and NaN (failed to measure) for all atoms in the result.

        The QuEra results are summarized via 3 values: status, pre_sequence, and post_sequence.

        Status is success or fail. The pre_sequence is 1 if an atom in the ground state was successfully
        initialized, and 0 otherwise. The post_sequence is 1 if an atom in the ground state was measured, \
        and 0 otherwise. Comparison of pre_sequence and post_sequence reveals one of 3 possible outcomes:

        0 --> 0: Atom failed to be placed, no measurement (no atom in the ground state either before or after)
        1 --> 0: Atom initialized, measured in Rydberg state (atom in ground state detected before, but not after)
        1 --> 1: Atom initialized, measured in ground state (atom in ground state detected both before and after)"""

        # if entire measurement failed, all NaN
        if not res.status.value.lower() == 'success':
            return [np.NaN, np.NaN, np.NaN]

        # if a single atom failed to initialize, NaN for that individual measurement
        pre_sequence = [i if i else np.NaN for i in res.pre_sequence]

        # set entry to 0 if ground state is measured, 1 if excited state is measured, NaN if measurement failed
        return np.array(pre_sequence - res.post_sequence)
