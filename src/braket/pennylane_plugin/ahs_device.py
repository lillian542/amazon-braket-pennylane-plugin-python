from functools import partial
import numpy as np

from braket.aws import AwsDevice
from braket.devices import LocalSimulator
from braket.ahs.atom_arrangement import AtomArrangement
from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
from braket.ahs.driving_field import DrivingField
from braket.timings.time_series import TimeSeries

from pennylane import QubitDevice
from pennylane._version import __version__
from pennylane.pulse.rydberg_hamiltonian import RydbergHamiltonian, RydbergPulse


class BraketAhsDevice(QubitDevice):
    """Abstract Amazon Braket device for analogue hamiltonian simulation with PennyLane.

    Args:
        wires (int or Iterable[Number, str]]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers
            (i.e., ``[-1, 0, 2]``) or strings (``['ancilla', 'q1', 'q2']``).
        shots (int): Number of executions to run to aquire measurements. Defaults to 100.
    """

    name = "Braket AHS PennyLane plugin"
    pennylane_requires = ">=0.29.0"
    version = __version__
    author = "Xanadu Inc."

    operations = {"ParametrizedEvolution"}

    def __init__(
            self,
            wires,
            device,
            *,
            shots=100):

        self._device = device
        super().__init__(wires=wires, shots=shots)

        self.ahs_program = None
        self.samples = None

    def apply(self, operations, **kwargs):
        """Convert the pulse operation to an AHS program and run on the connected device"""

        self._validate_operations(operations)

        ev_op = operations[0]  # only one!

        ahs_program = self.create_ahs_program(ev_op)

        task = self._run_task(ahs_program)

        self.samples = task.result()

    def _run_task(self, ahs_program):
        raise NotImplementedError("Running a task not implemented for the base class")

    def create_ahs_program(self, evolution):
        """Create AHS program for upload to hardware from a ParametrizedEvolution

        Returns:
            AnalogHamiltonianSimulation: a program containing the register and drive
                information for running an AHS task on simulation or hardware"""

        params = evolution.parameters

        # sets self.pulses to be the evaluated pulses (now only a function of time)
        self._evaluate_pulses(evolution, params)
        self._create_register(evolution.H.register)

        time_interval = evolution.t

        # no gurarentee that global drive is index 0 once we start allowing more just global drive
        drive = self._convert_pulse_to_driving_field(self.pulses[0], time_interval)

        ahs_program = AnalogHamiltonianSimulation(register=self.register, hamiltonian=drive)

        self.ahs_program = ahs_program

        return ahs_program

    def generate_samples(self):
        r"""Returns the computational basis samples measured for all wires.

        Returns:
             array[complex]: array of samples in the shape ``(dev.shots, dev.num_wires)``
        """
        return [self._result_to_sample_output(res) for res in self.samples.measurements]

    def _validate_operations(self, operations):
        """Confirms that the list of operations provided contains a single ParametrizedEvolution
        from a RydbergHamiltonian with only a single, global pulse"""

        if len(operations) > 1:
            raise NotImplementedError(
                f"Support for multiple ParametrizedEvolution operators in a single circuit is "
                f"not yet implemented. Recieved {len(operations)} operators.")

        ev_op = operations[0]  # only one!

        if not isinstance(ev_op.H, RydbergHamiltonian):
            raise RuntimeError(
                f"Expected a RydbergHamiltonian instance for interfacing with the device, but "
                f"recieved {type(ev_op.H)}.")

        self._validate_pulses(ev_op.H.pulses)

    def _validate_pulses(self, pulses):
        raise NotImplementedError("Validation of pulses not implemented in the base class")

    # could be static method or just completely separate from the class
    def _create_register(self, coordinates):
        """Create an AtomArrangement to describe the atom layout from the coordinates in the ParametrizedEvolution"""
        register = AtomArrangement()
        for [x, y] in coordinates:
            register.add([x * 1e-6, y * 1e-6])  # PL asks users to specify in um, Braket expects SI units

        self.register = register

    def _evaluate_pulses(self, ev_op, params):
        """Feeds in the parameters in order to partially evaluate the callables (amplitude, phase and/or detuning)
        describing the pulses, so they are only a function of time

        Args:
            ev_op(ParametrizedEvolution): the operator containing the pulses to be evaluated
            params(list): a list of the parameters to be passed to the respective callables
        """

        # ToDo: what happens if H.pulses is an empty list? I.e. if only interaction term
        pulses = ev_op.H.pulses
        coeffs = ev_op.H.coeffs_parametrized

        evaluated_coeffs = [partial(fn, param) for fn, param in zip(coeffs, params)]

        idx = 0

        for pulse in pulses:
            if callable(pulse.amplitude):
                pulse.amplitude = evaluated_coeffs[idx]
                idx += 1

            if callable(pulse.detuning):
                pulse.detuning = evaluated_coeffs[idx]
                idx += 1

            if callable(pulse.phase):
                pulse.phase = evaluated_coeffs[idx]
                idx += 1

        self.pulses = pulses

    # ToDo: should the 50ns number instead be retrieved from the HW dict (when connected to HW)?
    # could be static
    def _get_sample_times(self, time_interval):
        """Takes a time interval and returns an array of times with a minimum of 50ns spacing"""
        # time_interval from PL is in microseconds
        interval_ns = np.array(time_interval) * 1e3
        timespan = interval_ns[1] - interval_ns[0]

        # number of points must ensure at least 50ns between sample points
        num_points = int(timespan // 50)

        start = interval_ns[0]
        end = interval_ns[1]

        # we want an integer number of nanoseconds
        times = np.linspace(start, end, num_points, dtype=int)

        # we return time in seconds
        return times / 1e9

    # could be static?
    def _convert_to_time_series(self, pulse_parameter, time_points, scaling_factor=1):
        """Converts pulse information into a TimeSeries

        Args:
            pulse_parameter(Union[float, Callable]): a physical parameter (pulse, amplitude or detuning) of the
                pulse. If this is a callalbe, it has alreayd been partially evaluated, such that it is only a
                function of time.
            time_points(array): the times where parameters will be set in the TimeSeries, specified in seconds
            scaling_factor(float): A multiplication factor for the pulse_parameter where relevant to convert
                between units. Defaults to 1.

        Returns:
            TimeSeries: a description of setpoints and corresponding times
        """

        ts = TimeSeries()

        if callable(pulse_parameter):
            # convert time to microseconds to evaluate values - this is the expected unit for the PL functions
            vals = [float(pulse_parameter(t * 1e6))*scaling_factor for t in time_points]
        else:
            vals = [pulse_parameter for t in time_points]

        for t, v in zip(time_points, vals):
            ts.put(t, v)

        return ts

    # could be static?
    def _convert_pulse_to_driving_field(self, pulse, time_interval):
        """Converts a ``RydbergPulse`` from PennyLane describing a global drive to a ``DrivingField``
        from Braket AHS

        Args:
            pulse[RydbergPulse]: a dataclass object containing amplitude, phase and detuning information
            time_interval(array[Number, Number]]): The start and end time for the applied pulse

        Returns:
            DrivingField: the object representing the global drive for the AnalogueHamiltonianSimulation object
        """

        time_points = self._get_sample_times(time_interval)

        # scaling factor for amplitude and detunig convert MHz (expected PL input) to rad/s (upload units)
        amplitude = self._convert_to_time_series(pulse.amplitude, time_points, scaling_factor=2*np.pi*1e6)
        detuning = self._convert_to_time_series(pulse.detuning, time_points, scaling_factor=2*np.pi*1e6)
        phase = self._convert_to_time_series(pulse.phase, time_points)

        drive = DrivingField(amplitude=amplitude, detuning=detuning, phase=phase)

        return drive

    @staticmethod
    def _result_to_sample_output(res):
        """This function converts a single shot of the QuEra measurement results to 0 (ground), 1 (excited)
        and NaN (failed to measure) for all atoms in the result.

        The QuEra results are summarized via 3 values: status, pre_sequence, and post_sequence.

        Status is success or fail. The pre_sequence is 1 if an atom in the ground state was successfully
        initialized, and 0 otherwise. The post_sequence is 1 if an atom in the ground state was measured,
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


class BraketAquilaDevice(BraketAhsDevice):
    """Amazon Braket AHS device for QuEra Aquila hardware for PennyLane.

    Args:
        wires (int or Iterable[Number, str]]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers
            (i.e., ``[-1, 0, 2]``) or strings (``['ancilla', 'q1', 'q2']``).
        shots (int): Number of executions to run to aquire measurements. Defaults to 100.
    """

    name = "Braket QuEra Aquila PennyLane plugin"
    short_name = "braket.aws.aquila"

    ARN_NR = "arn:aws:braket:us-east-1::device/qpu/quera/Aquila"

    def __init__(
            self,
            wires,
            *,
            shots=100):

        dev = AwsDevice(self.ARN_NR)
        super().__init__(wires=wires, device=dev, shots=shots)

        self.ahs_program = None
        self.samples = None

    @property
    def hardware_capabilities(self):
        """Dictionary of hardware capabilities for the Aquila device"""
        return self._device.properties.paradigm.dict()

    def _run_task(self, ahs_program):
        discretized_ahs_program = ahs_program.discretize(self._device)
        task = self._device.run(discretized_ahs_program, shots=self.shots)

    def _validate_pulses(self, pulses):

        if len(pulses) > 1:
            raise NotImplementedError(
                f"Multiple pulses in a Rydberg Hamiltonian are not currently supported on "
                f"hardware. Recieved {len(pulses)} pulses.")

        if pulses[0].wires != self.wires:
            raise NotImplementedError(
                f"Only global drive is currently supported on hardware. Found drive defined for subset "
                f"{[pulses[0].wires]} of all wires [{self.wires}]")


class BraketLocalAquilaDevice(BraketAhsDevice):
    """Amazon Braket LocalSimulator AHS device for PennyLane.

    Args:
        wires (int or Iterable[Number, str]]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers
            (i.e., ``[-1, 0, 2]``) or strings (``['ancilla', 'q1', 'q2']``).
        shots (int): Number of executions to run to aquire measurements. Defaults to 100.
    """

    name = "Braket QuEra Aquila PennyLane plugin"
    short_name = "braket.local.aquila"

    def __init__(
            self,
            wires,
            *,
            shots=100):

        dev = LocalSimulator("braket_ahs")
        print(shots)
        super().__init__(wires=wires, device=dev, shots=shots)

    def _run_task(self, ahs_program):
        task = self._device.run(self.ahs_program, shots=self.shots, steps=100)
        return task

    def _validate_pulses(self, pulses):

        # ToDo: allow local drive

        if len(pulses) > 1:
            raise NotImplementedError(
                f"Multiple pulses in a Rydberg Hamiltonian are not currently supported on "
                f"hardware. Recieved {len(pulses)} pulses.")

        if pulses[0].wires != self.wires:
            raise NotImplementedError(
                f"Only global drive is currently supported on hardware. Found drive defined for subset "
                f"{[pulses[0].wires]} of all wires [{self.wires}]")