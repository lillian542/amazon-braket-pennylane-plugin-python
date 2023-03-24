from functools import partial
from typing import Iterable, Union
import numpy as np

from braket.aws import AwsDevice
from braket.devices import Device, LocalSimulator
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
        device (Device): The Amazon Braket device to use with PennyLane.
        shots (int): Number of executions to run to aquire measurements. Defaults to 100.
    """

    name = "Braket AHS PennyLane plugin"
    pennylane_requires = ">=0.29.0"
    version = __version__
    author = "Xanadu Inc."

    operations = {"ParametrizedEvolution"}

    def __init__(self, wires: Union[int, Iterable], device: Device, *, shots=100):
        if not shots:
            raise RuntimeError(f"This device requires shots. Recieved shots={shots}")
        self._device = device

        super().__init__(wires=wires, shots=shots)

        self.register = None
        self.ahs_program = None
        self.samples = None

    @property
    def settings(self):
        return {"interaction_coeff": 862690}  # MHz x um^6

    def apply(self, operations, **kwargs):
        """Convert the pulse operation to an AHS program and run on the connected device"""

        if not np.all([op.name in self.operations for op in operations]):
            raise NotImplementedError(
                "Device {self.short_name} expected only operations "
                "{self.operations} but recieved {operations}"
            )

        self._validate_operations(operations)

        ev_op = operations[0]  # only one!

        self._validate_pulses(ev_op.H.pulses)

        ahs_program = self.create_ahs_program(ev_op)

        task = self._run_task(ahs_program)

        self.samples = task.result()

    def _run_task(self, ahs_program):
        raise NotImplementedError("Running a task not implemented for the base class")

    def create_ahs_program(self, evolution):
        """Create AHS program for upload to hardware from a ParametrizedEvolution

        Args:
            evolution (ParametrizedEvolution): the PennyLane operator describing the pulse
                to be converted into an Analogue Hamiltonian Simulation program

        Returns:
            AnalogHamiltonianSimulation: a program containing the register and drive
                information for running an AHS task on simulation or hardware"""

        # sets self.pulses to be the evaluated pulses (now only a function of time)
        self._evaluate_pulses(evolution)
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
                f"not yet implemented. Recieved {len(operations)} operators."
            )

        ev_op = operations[0]  # only one!

        if not isinstance(ev_op.H, RydbergHamiltonian):
            raise RuntimeError(
                f"Expected a RydbergHamiltonian instance for interfacing with the device, but "
                f"recieved {type(ev_op.H)}."
            )

        if not set(ev_op.wires) == set(self.wires):
            raise RuntimeError(
                f"Device contains wires {self.wires}, but received a `ParametrizedEvolution` operator "
                f"working on wires {ev_op.wires}. Device wires must match wires of the evolution."
            )

        if len(ev_op.H.register) != len(self.wires):
            raise RuntimeError(
                f"The defined interaction term has register {ev_op.H.register} of length "
                f"{len(ev_op.H.register)}, which does not match the number of wires on the device "
                f"({len(self.wires)})"
            )

    def _validate_pulses(self, pulses):
        """Confirms that the list of RydbergPulses describes a single, global pulse

        Args:
            pulses: List of RydbergPulses

        Raises:
            RuntimeError, NotImplementedError"""

        if not pulses:
            raise RuntimeError("No pulses found in the ParametrizedEvolution")

        if len(pulses) > 1:
            raise NotImplementedError(
                f"Multiple pulses in a Rydberg Hamiltonian are not currently supported on "
                f"hardware. Recieved {len(pulses)} pulses."
            )

        if pulses[0].wires != self.wires:
            raise NotImplementedError(
                f"Only global drive is currently supported. Found drive defined for subset "
                f"{[pulses[0].wires]} of all wires [{self.wires}]"
            )

    def _create_register(self, coordinates):
        """Create an AtomArrangement to describe the atom layout from the coordinates in the
        ParametrizedEvolution, and saves it as self.register"""

        register = AtomArrangement()
        for [x, y] in coordinates:
            # PL asks users to specify in um, Braket expects SI units
            register.add([x * 1e-6, y * 1e-6])

        self.register = register

    def _evaluate_pulses(self, ev_op):
        """Feeds in the parameters in order to partially evaluate the callables (amplitude, phase and/or detuning)
        describing the pulses, so they are only a function of time. Saves the pulses on the device as `dev.pulses`.

        Args:
            ev_op(ParametrizedEvolution): the operator containing the pulses to be evaluated
            params(list): a list of the parameters to be passed to the respective callables
        """

        params = ev_op.parameters
        pulses = ev_op.H.pulses

        evaluated_pulses = []
        idx = 0

        for pulse in pulses:
            amplitude = pulse.amplitude
            if callable(pulse.amplitude):
                amplitude = partial(pulse.amplitude, params[idx])
                idx += 1

            phase = pulse.phase
            if callable(pulse.phase):
                phase = partial(pulse.phase, params[idx])
                idx += 1

            detuning = pulse.detuning
            if callable(pulse.detuning):
                detuning = partial(pulse.detuning, params[idx])
                idx += 1

            evaluated_pulses.append(
                RydbergPulse(amplitude=amplitude, phase=phase, detuning=detuning, wires=pulse.wires)
            )

        self.pulses = evaluated_pulses

    def _get_sample_times(self, time_interval):
        """Takes a time interval and returns an array of times with a minimum of 50ns spacing"""
        # time_interval from PL is in microseconds, we convert to ns
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

    def _convert_to_time_series(self, pulse_parameter, time_points, scaling_factor=1):
        """Converts pulse information into a TimeSeries

        Args:
            pulse_parameter(Union[float, Callable]): a physical parameter (pulse, amplitude or detuning) of the
                pulse. If this is a callalbe, it has already been partially evaluated, such that it is only a
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
            vals = [float(pulse_parameter(t * 1e6)) * scaling_factor for t in time_points]
        else:
            vals = [pulse_parameter for t in time_points]

        for t, v in zip(time_points, vals):
            ts.put(t, v)

        return ts

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
        amplitude = self._convert_to_time_series(
            pulse.amplitude, time_points, scaling_factor=2 * np.pi * 1e6
        )
        detuning = self._convert_to_time_series(
            pulse.detuning, time_points, scaling_factor=2 * np.pi * 1e6
        )
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
        1 --> 1: Atom initialized, measured in ground state (atom in ground state detected both before and after)
        """

        # if entire measurement failed, all NaN
        if not res.status.value.lower() == "success":
            return np.array([np.NaN for i in res.pre_sequence])

        # if a single atom failed to initialize, NaN for that individual measurement
        pre_sequence = [i if i else np.NaN for i in res.pre_sequence]

        # set entry to 0 if ground state is measured, 1 if excited state is measured, NaN if measurement failed
        return np.array(pre_sequence - res.post_sequence)


class BraketAquilaDevice(BraketAhsDevice):
    """Amazon Braket AHS device on QuEra Aquila hardware for PennyLane.

    Args:
        wires (int or Iterable[Number, str]]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers
            (i.e., ``[-1, 0, 2]``) or strings (``['ancilla', 'q1', 'q2']``).
        shots (int): Number of executions to run to aquire measurements. Defaults to 100.

    """

    name = "Braket QuEra Aquila PennyLane plugin"
    short_name = "braket.aws.aquila"

    ARN_NR = "arn:aws:braket:us-east-1::device/qpu/quera/Aquila"

    def __init__(self, wires: Union[int, Iterable], *, shots=100):
        device = AwsDevice(self.ARN_NR)
        super().__init__(wires=wires, device=device, shots=shots)

        self.ahs_program = None
        self.samples = None

    @property
    def hardware_capabilities(self):
        """Dictionary of hardware capabilities for the Aquila device"""
        return self._device.properties.paradigm.dict()

    def _run_task(self, ahs_program):
        discretized_ahs_program = ahs_program.discretize(self._device)
        task = self._device.run(discretized_ahs_program, shots=self.shots)
        return task


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

    def __init__(self, wires: Union[int, Iterable], *, shots=100):
        device = LocalSimulator("braket_ahs")
        super().__init__(wires=wires, device=device, shots=shots)

    def _run_task(self, ahs_program):
        task = self._device.run(ahs_program, shots=self.shots, steps=100)
        return task