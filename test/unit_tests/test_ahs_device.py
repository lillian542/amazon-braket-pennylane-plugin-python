import json
from typing import Any, Dict, Optional
from unittest import mock
from unittest.mock import Mock, PropertyMock, patch

import braket.ir as ir
import numpy as anp
import pennylane as qml
import pytest

from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
from braket.timings.time_series import TimeSeries
from braket.ahs.driving_field import DrivingField
from pennylane.pulse.rydberg_hamiltonian import RydbergHamiltonian, RydbergPulse
from dataclasses import dataclass
from braket.tasks.local_quantum_task import LocalQuantumTask
from braket.tasks.analog_hamiltonian_simulation_quantum_task_result import ShotResult

from braket.aws import AwsDevice, AwsDeviceType, AwsQuantumTask, AwsQuantumTaskBatch
from braket.circuits import Circuit, FreeParameter, Gate, Noise, Observable, result_types
from braket.circuits.noise_model import GateCriteria, NoiseModel, NoiseModelInstruction
from braket.device_schema import DeviceActionType
from braket.device_schema.openqasm_device_action_properties import OpenQASMDeviceActionProperties
from braket.device_schema.simulators import GateModelSimulatorDeviceCapabilities
from braket.devices import LocalSimulator
from braket.simulator import BraketSimulator
from braket.task_result import GateModelTaskResult
from braket.tasks import GateModelQuantumTaskResult
from pennylane import QuantumFunctionError, QubitDevice
from pennylane import numpy as np
from pennylane.tape import QuantumTape

import braket.pennylane_plugin.braket_device
from braket.pennylane_plugin import BraketAwsQubitDevice, BraketLocalQubitDevice, __version__
from braket.pennylane_plugin.braket_device import BraketQubitDevice, Shots


coordinates1 = [[0, 0], [0, 5], [5, 0], [10, 5], [5, 10], [10, 10]]
wires1 = [1, 6, 0, 2, 4, 3]

coordinates2 = [[0, 0], [5.5, 0.0], [2.75, 4.763139720814412]]
H_i = rydberg_interaction(coordinates2)


def f1(p, t):
    return p * np.sin(t) * (t - 1)


def f2(p, t):
    return p[0] * np.cos(p[1] * t**2)


# amplitude function must be 0 at start and end for hardware
def amp(p, t):
    f = p[0] * jnp.exp(-(t-p[1])**2/(2*p[2]**2))
    return qml.pulse.rect(f, windows=[0.1, 1.7])


params1 = 1.2
params2 = [3.4, 5.6]
params_amp = [2.5, 0.9, 0.3]

DEV_ATTRIBUTES = [(BraketAquilaDevice, "Aquila", "braket.aws.aquila"),
                  (BraketLocalQubitDevice, "RydbergAtomSimulator", "braket.local.aquila")]

dev_hw = BraketAquilaDevice(wires=3)
dev_sim = BraketLocalQubitDevice(wires=3, shots=17)


def dummy_ahs_program():

    # amplutide 10 for full duration
    amplitude = TimeSeries()
    amplitude.put(0, 10)
    amplitude.put(4e-6, 10)

    # phase and detuning 0 for full duration
    phi = TimeSeries().put(0, 0).put(4e-6, 0)
    detuning = TimeSeries().put(0, 0).put(4e-6, 0)

    # Hamiltonian
    H = DrivingField(amplitude, phi, detuning)

    # register
    register = AtomArrangement()
    for [x, y] in coordinates2:
        register.add([x*1e-6, y*1e-6])

    ahs_program = AnalogHamiltonianSimulation(
        hamiltonian=H,
        register=register
    )

    return ahs_program

# dummy data classes for testing result processing
@dataclass
class Status:
    value: str


@dataclass
class DummyMeasurementResult:
    status: Status
    pre_sequence: np.array
    post_sequence: np.array


DUMMY_RESULTS = [(DummyMeasurementResult(Status('Success'), np.array([1]), np.array([1])), np.array([0])),
           (DummyMeasurementResult(Status('Success'), np.array([1]), np.array([0])), np.array([1])),
           (DummyMeasurementResult(Status('Success'), np.array([0]), np.array([0])), np.array([np.NaN])),
           (DummyMeasurementResult(Status('Failure'), np.array([1]), np.array([1])), np.array([np.NaN])),
           (DummyMeasurementResult(Status('Success'), np.array([1, 1, 0]), np.array([1, 0, 0])), np.array([0, 1, np.NaN])),
           (DummyMeasurementResult(Status('Success'), np.array([1, 1]), np.array([0, 0])), np.array([1, 1])),
           (DummyMeasurementResult(Status('Success'), np.array([0, 1]), np.array([0, 0])), np.array([np.NaN, 1])),
           (DummyMeasurementResult(Status('Failure'), np.array([1, 1]), np.array([1, 1])), np.array([np.NaN, np.NaN]))
           ]


class TestBraketAhsDevice:
    """Tests that behaviour defined for both the LocalSimulator and the
    Aquila hardware in the base device work as expected"""

    @pytest.mark.parametrize("dev_cls, device_name", "short_name", DEV_ATTRIBUTES)
    def test_initialization(self, dev_cls, name, short_name):
        """Test the device initializes with the expected attributes"""

        dev = dev_cls(wires=3)

        assert dev._device.name == name
        assert dev.short_name == short_name
        assert dev.shots == 100
        assert dev.ahs_program is None
        assert dev.samples is None
        assert dev.pennylane_requires == ">=0.29.0"
        assert dev.operations == {"ParametrizedEvolution"}

    @pytest.mark.parametrize("dev_cls, shots", [(BraketAquilaDevice, 1000),
                                                (BraketAquilaDevice, 2),
                                                (BraketLocalQubitDevice, 1000),
                                                (BraketLocalQubitDevice, 2)])
    def test_setting_shots(self, dev_cls, shots):
        """Test that setting shots changes number of shots from default (100)"""
        dev = dev_cls(wires=3, shots=shots)
        assert dev.shots == shots

        if dev_cls == BraketLocalQubitDevice:
            global_drive = rydberg_drive(amp, f1, 2, wires=[0, 1, 2])
            ts = jnp.array([0.0, 1.75])
            params = [params_amp, params1]

            @qml.qnode(dev)
            def circuit(p):
                qml.evolve(Hd + global_drive)(p, ts)
                return qml.sample()

            assert len(circuit(params)) == shots

    @pytest.mark.parametrize("dev_cls, wires", [(BraketAquilaDevice, 2),
                                                (BraketAquilaDevice, [0, 2, 4]),
                                                (BraketLocalQubitDevice, [0, 'a', 7]),
                                                (BraketLocalQubitDevice, 7)])
    def test_setting_wires(self, dev_cls, wires):
        """Test setting wires"""
        dev = dev_cls(wires=wires)

        if wires is int:
            assert len(dev.wires) == wires
            assert wires.labels == tuple(i for i in range(wires))
        else:
            assert len(wires) == len(dev.wires)
            assert wires.labels == tupe(wires)

    def test_apply(self, dev_cls, operations):
        """Test that apply creates and saves an ahs_program and samples as expected"""
        pass

    def test_create_ahs_program(self, dev_cls, evolution):
        """Test creating an AnalogueHamiltonianSimulation from an evolution operator"""
        pass

    def test_generate_samples(self):
        """Test that generate_samples creates a list of arrays with the expected shape for the task run"""
        ahs_program = dummy_ahs_program()

        # correspondance between number of device wires and coordinates is checked in PL when creating the Hamiltonian
        # since this is done manually for the unit test, we confirm the values used for the test are valid here
        assert len(ahs_program.register.coordinate_list(0)) == dev_sim.wires

        task = dev_sim._run_task(ahs_program)

        dev_sim.samples = task.result()

        samples = dev_sim.generate_samples()

        assert len(samples) == 17
        assert len(samples[0]) == len(dev_sim.wires)
        assert samples[0] == np.zeros(len(dev_sim.wires))
        assert isinstance(samples[0], np.array)

    @pytest.mark.parametrize("dev", [dev_hw, dev_sim])
    def test_validate_operations_multiple_operators(self, dev):
        """Test that an error is raised if there are multiple operators"""

        H1 = rydberg_drive(amp, f1, 2, wires=[0, 1, 2])
        op1 = qml.evolve(H_i + H1)
        op2 = qml.evolve(H_i + H1)

        with pytest.warns("Support for multiple ParametrizedEvolution operators"):
            dev._validate_operations([op1, op2])

    @pytest.mark.parametrize("dev", [dev_hw, dev_sim])
    def test_validate_operations_not_rydberg_hamiltonian(self, dev):
        """Test that an error is raised if the ParametrizedHamiltonian on the operator
        is not a RydbergHamiltonian and so does not contain pulse upload information"""

        H1 = 2 * qml.PauliX(0) + f1 * qml.PauliY(1)
        op1 = qml.evolve(H1)

        with pytest.warns("Expected a RydbergHamiltonian instance"):
            dev._validate_operations([op1])

    @pytest.mark.parametrize("dev, coordinates", [(dev_hw, coordinates1),
                                     (dev_hw, coordinates2),
                                     (dev_sim, coordinates1),
                                     (dev_sim, coordinates2)])
    def test_create_register(self, dev, coordinates):
        """Test that an AtomArrangement with the expected coordinates is created
        and stored on the device"""
        reg = dev._create_register(coordinates)

        coordinates_from_reg = [[x*1e6, y*1e6] for x, y in zip(dev.register.coordinate_list(0), dev.register.coordinate_list(1))]

        assert isinstance(reg, AtomArrangement)
        assert dev.register == reg
        assert coordinates_from_reg == coordinates

    def test_evaluate_pulses(self, dev_cls):
        """Test that the callables describing pulses are partially evaluated"""
        pass

    @pytest.mark.parametrize("time_interval", [[1.5, 2.3], [0, 1.2], [0.111, 3.789]])
    def test_get_sample_times(self, time_interval):
        """Tests turning an array of [start, end] times into time set-points"""

        times = dev_sim._get_sample_times()

        num_points = len(times)
        diffs = [times[i]-times[i-1] for i in range(1, num_points)]

        # start and end times match but are in units of s and ns respectively
        assert times[0] == time_interval[0]*1e-9
        assert times[-1] == time_interval[1]*1e-9

        # distances between points are close to but exceed 50ns
        assert all(d > 50e-9 for d in diffs)
        assert np.allclose(diffs, 50e-9, atol=5e-9)

    def test_convert_to_time_series(self, dev_cls):
        """Test creating a TimeSeries from pulse information and time set-points"""
        pass

    def test_convert_pulse_to_driving_field(self, dev_cls):
        """Test that a pulse description (float or partially evaluated callable)
        and array of time setpoints can be converted into a DrivingField"""
        pass

    @pytest.mark.parametrize("res, expected_output", DUMMY_RESULTS)
    def test_result_to_sample_output(self, res, expected_output):
        """Test function for converting the task results as returned by the
        device into sample measurement results for PennyLane"""

        output = dev_sim._result_to_sample_output(res)

        assert isinstance(output, np.array)
        assert len(output) == len(res.post_sequence)
        assert output == expected_output


class TestBraketAquilaDevice:
    """Test functionality specific to the hardware device that can be tested
    without running a task on the hardware"""

    def test_hardware_capabilities(self):
        """Test hardware capabilities can be retrieved"""

        assert isinstance(dev.hardware_capabilities, dict)
        assert 'rydberg' in dev.hardware_capabilities.keys()
        assert 'lattice' in dev.hardware_capabilities.keys()

    def test_validate_operations_multiple_drive_terms(self):
        """Test that an error is raised if there are multiple drive terms on
        the Hamiltonian"""
        pulses = [RydbergPulse(3, 4, 5, [0, 1]), RydbergPulse(4, 6, 7, [1, 2])]

        with pytest.warns("Multiple pulses in a Rydberg Hamiltonian are not currently supported"):
            dev_hw._validate_pulses(pulses)

    @pytest.mark.parametrize("wires", [[0, 1, 2], [5, 6, 7, 8, 9], [0, 1, 2, 3, 6]])
    def test_validate_pulse_is_global_drive(self, wires):
        """Test that an error is raised if the pulse does not describe a global drive"""

        pulse = RydbergPulse(3, 4, 5, wires)

        with pytest.warns("Only global drive is currently supported on hardware"):
            dev_hw._validate_pulses([pulse])


class TestLocalAquilaDevice:
    """Test functionality specific to the local simulator device"""

    def test_validate_operations_multiple_drive_terms(self):
        """Test that an error is raised if there are multiple drive terms on
        the Hamiltonian"""
        pulses = [RydbergPulse(3, 4, 5, [0, 1]), RydbergPulse(4, 6, 7, [1, 2])]

        with pytest.warns("Multiple pulses in a Rydberg Hamiltonian are not currently supported"):
            dev_sim._validate_pulses(pulses)

    @pytest.mark.parametrize("wires", [[0, 1, 2], [5, 6, 7, 8, 9], [0, 1, 2, 3, 6]])
    def test_validate_pulse_is_global_drive(self, wires):
        """Test that an error is raised if the pulse does not describe a global drive"""

        pulse = RydbergPulse(3, 4, 5, wires)

        with pytest.warns("Only global drive is currently supported"):
            dev_sim._validate_pulses([pulse])

    def test_run_task(self):
        ahs_program = dummy_ahs_program()

        task = dev_sim._run_task(ahs_program)

        assert isinstance(task, LocalQuantumTask)
        assert len(task.result().measurements) == 17  # dev_sim takes 17 shots
        assert isinstance(task.result().measurements[0], ShotResult)
